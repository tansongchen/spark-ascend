"""真实权重单层算子 benchmark；先验证，再使用已提交的 NPU event 计时。"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.kernel_catalog import KERNELS


def check_kv_codec(a):
    import torch
    import torch_npu  # noqa: F401
    from utils.validate_kernel import stats
    torch.set_num_threads(8);torch.manual_seed(a.seed);torch.npu.set_device(0)
    n=65536
    src=(torch.randn(n)*6).bfloat16()
    src[:24]=torch.tensor([-65504,-512,-448,-384,-240,-16,-1,-.001953125,-0.,0.,
                          .001953125,.01,1,7.5,15,240,384,448,512,65504,
                          -1e-20,1e-20,-.015625,.015625]).bfloat16()
    expected=src.float().clamp(-448,448).to(torch.float8_e4m3fn)
    x=src.npu();encoded=torch.full((n+128,),0xa5,dtype=torch.uint8,device='npu')
    decoded=torch.full((n+128,),float('nan'),dtype=torch.bfloat16,device='npu')
    lib=ctypes.CDLL(str(a.build/KERNELS[a.kernel].library));fn=lib.launch_reborn_kv8_codec
    fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2;fn.restype=None
    stream=torch.npu.current_stream();stream.synchronize()
    fn(stream.npu_stream,x.data_ptr(),encoded.data_ptr(),n,0)
    fn(stream.npu_stream,encoded.data_ptr(),decoded.data_ptr(),n,1);stream.synchronize()
    q,y=encoded.cpu(),decoded.cpu()
    checks={'encoded_bit_exact':torch.equal(q[:n],expected.view(torch.uint8)),
            'decoded_bit_exact':torch.equal(y[:n].view(torch.int16),expected.bfloat16().view(torch.int16)),
            'encoded_guard':bool((q[n:]==0xa5).all()),'decoded_guard':bool(y[n:].isnan().all())}
    result={'checks':checks,'pass':all(checks.values()),'values':n,'physical_npu':a.npu,
            'quantization':stats(y[:n],src),'format':'fp8_e4m3fn','saturate':True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['pass'] else 1


def measure_linear(a):
    """Screen a new GEMV primitive before integrating layer scheduling."""
    import torch
    import torch_npu  # noqa: F401
    from safetensors import safe_open
    from utils.w4a16_packing import unpack_checkpoint, pack_outer
    from utils.validate_kernel import stats
    torch.set_num_threads(8)
    torch.manual_seed(a.seed)
    torch.npu.set_device(0)
    model = a.model or Path('/data1/t00995472/models/Spark-X2.5-4B-W4A16-G128')
    index = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    names = {'qkv':['self_attn.q_k_v_proj'], 'out':['self_attn.out_proj'],
             'gate_up':['mlp.gate_proj','mlp.up_proj'], 'down':['mlp.down_proj'],
             'lm_head':[]}[a.linear_only]
    qs, ss = [], []
    def read(key):
        with safe_open(model/index[key], framework='pt') as f:
            return f.get_tensor(key)
    for name in names:
        key = f'model.layers.{a.layer}.{name}'
        q,s = unpack_checkpoint(read(key+'.weight'),read(key+'.weight_scale'))
        qs.append(q); ss.append(s)
    original = None
    if a.linear_only == 'lm_head':
        from utils.quantize_spark_w4a16 import quantize
        original = read('model.embedding.weight')
        words,scales,weight_error = quantize(original,128)
        q,s = unpack_checkpoint(words,scales)
    else:
        q,s = torch.cat(qs),torch.cat(ss)
    n,k = q.shape
    groups = 8 if k == 4096 else 10
    splits = k//(groups*128)
    x = torch.randn(k).bfloat16()
    w = (q.float()*s.float().repeat_interleave(128,1)).bfloat16()
    ref = torch.mv(w.float(),x.float())
    packed = pack_outer(q,s)
    xd = x.npu()
    weights = [packed.npu() for _ in range(a.weight_rotations)]
    head = a.linear_only == 'lm_head'
    y = torch.full((n+128,) if head else (n*splits+128,),float('nan'),device='npu',
                   dtype=torch.bfloat16 if head else torch.float32)
    partial = torch.empty(n*splits,device='npu',dtype=torch.float32) if head else None
    libpath = a.build/KERNELS[a.kernel].library
    lib = ctypes.CDLL(str(libpath))
    fn = lib.launch_w4a16_reborn_head if head else lib.launch_w4a16_reborn_linear
    fn.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int] if head else [ctypes.c_void_p]*4+[ctypes.c_int]*3
    fn.restype = None
    stream = torch.npu.current_stream()
    kind = {2560:0,4096:1,10240:2}[k]
    iteration = 0
    def launch():
        nonlocal iteration
        if head:
            fn(stream.npu_stream,xd.data_ptr(),weights[iteration%len(weights)].data_ptr(),
               partial.data_ptr(),y.data_ptr(),n)
        else:
            fn(stream.npu_stream,xd.data_ptr(),weights[iteration%len(weights)].data_ptr(),
               y.data_ptr(),n,kind,a.cores)
        iteration += 1
    stream.synchronize()
    launch(); stream.synchronize()
    actual = y[:n].cpu() if head else y[:n*splits].reshape(n//128,splits,128).sum(1).flatten().cpu()
    accuracy = stats(actual,ref)
    guard = bool(y[n if head else n*splits:].isnan().all())
    if not accuracy['finite'] or not guard:
        raise RuntimeError(f'Linear output failed: {accuracy}, guard={guard}')
    for _ in range(a.warmup): launch()
    stream.synchronize()
    observations=[]
    for _ in range(a.runs):
        start,end = torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        start.record(); stream.synchronize()
        for _ in range(a.iterations): launch()
        end.record();end.synchronize()
        observations.append(start.elapsed_time(end)*1000/a.iterations)
    result={'kernel':a.kernel,'matrix':a.linear_only,'n':n,'k':k,'splits':splits,
            'physical_npu':a.npu,'cores':56 if head else a.cores,'layer':a.layer,'accuracy':accuracy,
            'guard':guard,'mean_us':statistics.mean(observations),'observations_us':observations,
            'packed_bytes':packed.numel(),'rotating_weights':len(weights),
            'library_sha256':hashlib.sha256(libpath.read_bytes()).hexdigest(),
            'scope':'linear partial sums only; excludes final K merge and full-model overhead'}
    if head:
        result['scope'] = 'complete W4A16 decode head including final K merge; constructed BF16 activation'
        result['head_quantization'] = {'relative_weight_l2':weight_error,
            'logits_against_original_bf16':stats(actual,torch.mv(original.float(),x.float())),
            'quality_benchmark':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if accuracy['pass'] else 1


def measure(context):
    import torch
    from utils.validate_kernel import stats
    from utils.packing import tile_weights
    c, a = context, context.args
    stream = torch.npu.current_stream()
    launches = {'candidate': c.fn}
    pointers = {'candidate': c.ptrs}
    baseline_tensors = []
    if a.baseline_library:
        baseline_lib = ctypes.CDLL(str(Path(a.baseline_library).resolve()))
        baseline = baseline_lib.launch_decode_layer_stage_prefetch
        baseline.argtypes, baseline.restype = c.fn.argtypes, None
        baseline_tensors = [tile_weights(c.w1, 8).npu(), tile_weights(c.w['mlp.down_proj.weight'], 4).npu()]
        manifest = Path(a.baseline_library).parent/'execution.json'
        if manifest.exists() and json.loads(manifest.read_text()).get('weight_encoding') == 'bf16_lossless12':
            from utils.bf16_codec import pack_weight
            baseline_tensors = [pack_weight(tile_weights(c.w1,8),2560,8).npu(),
                                pack_weight(tile_weights(c.w['mlp.down_proj.weight'],4),10240,4).npu()]
        base_ptr = (ctypes.c_void_p*14)(*[t.data_ptr() for t in c.dev])
        base_ptr[4], base_ptr[5] = [t.data_ptr() for t in baseline_tensors]
        launches.update(baseline=baseline, null=baseline)
        pointers.update(baseline=base_ptr, null=base_ptr)
    # Keep all tensors alive until every asynchronous launch has completed.
    banks = [c.caches] + [[t.clone() for t in c.caches] for _ in range(a.cache_banks-1)]
    weight_banks, ptr_banks = {}, {}
    for name in launches:
        tensors = list(c.dev)
        if name != 'candidate':
            tensors[4:6] = baseline_tensors
        weight_banks[name] = [[tensors[j].clone() for j in (2,3,4,5)] for _ in range(a.weight_rotations-1)]
        ptr_banks[name] = [pointers[name]]
        for weights in weight_banks[name]:
            pp = (ctypes.c_void_p*14)(*[t.data_ptr() for t in tensors])
            for j, tensor in zip((2,3,4,5), weights): pp[j] = tensor.data_ptr()
            ptr_banks[name].append(pp)
    stream.synchronize()
    counts = dict.fromkeys(launches, 0)
    def launch(name):
        i = counts[name]
        counts[name] += 1
        launches[name](stream.npu_stream, ptr_banks[name][i % a.weight_rotations],
                       *[t.data_ptr() for t in banks[i % a.cache_banks]],
                       c.ws.data_ptr(), c.y.data_ptr(), c.rout.data_ptr(),
                       c.rd, int(a.slot32), 1e-6, 1e-6)
    comparisons = {}
    if a.baseline_library:
        snapshots = {}
        for name in ('baseline', 'candidate'):
            launch(name); stream.synchronize()
            snapshots[name] = {'y': c.y[:2560].cpu(), 'resout': c.rout[:2560].cpu()}
        comparisons = {name: {'bit_equal': torch.equal(snapshots['baseline'][name], snapshots['candidate'][name]),
                              **stats(snapshots['candidate'][name], snapshots['baseline'][name])}
                       for name in ('y', 'resout')}
        if not all(row['pass'] for row in comparisons.values()):
            raise RuntimeError(f'候选与基线比较未通过，停止计时：{comparisons}')
    for name in launches:
        for _ in range(a.warmup): launch(name)
    stream.synchronize()
    if a.profile:
        from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        from torch_npu.profiler import _ExperimentalConfig, ProfilerLevel, AiCMetrics
        with profile(activities=[ProfilerActivity.NPU], on_trace_ready=tensorboard_trace_handler(a.profile),
                     experimental_config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
                                                              aic_metrics=AiCMetrics.PipeUtilization)):
            for name in launches:
                for _ in range(3): launch(name)
                stream.synchronize()
    observations = []
    for i in range(a.runs):
        order = (['baseline','null','candidate'] if i % 2 == 0 else ['candidate','null','baseline']) if a.baseline_library else ['candidate']
        row = {'round': i, 'order': order}
        for name in order:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record(); stream.synchronize()
            for _ in range(a.iterations): launch(name)
            end.record(); end.synchronize()
            row[name+'_us'] = start.elapsed_time(end)*1000/a.iterations
        observations.append(row)
    means = {name: statistics.mean(row[name+'_us'] for row in observations) for name in launches}
    result = {'kernel': a.kernel, 'physical_npu': a.npu, 'layer': a.layer, 'length': a.length,
              'mean_us': means, 'warmup': a.warmup, 'iterations': a.iterations,
              'observations': observations, 'comparisons': comparisons,
              'candidate_library': str(c.libpath), 'baseline_library': a.baseline_library,
              'library_sha256': hashlib.sha256(c.libpath.read_bytes()).hexdigest(),
              'cache_banks': a.cache_banks, 'rotating_weight_allocations': a.weight_rotations,
              'up_tile': a.up_tile, 'down_tile': a.down_tile,
              'event_start_submitted_before_raw_launch': True,
              'scope': 'single real-weight layer; includes submission gaps; screening only, not full-model TPOT',
              'profile_collected': bool(a.profile)}
    if a.baseline_library:
        result['baseline_library_sha256'] = hashlib.sha256(Path(a.baseline_library).read_bytes()).hexdigest()
        result['candidate_reduction_percent'] = 100*(1-means['candidate']/means['baseline'])
        result['null_reduction_percent'] = 100*(1-means['null']/means['baseline'])
    Path(str(a.output)+'.benchmark.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel', choices=KERNELS, default='stage_prefetch')
    p.add_argument('--build', type=Path, required=True, help='共享库及 execution.json 所在目录')
    p.add_argument('--npu', type=int, choices=range(1,8), default=2)
    p.add_argument('--model', type=Path, help='真实权重模型；W4A16 默认选量化模型')
    p.add_argument('--layer', type=int, choices=range(36), default=0)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--slot32', action='store_true')
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--repeat', type=int, default=3, help='计时前正确性验证次数，1..10')
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--iterations', type=int, default=250)
    p.add_argument('--runs', type=int, default=8)
    p.add_argument('--baseline-library', help='可选 BF16 配对基线，共用 QKV/O 布局')
    p.add_argument('--weight-rotations', type=int, default=1)
    p.add_argument('--cache-banks', type=int, default=1)
    p.add_argument('--profile', help='可选独立 profiler 采集目录')
    p.add_argument('--linear-only', choices=['qkv','out','gate_up','down','lm_head'], help='验证独立新线性层；lm_head 包含最终合并')
    p.add_argument('--cores', type=int, default=56, help='独立线性层原型的 AIV 数')
    p.add_argument('--kv-codec-only', action='store_true', help='独立检查 FP8 KV 编码、解码和保护区')
    p.add_argument('--health-interval', type=float, default=0, help='设备采样间隔秒；0 关闭')
    p.add_argument('--debug')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    a.up_tile, a.down_tile, a.transpose_down = 8, 4, False
    if a.kernel == 'w4a16_reborn':
        a.up_tile = a.down_tile = 128
    if not 1 <= a.length <= 40960 or not 1 <= a.repeat <= 10:
        p.error('length 应为 1..40960，repeat 应为 1..10')
    if a.warmup < 0 or min(a.iterations,a.runs,a.weight_rotations,a.cache_banks) < 1:
        p.error('预热次数应非负，其余计时参数应为正数')
    a.variant, a.stop = KERNELS[a.kernel].variant, 'full'
    manifest_path = a.build/'execution.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get('kernel', a.kernel) != a.kernel:
        p.error('--kernel 与构建 manifest 不一致')
    if a.kernel == 'lossless_prefetch' and manifest.get('weight_encoding') != 'bf16_lossless12':
        p.error('无损 kernel 必须携带 weight_encoding=bf16_lossless12 的 execution.json')
    if a.variant == 'w4a16' and manifest.get('weight_encoding') != KERNELS[a.kernel].manifest['weight_encoding']:
        p.error('W4A16 kernel 必须携带匹配权重布局的 execution.json')
    if a.baseline_library and a.variant != 'stage_prefetch':
        p.error('配对基线仅支持共用 QKV/O 布局的 stage_prefetch ABI')
    if not (a.build/KERNELS[a.kernel].library).is_file():
        p.error(f'共享库不存在：{a.build/KERNELS[a.kernel].library}')
    from utils.device import select_idle_npu
    select_idle_npu(a.npu)
    if a.kv_codec_only:
        return check_kv_codec(a)
    if a.linear_only:
        if not (manifest.get('linear_probe') or manifest.get('linear_probe_only')):
            p.error('--linear-only 需要包含独立线性层入口的构建')
        return measure_linear(a)
    if manifest.get('linear_probe_only'):
        p.error('该构建尚为独立线性层原型，需要 --linear-only')
    from utils import validate_kernel as v
    from utils.telemetry import record_device
    if a.variant == 'w4a16':
        from utils.w4a16_reference import make_reference
        model = a.model or Path('/data1/t00995472/models/Spark-X2.5-4B-W4A16-G128')
        loader, packer = make_reference(model, manifest['weight_encoding'])
        with record_device(a.npu, str(a.output)+'.health.json', a.health_interval):
            return v.run(a, weights_loader=loader, input_packer=packer, after_validation=measure)
    if a.model: v.MODEL = a.model
    with record_device(a.npu, str(a.output)+'.health.json', a.health_interval):
        return v.run(a, after_validation=measure)


if __name__ == '__main__':
    raise SystemExit(main())
