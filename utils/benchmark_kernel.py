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
    p.add_argument('--model', type=Path, required=True, help='显式 BF16 checkpoint；不支持官方混合 INT4/INT8')
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
    p.add_argument('--health-interval', type=float, default=0, help='设备采样间隔秒；0 关闭')
    p.add_argument('--debug')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    a.up_tile, a.down_tile, a.transpose_down = 8, 4, False
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
    if a.baseline_library and a.variant != 'stage_prefetch':
        p.error('配对基线仅支持共用 QKV/O 布局的 stage_prefetch ABI')
    if not (a.build/KERNELS[a.kernel].library).is_file():
        p.error(f'共享库不存在：{a.build/KERNELS[a.kernel].library}')
    from utils.device import select_idle_npu
    select_idle_npu(a.npu)
    from utils import validate_kernel as v
    from utils.telemetry import record_device
    if a.model: v.MODEL = a.model
    with record_device(a.npu, str(a.output)+'.health.json', a.health_interval):
        return v.run(a, after_validation=measure)


if __name__ == '__main__':
    raise SystemExit(main())
