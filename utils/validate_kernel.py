"""Independent single-layer validation; no changes to vLLM or installed packages.

CPU FP32 PyTorch reference, real checkpoint weights, deterministic constructed
hidden states/history. This is not a full-model quality or speed benchmark.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import time
import sys

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.packing import make_luts, pack_qkv, tile_weights
MODEL = None
BF = torch.bfloat16


def bf(x):
    return x.to(BF)


def linear(x, w):
    return bf(F.linear(x.float(), w.float()))


def norm(x, gamma):
    x = x.float()
    return bf(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * gamma.float())


def load_weights(layer):
    if MODEL is None:
        raise ValueError('必须用 --model 指定 BF16 checkpoint')
    if json.loads((MODEL / 'config.json').read_text()).get('quantization_config'):
        raise ValueError('BF16 kernel 验证不能使用量化 checkpoint')
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'model.layers.{layer}.'
    keys = [k for k in index if k.startswith(prefix)]
    keys += [f'model.layers.{layer+1}.input_layernorm.weight'] if layer < 35 else ['model.norm.weight']
    out = {}
    for key in keys:
        with safe_open(MODEL / index[key], framework='pt', device='cpu') as f:
            out[key.removeprefix(prefix)] = f.get_tensor(key).contiguous()
    return out, keys[-1]


def rope(x, cos, sin, rd):
    z = x.float().clone()
    a, b = x[..., :rd//2].float(), x[..., rd//2:rd].float()
    z[..., :rd//2] = a*cos - b*sin
    z[..., rd//2:rd] = b*cos + a*sin
    return bf(z)


def attention(q, k, v, start, end):
    if end <= start:
        return torch.zeros((16, 256), dtype=BF), torch.full((16,), -1e30)
    kk = k[start:end].float().repeat_interleave(4, dim=1).transpose(0, 1)
    vv = v[start:end].float().repeat_interleave(4, dim=1).transpose(0, 1)
    scores = torch.einsum('hd,htd->ht', q.float(), kk) / 16
    return bf(torch.einsum('ht,htd->hd', scores.softmax(-1), vv)), scores.logsumexp(-1)


def stats(actual, ref):
    a, r = actual.float().flatten(), ref.float().flatten()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    if not finite:
        return {'finite': False, 'pass': False}
    delta = a-r
    nrmse = float(delta.norm() / r.norm().clamp_min(1e-12))
    cosine = float(F.cosine_similarity(a[None], r[None])) if r.norm() else float(a.norm() == 0)
    # Fixed engineering screen, NOT a vendor accuracy standard. Record full metrics.
    # 3% NRMSE plus 0.999 cosine catches layout/sync defects without requiring
    # bit identity across BF16 rounding and different reduction trees.
    return {'finite': True, 'max_abs': float(delta.abs().max()),
            'nrmse': nrmse, 'cosine': cosine,
            'exact_fraction': float((a == r).float().mean()),
            'pass': nrmse <= .03 and cosine >= .999}


def run(args, *, weights_loader=None, input_packer=None, after_validation=None):
    visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES','').split(',')
    if len(visible)!=1 or not visible[0].isdigit() or int(visible[0])==0:
        raise RuntimeError('Explicitly expose a checked idle physical NPU other than 0')
    libpath = Path(args.build)/f'libdecode_layer_{args.variant}.so'
    if not libpath.is_file():
        raise FileNotFoundError(f'Build the diagnostic first: {libpath}')
    manifest_path=Path(args.build)/'execution.json'
    manifest_data=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.npu.set_device(0)
    props = torch.npu.get_device_properties(0)
    print('DEVICE', props, flush=True)
    # Bound device execution time, including a possible cross-core deadlock.
    acl = ctypes.CDLL('libascendcl.so')
    acl.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
    rc = acl.aclrtSetOpExecuteTimeOut(10)
    if rc:
        raise RuntimeError(f'aclrtSetOpExecuteTimeOut: {rc}')
    layer = args.layer
    rd = 64 if (layer+1) % 4 == 0 else 256
    w, nextkey = (weights_loader or load_weights)(layer)
    qweight = w['self_attn.q_k_v_proj.weight']
    gateweight = w['self_attn.g_proj.weight']
    gatew, upw = w['mlp.gate_proj.weight'], w['mlp.up_proj.weight']
    w1 = torch.stack((gatew.reshape(-1,4,2560), upw.reshape(-1,4,2560)), dim=1).reshape(20480,2560)
    lg, ls = make_luts()
    g1 = w['post_attention_layernorm.weight']
    g2 = w[nextkey]
    residual = bf(torch.randn(2560)*.2)
    x = norm(residual, w['input_layernorm.weight'])
    pos = args.length-1
    theta = 5000000 if rd == 64 else 10000
    inv = 1. / (theta ** (torch.arange(0, rd, 2).float()/rd))
    angles = torch.arange(args.length).float()[:,None] * inv
    cs = bf(torch.cat((angles.cos(), angles.sin()), dim=-1)).contiguous()
    cos, sin = cs[pos,:rd//2].float(), cs[pos,rd//2:].float()
    rawqkv = linear(x, qweight)
    q = rope(rawqkv[:4096].reshape(16,256), cos, sin, rd)
    k = rope(rawqkv[4096:5120].reshape(4,256), cos, sin, rd)
    v = rawqkv[5120:].reshape(4,256)
    gate = linear(x, gateweight)
    # Deterministic history, with deliberately nonidentity physical page mapping.
    pages = math.ceil(args.length/128)
    table = torch.randperm(pages, dtype=torch.int32)
    khist = bf(torch.randn(pages*128,4,256)*.3)
    vhist = bf(torch.randn(pages*128,4,256)*.3)
    khist[pos], vhist[pos] = k, v
    kc, vc = torch.zeros_like(khist).reshape(pages,128,4,256), torch.zeros_like(vhist).reshape(pages,128,4,256)
    kc[table.long()] = khist.reshape(pages,128,4,256)
    vc[table.long()] = vhist.reshape(pages,128,4,256)
    def nz(t):
        return t.reshape(pages,128,4,16,16).permute(0,2,3,1,4).contiguous()
    nk, nv = nz(kc), nz(vc)
    slot = int(table[pos//128])*128 + pos%128
    # Poison current slot so a missing cache write cannot accidentally pass.
    kc.reshape(-1,4,256)[slot].fill_(float('nan'))
    vc.reshape(-1,4,256)[slot].fill_(float('nan'))
    nk.reshape(pages,4,16,128,16)[slot//128,:,:,slot%128,:].fill_(float('nan'))
    nv.reshape(pages,4,16,128,16)[slot//128,:,:,slot%128,:].fill_(float('nan'))
    inputs = [x, residual, pack_qkv(qweight,gateweight,rd), w['self_attn.out_proj.weight'],
              w1, w['mlp.down_proj.weight'], lg, ls, g1, g2,
              torch.tensor([pos],dtype=torch.int64), cs,
              torch.tensor([slot],dtype=torch.int32 if args.slot32 else torch.int64), table]
    if input_packer is not None:
        inputs = input_packer(inputs, layer, rd)
    elif args.variant=='stage_prefetch':
        inputs[2]=torch.cat((tile_weights(inputs[2][:6144],8),inputs[2][6144:]))
        inputs[3]=tile_weights(inputs[3],8)
        inputs[4]=tile_weights(inputs[4],getattr(args,'up_tile',8))
        inputs[5]=(inputs[5].T.contiguous() if getattr(args,'transpose_down',False)
                   else tile_weights(inputs[5],getattr(args,'down_tile',4)))
        if manifest_data.get('weight_encoding') == 'bf16_lossless12':
            from utils.bf16_codec import pack_weight
            inputs[4]=pack_weight(inputs[4],2560,8)
            inputs[5]=pack_weight(inputs[5],10240,4)
    dev = [t.npu() for t in inputs]
    caches = [t.npu() for t in (kc,vc,nk,nv)]
    workspace_bytes=manifest_data.get('workspace_bytes',128*1024)
    ws = torch.full((workspace_bytes+4096,), 0xa5, dtype=torch.uint8,device='npu')
    y = torch.full((2560+128,),float('nan'),dtype=BF,device='npu')
    rout = y.clone()
    variant = args.variant
    lib = ctypes.CDLL(str(libpath))
    fn = getattr(lib, f'launch_decode_layer_{variant}')
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_void_p]*7 + [ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float]
    fn.restype = None
    ptrs = (ctypes.c_void_p*14)(*[t.data_ptr() for t in dev])
    torch.npu.synchronize()
    started = time.monotonic()
    print(f'LAUNCH variant={variant} layer={layer} length={args.length} rd={rd}',flush=True)
    for iteration in range(args.repeat):
        if iteration:
            # Check persistent flags and reused allocations across launches.
            ws.fill_(0xa5);y.fill_(float('nan'));rout.fill_(float('nan'))
            torch.npu.synchronize()
        fn(torch.npu.current_stream().npu_stream, ptrs,
           *[t.data_ptr() for t in caches], ws.data_ptr(),y.data_ptr(),rout.data_ptr(),rd,int(args.slot32),1e-6,1e-6)
        torch.npu.synchronize()
    elapsed = time.monotonic()-started
    print('LAUNCH_OK',elapsed,flush=True)
    workspace = ws.cpu()
    def view(off,n,dtype=BF):
        width = torch.empty((),dtype=dtype).element_size()
        return workspace[off:off+n*width].view(dtype).clone()
    offsets = (80128,88320,93440,113920,119040) if variant == 'stage_prefetch' else (78592,86784,91904,112384,117504)
    aq,ak,av,ag = view(0,4096).reshape(16,256),view(8192,1024).reshape(4,256),view(10240,1024).reshape(4,256),view(12288,16)
    if args.debug:
        torch.save({'q':aq,'k':ak,'v':av,'gate':ag,'expected_q':q,'expected_k':k,'expected_v':v,
                    'rawqkv':rawqkv,'workspace':workspace},args.debug)
    merged,oproj,mid,raw,resmid = [view(o,n) for o,n in zip(offsets,(4096,2560,10240,2560,2560))]
    begin = max(0,pos-512) if rd==256 else 0
    katt,vatt=khist,vhist
    att,_ = attention(q,katt,vatt,begin,pos+1)
    gated = bf(att * bf(torch.sigmoid(gate.float()))[:,None]).flatten()
    op = linear(gated,w['self_attn.out_proj.weight'])
    rmid = bf(op.float()+residual.float())
    n1 = norm(op.float()+residual.float(),g1)
    m = bf(bf(F.gelu(linear(n1,gatew).float(),approximate='none')) * linear(n1,upw))
    down = linear(m,w['mlp.down_proj.weight'])
    rfinal = bf(down.float()+rmid.float())
    final = norm(down.float()+rmid.float(),g2)
    stages = {name:stats(a,r) for name,a,r in [
        ('q_rope',aq,q),('k_rope',ak,k),('v',av,v),('gate',ag,gate),
        ('attention_gate',merged,gated),('out_proj',oproj,op),('resmid',resmid,rmid),
        ('ffn_mid',mid,m),('ffn_down',raw,down),('y',y[:2560].cpu(),final),('resout',rout[:2560].cpu(),rfinal)]}
    stages['cache_k'] = stats(caches[0].reshape(-1,4,256)[slot].cpu(),k)
    stages['cache_v'] = stats(caches[1].reshape(-1,4,256)[slot].cpu(),v)
    if rd==64:
        for j,key,reference in [(2,'cache_nz_k',k),(3,'cache_nz_v',v)]:
            actual=caches[j].reshape(pages,4,16,128,16)[slot//128,:,:,slot%128,:].reshape(4,256).cpu()
            stages[key]=stats(actual,reference)
    partial = (view(131072,8*16*256,torch.float32) if manifest_data.get('attention_partials')=='float32'
               else view(12544,8*16*256)).reshape(8,16,256)
    if variant == 'stage_prefetch':
        lse = view(78080,512,torch.float32).reshape(8,8,8)[...,:2].reshape(8,16)
    else:
        lse = view(78080,128,torch.float32).reshape(8,16)
    width=manifest_data.get('full_part_width',5120) if rd==64 else manifest_data.get('swa_part_width',128)
    base=0 if rd==64 else (begin//128)*128
    for part in range(8):
        lo=max(base+part*width,begin)
        hi=min(base+(part+1)*width,pos+1)
        if hi<=lo:
            continue
        pa,pl = attention(q,katt,vatt,lo,hi)
        stages[f'partial_{part}'] = stats(partial[part],pa)
        stages[f'lse_{part}'] = stats(lse[part],pl)
    # Local references distinguish defects here from upstream error propagation.
    localop = linear(merged,w['self_attn.out_proj.weight'])
    localn = norm(oproj.float()+residual.float(),g1)
    localm = bf(bf(F.gelu(linear(localn,gatew).float())) * linear(localn,upw))
    for name,a,r in [('out_proj_local',oproj,localop),('ffn_mid_local',mid,localm),
                     ('ffn_down_local',raw,linear(mid,w['mlp.down_proj.weight'])),
                     ('norm_final_local',y[:2560].cpu(),norm(raw.float()+resmid.float(),g2))]:
        stages[name]=stats(a,r)
    guards = {'workspace_tail':bool((workspace[workspace_bytes:]==0xa5).all()),
              'y_tail':bool(y[2560:].isnan().all()),'resout_tail':bool(rout[2560:].isnan().all())}
    if args.stop=='qkv':
        stages={k:v for k,v in stages.items() if k in ('q_rope','k_rope','v','gate','cache_k','cache_v','cache_nz_k','cache_nz_v')}
    elif args.stop=='attention':
        stages={k:v for k,v in stages.items() if k in ('q_rope','k_rope','v','gate','cache_k','cache_v','cache_nz_k','cache_nz_v') or k.startswith(('partial_','lse_'))}
    source_path = Path(args.build)/manifest_data.get('source',f'decode_layer_{variant}.asc')
    if not source_path.exists():
        source_path = ROOT/'kernels'/f'decode_layer_{variant}.asc'
    result = {'variant':variant,'stop':args.stop,'layer':layer,'length':args.length,'slot32':args.slot32,'repeat':args.repeat,
              'seed':args.seed,'visible_devices':os.environ['ASCEND_RT_VISIBLE_DEVICES'],
              'device':str(props),'launch_wall_seconds_not_benchmark':elapsed,
              'reference':'CPU FP32 operations with BF16 stage boundaries; real weights, constructed activations/history',
              'screen':{'nrmse_max':.03,'cosine_min':.999,'official_standard':False},
              'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source_path,ROOT/'kernels/decode_cube_matmul.h',libpath]},
              'stages':stages,'guards':guards,
              'pass':all(s['pass'] for s in stages.values()) and all(guards.values())}
    manifest=Path(args.build)/'execution.json'
    result['execution']=json.loads(manifest.read_text()) if manifest.exists() else {'execution':'supplied_kernel' if args.stop=='full' else 'early_return_diagnostic'}
    generated=Path(args.build)/f'decode_layer_{variant}.asc'
    if generated.exists():
        result['generated_source_sha256']=hashlib.sha256(generated.read_bytes()).hexdigest()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    if result['pass'] and after_validation is not None:
        from types import SimpleNamespace
        context=SimpleNamespace(args=args,fn=fn,lib=lib,libpath=libpath,dev=dev,ptrs=ptrs,
                                caches=caches,ws=ws,y=y,rout=rout,rd=rd,w=w,w1=w1,
                                manifest=manifest_data)
        result['benchmark']=after_validation(context)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['pass'] else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['stage_prefetch'],default='stage_prefetch')
    p.add_argument('--model', type=Path, required=True, help='显式 BF16 checkpoint')
    p.add_argument('--build',default='/data1/t00995472/spark-kernel-validation/build')
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--slot32',action='store_true')
    p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--stop',choices=['full','qkv','attention'],default='full')
    p.add_argument('--debug')
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    global MODEL
    MODEL = a.model
    if not 0 <= a.layer < 36 or not 1 <= a.length <= 40960:
        p.error('Supported validation domain: layer 0..35, length 1..40960')
    if not 1 <= a.repeat <= 10:
        p.error('--repeat must be 1..10')
    return run(a)


if __name__=='__main__':
    raise SystemExit(main())
