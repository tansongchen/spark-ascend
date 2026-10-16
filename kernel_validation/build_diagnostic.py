"""Generate an early-return diagnostic copy; supplied sources remain untouched."""
import argparse
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--variant',choices=['mega','stage_prefetch'],default='stage_prefetch')
p.add_argument('--stop',choices=['qkv','attention','merge','out','up'],required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
src=(root/f'decode_layer_{a.variant}.asc').read_text()
start=src.index('extern "C" __global__')
body=src[start:]
if a.stop=='qkv':
    body=body.replace('#ifdef __DAV_C310_CUBE__\n', '#ifdef __DAV_C310_CUBE__\n return;\n',1)
    # Keep all vector blocks in the same prefix of the kernel.
    needle='SyncAll<true>();pipe.Reset();CrossCoreSetFlag<4,PIPE_MTE3>(7);'
    body=body.replace(needle,'SyncAll<true>(); return;',1)
else:
    if a.stop=='attention':
        needle='if(GetBlockIdx()<16)MergeStage::run'
    elif a.stop=='merge':
        needle='stage_barrier();pipe.Reset();int up_slot' if a.variant=='stage_prefetch' else 'SyncAll<true>();pipe.Reset();FfnStage::linear_stage'
    elif a.stop=='out':
        needle='stage_barrier();pipe.Reset();FfnStage::run' if a.variant=='stage_prefetch' else 'SyncAll<true>();pipe.Reset();FfnStage::run'
    else:
        raise ValueError('up cut not implemented')
    at=body.index(needle)
    body=body[:at]+'return;\n'+body[at:]
src=src[:start]+body
target=out/f'decode_layer_{a.variant}.asc'
target.write_text(src)
cmd=['bisheng','--npu-arch=dav-3510','-fPIC','-shared','-I'+str(root),str(target),'-o',str(out/f'libdecode_layer_{a.variant}.so'),'-lascendcl']
print(' '.join(cmd),flush=True)
subprocess.run(cmd,check=True)
