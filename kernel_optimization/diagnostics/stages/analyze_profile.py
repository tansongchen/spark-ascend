"""Summarize raw device counters from validate.py --debug workspace captures."""
import argparse
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('capture',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
w=torch.load(a.capture,weights_only=True)['workspace']
t=w[124416:128000].view(torch.int64).reshape(56,8)
d=t[:,1:]-t[:,:-1]
if not bool((d>=0).all()):
    raise RuntimeError('Missing or nonmonotonic timestamps')
names=['qkv_rope_cache_sync','attention_sync','merge_gate_sync','out_sync','up_gelu_sync','down_sync','final_norm']
out={'counter_units':'raw GetSystemCycle ticks; not converted to microseconds',
     'includes_existing_barrier_waits':True,'instrumented_not_production':True,
     'per_core_ticks':d.tolist(),
     'core0_fraction':{name:float(d[0,i]/d[0].sum()) for i,name in enumerate(names)},
     'stages':{name:{'core0_ticks':int(d[0,i]),'median_ticks':float(d[:,i].float().median()),
                     'max_ticks':int(d[:,i].max())} for i,name in enumerate(names)}}
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='per_core_ticks'},indent=2))
