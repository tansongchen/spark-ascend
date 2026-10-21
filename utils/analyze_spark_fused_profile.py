"""Reconcile the independent fused decode capture without timing claims."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
a=p.parse_args()
paths=list(a.directory.rglob('kernel_details.csv'))
if len(paths)!=1:raise RuntimeError(f'Expected one capture, found {len(paths)}')
rows=list(csv.DictReader(paths[0].open()))
rows.sort(key=lambda r:float(r['Start Time(us)'].strip()))
counts=collections.Counter(r['Type'] for r in rows)
expected={'decode_layer_stage_prefetch':576,'RmsNorm':16,'MatMulV3':16,'ArgMaxV2':16}
if any(counts[k]!=v for k,v in expected.items()):raise RuntimeError(f'Incomplete capture: {counts}')
steps=[];pending=[]
for row in rows:
    pending.append(row)
    if row['Type']=='ArgMaxV2':
        fused=[r for r in pending if r['Type']=='decode_layer_stage_prefetch']
        if len(fused)!=36:raise RuntimeError('Step does not contain exactly 36 fused layers')
        steps.append(pending);pending=[]
if len(steps)!=16 or any(r['Type'] in expected for r in pending):
    raise RuntimeError('Unexpected partial model steps')
# The sampler emits two dtype-conversion tasks after the final ArgMax.
tail_types=[r['Type'] for r in pending]
steps[-1].extend(pending)
fusion=[r for r in rows if r['Type']=='decode_layer_stage_prefetch']
swa=[float(r['Duration(us)']) for i,r in enumerate(fusion) if i%36%4!=3]
full=[float(r['Duration(us)']) for i,r in enumerate(fusion) if i%36%4==3]
result={'capture_file':str(paths[0]),'complete':True,'decode_steps':16,
        'kernels':len(rows),'kernels_per_step':[len(s) for s in steps],
        'counts':dict(counts),'fused_launches_per_step':36,
        'post_argmax_tail_types':tail_types,
        'profiled_kernel_duration_sum_per_step_ms':sum(float(r['Duration(us)']) for r in rows)/16/1000,
        'profiled_fused_duration_sum_per_step_ms':sum(float(r['Duration(us)']) for r in fusion)/16/1000,
        'profiled_fused_swa_mean_us':statistics.mean(swa),'profiled_fused_full_mean_us':statistics.mean(full),
        'timing_is_production_tpot':False}
target=a.directory.parent/'profile-analysis.json'
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
