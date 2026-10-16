"""Summarize completed raw results without converting microseconds to model TPOT."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('/data1/t00995472/spark-tpot7/attention_pipeline'));a=p.parse_args();out={'scope':'NPU4 single-layer screening, not full-model TPOT','micro':[],'validation':[],'artifacts':{}}
for f in sorted(a.results.glob('*.json.benchmark.json')):
 j=json.loads(f.read_text());out['micro'].append({'file':f.name,**{k:j[k] for k in ['mean_us','candidate_reduction_percent','null_reduction_percent','comparisons','scope']},'cache_banks':j.get('cache_banks',1)})
for f in sorted(a.results.glob('*.json')):
 j=json.loads(f.read_text())
 if 'stages' in j and 'pass' in j:out['validation'].append({'file':f.name,**{k:j[k] for k in ['layer','length','repeat','pass','guards']},'generated_source_sha256':j.get('generated_source_sha256')})
for name in ['v1','v2_bulk','v3_reuse','v4_cache','v5_prefetch','v6_prefetch_safe','v2_profile']:
 d=a.results/name
 if d.exists():out['artifacts'][name]={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [d/'decode_layer_stage_prefetch.asc',d/'libdecode_layer_stage_prefetch.so'] if f.exists()}
(a.results/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'validation':out['validation'],'micro':[(r['file'],r['mean_us'],r['candidate_reduction_percent']) for r in out['micro']]},indent=2))
