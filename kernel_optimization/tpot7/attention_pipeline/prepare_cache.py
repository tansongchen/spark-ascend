"""Test KV L2 bypass on the balanced7_tile384 original pipeline."""
import argparse,subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
subprocess.run([sys.executable,str(ROOT/'kernel_optimization/attention/prepare.py'),'--candidate','balanced7_tile384','--output',a.output],check=True)
path=out/'decode_layer_stage_prefetch.asc';s=path.read_text();st=s.index('namespace FullStage {');en=s.index('namespace SwaStage {',st);f=s[st:en]
needle='v.SetGlobalBuffer((__gm__ bfloat16_t*)vp);';assert f.count(needle)==1
f=f.replace(needle,needle+'k.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);v.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);')
s=s[:st]+f+s[en:];path.write_text(s)
m=json.loads((out/'execution.json').read_text());m['execution']='tile384_kv_l2_disable';m['details']='Original balanced7_tile384 pipeline, only global K/V L2 hint disabled; ABI unchanged.';(out/'execution.json').write_text(json.dumps(m,indent=2)+'\n')
