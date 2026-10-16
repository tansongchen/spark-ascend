"""Keep Q in L0A for the whole QK phase instead of loading it for every sub-tile."""
import argparse,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_bulk.py')),'--output',a.output],check=True)
path=out/'decode_layer_stage_prefetch.asc';s=path.read_text();st=s.index('namespace FullStage {');en=s.index('namespace SwaStage {',st);f=s[st:en]
old='fa_base_matmul::LoadDataToL0A(a0,ql,param,0,D,4);';assert f.count(old)==1
f=f.replace(old,'if(tile==0&&sub==0)fa_base_matmul::LoadDataToL0A(a0,ql,param,0,D,4);')
s=s[:st]+f+s[en:];path.write_text(s)
m=json.loads((out/'execution.json').read_text());m['execution']='two_pass_bulk_reuse_q';m['details']+=' Q L0A loaded once for QK phase.';(out/'execution.json').write_text(json.dumps(m,indent=2)+'\n')
