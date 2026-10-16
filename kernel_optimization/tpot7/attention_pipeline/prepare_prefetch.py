"""Overlap the first global K tile prefetch with the unchanged QKV vector stage."""
import argparse,subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
subprocess.run([sys.executable,str(ROOT/'kernel_optimization/attention/prepare.py'),'--candidate','balanced7_tile384','--output',a.output],check=True)
path=out/'decode_layer_stage_prefetch.asc';s=path.read_text();st=s.index('namespace FullStage {');en=s.index('namespace SwaStage {',st);f=s[st:en]
wait='CrossCoreWaitFlag<4,PIPE_MTE2>(7);CrossCoreWaitFlag<4,PIPE_MTE2>(23);'
f=f.replace('if(valid==0)return;','if(valid==0){if(logical<28){'+wait+'}return;}',1)
prefetch=' Mutex::Lock<PIPE_MTE2>(4);copy_pages(kv,k,table,start,imin(N,valid),head);Mutex::Unlock<PIPE_MTE2>(4);'
assert f.count(prefetch)==1;f=f.replace(prefetch,'',1)
needle=' Mutex::Lock<PIPE_MTE2>(0);Nd2NzParams nd;';assert f.count(needle)==1
insert=''' // Prefetch only if every physically copied page precedes the current-token page.
 // The boundary is task-local and includes page rounding / repeated tail pages.
 bool early=start+((imin(N,valid)+127)/128)*128<=((int)pos.GetValue(0)/128)*128;
 if(!early&&logical<28){WAIT}
 PREFETCH
 if(early&&logical<28){WAIT}
'''.replace('WAIT',wait).replace('PREFETCH',prefetch.strip())
f=f.replace(needle,insert+needle,1);s=s[:st]+f+s[en:]
needle='\n '+wait+'\n for(int logical=GetBlockIdx();';assert s.count(needle)==1
s=s.replace(needle,'\n if(!full){'+wait+'}\n for(int logical=GetBlockIdx();',1)
path.write_text(s)
m=json.loads((out/'execution.json').read_text());m['execution']='tile384_early_k_prefetch';m['details']='First global K tile prefetch precedes QKV barrier only if its rounded final copied page is before current-token page. Original ABI/math/packing.';(out/'execution.json').write_text(json.dumps(m,indent=2)+'\n')
