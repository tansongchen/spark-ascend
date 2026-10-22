"""Count per-tile BF16 exponent repairs in actual MLP weights."""
import argparse,json,time
from pathlib import Path
import torch
from safetensors import safe_open

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',type=Path,required=True,help='显式 BF16 checkpoint；不支持官方混合 INT4/INT8')
 p.add_argument('--layers',type=int,nargs='+',choices=range(36),default=list(range(36)))
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error('Use a new output path; existing evidence is immutable')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(8);model=a.model;index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];rows=[];started=time.monotonic()
 def load(key):
  with safe_open(model/index[key],framework='pt',device='cpu') as f:return f.get_tensor(key)
 def count(w,m):
  bits=w.view(torch.int16).reshape(-1,m*w.shape[1]);counts=[]
  for begin in range(0,len(bits),64):
   ex=((bits[begin:begin+64].to(torch.int32)&65535)>>7)&255;counts.append(((ex<109)|(ex>=125)).sum(1))
  return torch.cat(counts)
 for layer in a.layers:
  prefix=f'model.layers.{layer}.mlp.'
  w1=count(load(prefix+'gate_proj.weight'),4)+count(load(prefix+'up_proj.weight'),4)
  w2=count(load(prefix+'down_proj.weight'),4)
  for name,c in [('w1',w1),('w2',w2)]:
   row=dict(layer=layer,matrix=name,tiles=len(c),max_repairs=int(c.max()),worst_tile=int(c.argmax()),total_repairs=int(c.sum()),tiles_over32=int((c>32).sum()),tiles_over64=int((c>64).sum()),counts=c.tolist());rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='counts'}),flush=True)
 result=dict(complete=True,base=109,capacity_needed=max(r['max_repairs'] for r in rows),worst=max(rows,key=lambda r:r['max_repairs']),matrices=rows,scope='tile count is invariant under tile_weights permutation; W1 M8 combines four gate plus four up rows',wall_seconds=time.monotonic()-started)
 result.update(model=str(model),layers=a.layers,all_model=set(a.layers)==set(range(36)))
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('matrices','worst')}),flush=True)
if __name__=='__main__':main()
