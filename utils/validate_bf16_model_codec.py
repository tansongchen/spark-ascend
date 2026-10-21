"""Validate lossless packing of every actual model MLP matrix, without saving weights."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.bf16_codec import pack_weight,unpack_weight,REPAIRS
from utils.packing import tile_weights

def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',type=Path,default=Path('/data1/t00995472/models/Spark-X2.5-4B'))
 p.add_argument('--layers',type=int,nargs='+',choices=range(36),default=list(range(36)))
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error('Use a new output path; existing evidence is immutable')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(8);model=a.model;index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
 out=a.output;result={'pass':False,'complete':False,'model':str(model),'layers':a.layers,'all_model':set(a.layers)==set(range(36)),'base':109,'repair_capacity':REPAIRS,'matrices':[],'codec_sha256':hashlib.sha256((ROOT/'utils/bf16_codec.py').read_bytes()).hexdigest()};start=time.monotonic()
 def load(key):
  with safe_open(model/index[key],framework='pt',device='cpu') as f:return f.get_tensor(key).contiguous()
 try:
  for layer in a.layers:
   prefix=f'model.layers.{layer}.mlp.';gate=load(prefix+'gate_proj.weight');up=load(prefix+'up_proj.weight')
   w1=torch.stack((gate.reshape(-1,4,2560),up.reshape(-1,4,2560)),dim=1).reshape(20480,2560);del gate,up
   for name,w,m in [('w1',w1,8),('w2',load(prefix+'down_proj.weight'),4)]:
    n,k=w.shape;tiled=tile_weights(w,m);packed=pack_weight(tiled,k,m);decoded=unpack_weight(packed,k,m).reshape_as(tiled);assert torch.equal(tiled.view(torch.int16),decoded.view(torch.int16)),(layer,name,'raw bits differ')
    count=packed[:,:32].view(torch.int32)[:,0]
    row={'layer':layer,'matrix':name,'shape':[n,k],'tile_m':m,'tiles':len(packed),'max_repairs':int(count.max()),'total_repairs':int(count.sum()),'raw_bytes':tiled.numel()*2,'encoded_bytes':packed.numel(),'source_sha256':digest(tiled),'encoded_sha256':digest(packed),'decoded_sha256':digest(decoded),'raw_bits_exact':True};result['matrices'].append(row);print(json.dumps(row),flush=True)
    del tiled,packed,decoded
   del w1,w
   out.write_text(json.dumps(result,indent=2)+'\n')
  result['pass']=True;result['complete']=True;result['max_repairs']=max(r['max_repairs'] for r in result['matrices']);result['total_raw_bytes']=sum(r['raw_bytes'] for r in result['matrices']);result['total_encoded_bytes']=sum(r['encoded_bytes'] for r in result['matrices']);result['wall_seconds']=time.monotonic()-start
 finally:out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='matrices'}),flush=True)
if __name__=='__main__':main()
