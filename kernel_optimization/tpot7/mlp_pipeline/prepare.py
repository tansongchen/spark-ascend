import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[3];source=root/'kernels/decode_layer_stage_prefetch.asc';s=source.read_text()
extra=(Path(__file__).parent/'local_down.h').read_text()
s=s.replace('__aicore__ inline void norm_final(',extra+'\n__aicore__ inline void norm_final(',1)
s=s.replace('int mode,int up_slot){','int mode,int up_slot,GM_ADDR scratch){',1)
old='for(int logical=GetBlockIdx();logical<56;logical+=56){down_slot=up_stage(pipe,x,res,g1,resmid,e1,w1,lut,mid,logical,up_slot,w2);PipeBarrier<PIPE_ALL>();pipe.Reset();}\n stage_barrier();pipe.Reset();for(int logical=GetBlockIdx();logical<56;logical+=56){linear_stage<10240,4>(pipe,mid,w2,raw,2560,logical,down_slot,true);PipeBarrier<PIPE_ALL>();pipe.Reset();}'
new='for(int logical=GetBlockIdx();logical<56;logical+=56){up_stage(pipe,x,res,g1,resmid,e1,w1,lut,mid,logical,up_slot,nullptr);PipeBarrier<PIPE_ALL>();pipe.Reset();local_down(pipe,mid,w2,scratch,logical);PipeBarrier<PIPE_ALL>();pipe.Reset();}\n stage_barrier();pipe.Reset();reduce_down(pipe,scratch,raw);PipeBarrier<PIPE_ALL>();pipe.Reset();'
assert old in s;s=s.replace(old,new,1);s=s.replace('e1,e2,1,up_slot);','e1,e2,1,up_slot,ws+131072);',1)
a.output.mkdir(parents=True,exist_ok=True);(a.output/'decode_layer_stage_prefetch.asc').write_text(s)
(a.output/'execution.json').write_text(json.dumps({'candidate':'local_down','workspace_bytes':131072+56*2560*4,'w2_packing':'transpose contiguous [10240,2560] BF16','base_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n')
