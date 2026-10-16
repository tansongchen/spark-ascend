"""Boundary witness for HF's window length vs FIA's pre_tokens distance."""
import json
import os
import argparse
from pathlib import Path
import torch
import torch_npu
print('IMPORT_OK',flush=True)
parser=argparse.ArgumentParser()
parser.add_argument('--paged',action='store_true')
args=parser.parse_args()

visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES','').split(',')
if len(visible)!=1 or not visible[0].isdigit() or int(visible[0])==0:
    raise RuntimeError('Expose one checked idle physical NPU other than 0')
torch.npu.set_device(0)
torch.set_num_threads(8)
q=torch.zeros((1,16,256),dtype=torch.bfloat16,device='npu')
k=torch.zeros((513,4,256),dtype=torch.bfloat16,device='npu')
v=torch.zeros_like(k);v[0]=1
extra={}
if args.paged:
    k=torch.zeros((5,128,1024),dtype=torch.bfloat16,device='npu')
    v=torch.zeros_like(k);v[0,0]=1
    extra={'block_table':torch.arange(5,dtype=torch.int32,device='npu')[None], 'block_size':128}
mask=torch.triu(torch.ones((2048,2048),dtype=torch.bool),diagonal=1).npu()
print('FIA_LAUNCH',flush=True)
output,_=torch_npu.npu_fused_infer_attention_score(
    query=q,key=k,value=v,atten_mask=mask,input_layout='TND',
    actual_seq_lengths=[1],actual_seq_lengths_kv=[513],
    num_heads=16,num_key_value_heads=4,scale=.0625,
    pre_tokens=512,next_tokens=0,sparse_mode=4,**extra)
torch.npu.synchronize()
# Current transformers/masking_utils.py: sliding_window_overlay returns
# kv_idx > q_idx - sliding_window, combined with kv_idx <= q_idx.
hfkept=[i for i in range(513) if i > 512-512 and i <= 512]
result={'query_position':512,'history_length':513,'paged':args.paged,
        'visible_physical_npu':int(visible[0]),
        'data':'Q=K=0; V[token0]=1; all other V=0; no output gate',
        'hf_keeps_token0':0 in hfkept,'hf_window_tokens':len(hfkept),
        'hf_expected':0.,'source_513_expected_bf16':float(torch.tensor(1/513).bfloat16()),
        'fia_pre_tokens_512_actual':float(output.cpu().flatten()[0]),
        'fia_output_uniform':bool((output==output.flatten()[0]).all())}
result['fia_matches_source_513']=result['fia_pre_tokens_512_actual']==result['source_513_expected_bf16']
path=Path('/data1/t00995472/spark-kernel-validation/window-semantics.json')
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
