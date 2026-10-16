"""Diagnostic seven-launch chain. NOT the original fused kernel or a speed test.

Kernel launch boundaries replace global stage barriers. First FFN tiles are
explicitly loaded because UB contents cannot be carried across launches.
All math helpers, intra-attention handshakes and layouts come from supplied code.
"""
import argparse
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
s=(root/'kernels/decode_layer_stage_prefetch.asc').read_text()
s=s[:s.index('extern "C" __global__')]
s=s.replace('// The first tile was loaded by O projection before its final GEMV.',
            '// Independent diagnostic launch: explicitly load the first tile.\n DataCopy(first,b[(int64_t)start*K],M*K);')
params='GM_ADDR x,GM_ADDR res,GM_ADDR wq,GM_ADDR wo,GM_ADDR w1,GM_ADDR w2,GM_ADDR lg,GM_ADDR ls,GM_ADDR g1,GM_ADDR g2,GM_ADDR pos,GM_ADDR cs,GM_ADDR slots,GM_ADDR table,GM_ADDR kc,GM_ADDR vc,GM_ADDR nk,GM_ADDR nv,GM_ADDR ws,GM_ADDR y,GM_ADDR resout,int rd,int slot32,float e1,float e2'
layout=''.join(f'GM_ADDR {name}=ws+{offset};' for name,offset in zip(
    ['q','k','v','gate','partial','lse','merged','oproj','mid','raw','resmid'],
    [0,8192,10240,12288,12544,78080,80128,88320,93440,113920,119040]))+' int full=rd==64,width=full?5120:128;'
stages={
 'qkv':('MIX_AIC_1_2', '#ifndef __DAV_C310_CUBE__\n TPipe pipe; QkvStage::run(pipe,x,wq,pos,cs,slots,q,k,v,gate,kc,vc,nk,nv,rd,1,slot32,full);\n#endif'),
 'attention':('MIX_AIC_1_2', '''#ifdef __DAV_C310_CUBE__
 for(int logical=GetBlockIdx();logical<32;logical+=28){if(full)FullStage::cube(q,nk,nv,table,pos,width,logical);else SwaStage::cube(q,kc,vc,table,pos,width,logical);PipeBarrier<PIPE_ALL>();}
 #else
 TPipe pipe;for(int logical=GetBlockIdx();logical<64;logical+=56){if(full)FullStage::vec(pipe,pos,partial,lse,width,logical);else SwaStage::vec(pipe,pos,partial,lse,width,logical);PipeBarrier<PIPE_ALL>();pipe.Reset();}
 #endif'''),
 'merge':('AIV_ONLY','TPipe pipe;MergeStage::run(pipe,partial,lse,pos,gate,ls,merged,width);'),
 'out':('AIV_ONLY','TPipe pipe;FfnStage::linear_stage<4096,8>(pipe,merged,wo,oproj,2560,GetBlockIdx(),0,false,nullptr);'),
 'up':('AIV_ONLY','TPipe pipe;FfnStage::up_stage(pipe,oproj,res,g1,resmid,e1,w1,lg,mid,GetBlockIdx(),0,nullptr);'),
 'down':('AIV_ONLY','TPipe pipe;FfnStage::linear_stage<10240,4>(pipe,mid,w2,raw,2560,GetBlockIdx(),0,false,nullptr);'),
 'norm':('AIV_ONLY','TPipe pipe;FfnStage::norm_final(pipe,raw,resmid,g2,y,resout,e2);'),
}
for name,(kind,body) in stages.items():
    if kind=='AIV_ONLY':
        body='#ifndef __DAV_C310_CUBE__\n'+body+'\n#endif'
    s+=f'\nextern "C" __global__ __aicore__ void check_{name}({params}){{\nKERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_{kind});InitSocState();{layout}\n{body}\n}}\n'
s+='\nextern "C" void launch_decode_layer_stage_prefetch(void*s,uint8_t**a,uint8_t*kc,uint8_t*vc,uint8_t*nk,uint8_t*nv,uint8_t*ws,uint8_t*y,uint8_t*resout,int rd,int slot32,float e1,float e2){\n'
for name,n in zip(stages,[28,28,16,56,56,56,1]):
    s+=f'check_{name}<<<{n},nullptr,s>>>(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],kc,vc,nk,nv,ws,y,resout,rd,slot32,e1,e2);\n'
s+='}\n'
target=out/'decode_layer_stage_prefetch.asc';target.write_text(s)
cmd=['bisheng','--npu-arch=dav-3510','-fPIC','-shared','-I'+str(root/'kernels'),str(target),'-o',str(out/'libdecode_layer_stage_prefetch.so'),'-lascendcl']
print(' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
(out/'execution.json').write_text(json.dumps({'execution':'diagnostic_seven_launch_chain','changes':['replace inter-stage global barriers with launch boundaries','load first FFN tile instead of carrying UB prefetch across launches']},indent=2))
