"""Create an instrumented copy of the fused chain; original source stays intact.

Eight per-AIV timestamps occupy the unused workspace region [124416,128000).
They include existing synchronization and are diagnostic, not production timing.
"""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[3]
source=root/'kernels/decode_layer_stage_prefetch.asc'
s=source.read_text()
stamp='''
__aicore__ inline void profile_stamp(GM_ADDR ws,int stage){
 GlobalTensor<uint64_t> trace;
 trace.SetGlobalBuffer((__gm__ uint64_t*)(ws+124416));
 trace.SetValue(GetBlockIdx()*8+stage,(uint64_t)GetSystemCycle());
}
'''
s=s.replace('namespace QkvStage {',stamp+'\nnamespace QkvStage {',1)
s=s.replace('int mode,int up_slot){','int mode,int up_slot,GM_ADDR profiling_ws){',1)
s=s.replace('stage_barrier();pipe.Reset();for(int logical=GetBlockIdx();logical<56;logical+=56){linear_stage<10240,4>',
            'stage_barrier();profile_stamp(profiling_ws,5);pipe.Reset();for(int logical=GetBlockIdx();logical<56;logical+=56){linear_stage<10240,4>',1)
s=s.replace('if(mode==1){stage_barrier();pipe.Reset();if(GetBlockIdx()==0)',
            'if(mode==1){stage_barrier();profile_stamp(profiling_ws,6);pipe.Reset();if(GetBlockIdx()==0)',1)
at=s.index('extern "C" __global__')
prefix,body=s[:at],s[at:]
body=body.replace('TPipe pipe;\n QkvStage::run','TPipe pipe;profile_stamp(ws,0);\n QkvStage::run',1)
body=body.replace('SyncAll<true>();pipe.Reset();CrossCoreSetFlag','SyncAll<true>();profile_stamp(ws,1);pipe.Reset();CrossCoreSetFlag',1)
body=body.replace('SyncAll<true>();pipe.Reset();\n if(GetBlockIdx()<16)',
                  'SyncAll<true>();profile_stamp(ws,2);pipe.Reset();\n if(GetBlockIdx()<16)',1)
body=body.replace('stage_barrier();pipe.Reset();int up_slot',
                  'stage_barrier();profile_stamp(ws,3);pipe.Reset();int up_slot',1)
body=body.replace('stage_barrier();pipe.Reset();FfnStage::run',
                  'stage_barrier();profile_stamp(ws,4);pipe.Reset();FfnStage::run',1)
body=body.replace('e1,e2,1,up_slot);','e1,e2,1,up_slot,ws);profile_stamp(ws,7);',1)
s=prefix+body
for header in ('decode_cube_matmul.h','decode_sigmoid_lut_vf.h'):
    s=s.replace(f'"{header}"',f'"{root/"kernels"/header}"')
assert s.count('profile_stamp(')==9
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'decode_layer_stage_prefetch.asc').write_text(s)
(a.output/'execution.json').write_text(json.dumps({
    'execution':'instrumented_fused_chain',
    'timestamp_interface':'AscendC::GetSystemCycle; raw counter units, no frequency assumed',
    'timestamp_offset':124416,'timestamp_shape':[56,8],
    'stages':['qkv_rope_cache_sync','attention_sync','merge_gate_sync','out_sync','up_gelu_sync','down_sync','final_norm'],
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'instrumentation_changes':['8 scalar timestamp writes per vector core','unchanged existing barriers and math'],
    'not_production_timing':True},indent=2)+'\n')
print(a.output/'decode_layer_stage_prefetch.asc')
