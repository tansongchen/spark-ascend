"""Diagnostic timestamps, with explicit full-pipe barriers. Not production timing."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
s=(src/'decode_layer_stage_prefetch.asc').read_text();st=s.index('namespace FullStage {');en=s.index('namespace SwaStage {',st);f=s[st:en]
f=f.replace('namespace R=AscendC::Reg;', '''namespace R=AscendC::Reg;
__aicore__ inline void stamp(GM_ADDR tracep,int id,int part){PipeBarrier<PIPE_ALL>();GlobalTensor<uint64_t> t;t.SetGlobalBuffer((__gm__ uint64_t*)tracep);t.SetValue(id*8+part,GetSystemCycle());}''')
f=f.replace(' int tiles=(valid+N-1)/N;',' stamp(qp+124416,GetBlockIdx(),0);\n int tiles=(valid+N-1)/N;',1)
f=f.replace(' CrossCoreSetFlag<4,PIPE_FIX>(2);CrossCoreSetFlag<4,PIPE_FIX>(18);',' stamp(qp+124416,GetBlockIdx(),1);\n CrossCoreSetFlag<4,PIPE_FIX>(2);CrossCoreSetFlag<4,PIPE_FIX>(18);',1)
f=f.replace('CrossCoreWaitFlag<4,PIPE_MTE1>(17);Mutex::Lock<PIPE_M>(3);','CrossCoreWaitFlag<4,PIPE_MTE1>(17);stamp(qp+124416,GetBlockIdx(),2);Mutex::Lock<PIPE_M>(3);',1)
f=f.replace('fix(c0,PVUB,D);Mutex::Unlock<PIPE_FIX>(3);','fix(c0,PVUB,D);Mutex::Unlock<PIPE_FIX>(3);stamp(qp+124416,GetBlockIdx(),3);',1)
f=f.replace('  CrossCoreWaitFlag<4,PIPE_V>(2);softmax_all(', '  stamp(yp+113664,GetBlockIdx(),0);CrossCoreWaitFlag<4,PIPE_V>(2);stamp(yp+113664,GetBlockIdx(),1);softmax_all(',1)
f=f.replace('state.GetPhyAddr(),valid,tiles);','state.GetPhyAddr(),valid,tiles);stamp(yp+113664,GetBlockIdx(),2);',1)
f=f.replace('  CrossCoreSetFlag<4,PIPE_MTE3>(1);CrossCoreWaitFlag<4,PIPE_V>(2);','  CrossCoreSetFlag<4,PIPE_MTE3>(1);stamp(yp+113664,GetBlockIdx(),3);CrossCoreWaitFlag<4,PIPE_V>(2);stamp(yp+113664,GetBlockIdx(),4);',1)
s=s[:st]+f+s[en:];(out/'decode_layer_stage_prefetch.asc').write_text(s);m=json.loads((src/'execution.json').read_text());m['diagnostic']='full-pipe-barrier timestamps, aic@124416 [28,8], aiv@126208 [56,8]';(out/'execution.json').write_text(json.dumps(m,indent=2)+'\n')
