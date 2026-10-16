"""Build on the validated two-pass generator: bulk P transfer, 128-token PV."""
import argparse,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
subprocess.run([sys.executable,str(Path(__file__).with_name('prepare.py')),'--output',a.output],check=True)
path=out/'decode_layer_stage_prefetch.asc';s=path.read_text();start=s.index('namespace FullStage {');end=s.index('namespace SwaStage {',start);f=s[start:end]
f=f.replace('PL1=2*16*N*2,QL1=PL1,KL1=QL1+16*D*2','PL1=16*MAXS*2,QL1=PL1,KL1=QL1+16*D*2')
f=f.replace('pl(TPosition::A1,0,2*16*N)','pl(TPosition::A1,0,16*MAXS)')
f=f.replace('kv(TPosition::B1,KL1,2*N*D);','kv(TPosition::B1,KL1,N*D),kv_alt(TPosition::B1,0,N*D),vv(TPosition::B1,KL1,2*128*D);')
f=f.replace('copy_pages(kv[nb*N*D],k,','copy_pages(nb?kv_alt:kv,k,').replace('auto kl=kv[buf*N*D];','auto kl=buf?kv_alt:kv;')
beg=f.index(' CrossCoreSetFlag<4,PIPE_MTE1>(0);');stop=f.index(' Mutex::Unlock<PIPE_M>(3);Mutex::Lock<PIPE_FIX>(3);fix(c0,PVUB,D);',beg)
f=f[:beg]+r'''
 // K phase has retired; P now owns the low L1 region. V uses two 128-token buffers.
 DataCopyParams vcp{D/16,128,0,0};
 Mutex::Lock<PIPE_MTE2>(4);int physical=table.GetValue(start/128);DataCopy(vv,v[(int64_t)physical*128*1024+head*128*D],vcp);Mutex::Unlock<PIPE_MTE2>(4);
 CrossCoreWaitFlag<4,PIPE_MTE1>(1);CrossCoreWaitFlag<4,PIPE_MTE1>(17);Mutex::Lock<PIPE_M>(3);
 for(int off=0;off<valid;off+=128){
  int slot=(off/128)%2,ev=4+slot;
  if(off+128<valid){int nb=1-slot;Mutex::Lock<PIPE_MTE2>(4+nb);int phys=table.GetValue((start+off+128)/128);DataCopy(vv[nb*128*D],v[(int64_t)phys*128*1024+head*128*D],vcp);Mutex::Unlock<PIPE_MTE2>(4+nb);}
  Mutex::Lock<PIPE_MTE1>(ev);
  auto param=fa_base_matmul::MakeMMParam(4,D,128,false,false);Mutex::Lock<PIPE_MTE1>(2);fa_base_matmul::LoadDataToL0A(a0,pl,param,off*16,128,4);fa_base_matmul::LoadDataToL0B(b0,vv[slot*128*D],param,0,128,D);Mutex::Unlock<PIPE_MTE1>(2);Mutex::Lock<PIPE_M>(2);
  MmadParams mm;mm.m=4;mm.n=D;mm.k=128;mm.cmatrixInitVal=off==0;Mmad(c0,a0,b0,mm);Mutex::Unlock<PIPE_M>(2);Mutex::Unlock<PIPE_MTE1>(ev);
 }
''' + f[stop:]
f=f.replace('pout+offset,bf,p','pout+row*MAXS+tile*N+j*64,bf,p')
beg=f.index('  for(int tile=0;tile<tiles;tile++){',f.index('__aicore__ inline void vec('));stop=f.index('  finish(',beg)
f=f[:beg]+r'''  for(int row=0;row<2;row++){DataCopyParams cp{(uint16_t)(tiles*N/16),1,0,15};DataCopy(pl[(sub*2+row)*16],p[row*MAXS],cp);}
  CrossCoreSetFlag<4,PIPE_MTE3>(1);CrossCoreWaitFlag<4,PIPE_V>(2);
''' + f[stop:]
s=s[:start]+f+s[end:];path.write_text(s)
m=json.loads((out/'execution.json').read_text());m['execution']='two_pass_bulk_probability';m['details']='Full QK in UB, bulk P transfer to L1, double-buffered 128-token PV accum in L0C; ABI unchanged.';(out/'execution.json').write_text(json.dumps(m,indent=2)+'\n')
