"""Two-pass full-attention generator; original source and public ABI stay intact."""
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
s=(ROOT/'kernels/decode_layer_stage_prefetch.asc').read_text()
start=s.index('namespace FullStage {');end=s.index('namespace SwaStage {',start)
f=s[start:end]
f=f.replace('constexpr int N=384,D=256,M=4,','constexpr int MAXS=6144,PVUB=2*MAXS*4;\nconstexpr int N=384,D=256,M=4,')
begin=f.index('__aicore__ inline bool phase_q(');stop=f.index('\n#endif',begin)
f=f[:begin]+r'''// Materialize all QK tiles in UB, then accumulate every PV tile in L0C.
__aicore__ inline void cube(GM_ADDR qp,GM_ADDR kp,GM_ADDR vp,GM_ADDR tp,GM_ADDR pp,int width,int logical){
 int task=logical,part=task/4,head=task%4,start=part*width;GlobalTensor<int64_t>pos;pos.SetGlobalBuffer((__gm__ int64_t*)pp);int valid=imin(width,imax(0,(int)pos.GetValue(0)+1-start));if(valid==0)return;
 GlobalTensor<bfloat16_t>q,k,v;q.SetGlobalBuffer((__gm__ bfloat16_t*)qp);k.SetGlobalBuffer((__gm__ bfloat16_t*)kp);v.SetGlobalBuffer((__gm__ bfloat16_t*)vp);GlobalTensor<int32_t>table;table.SetGlobalBuffer((__gm__ int32_t*)tp);
 LocalTensor<bfloat16_t>pl(TPosition::A1,0,2*16*N),ql(TPosition::A1,QL1,16*D),kv(TPosition::B1,KL1,2*N*D);
 LocalTensor<bfloat16_t>a0(TPosition::A2,0,16*D),b0(TPosition::B2,0,128*D);LocalTensor<float>c0(TPosition::CO1,0,16*imax(N,D));
 Mutex::Lock<PIPE_MTE2>(0);Nd2NzParams nd;nd.ndNum=1;nd.nValue=4;nd.dValue=D;nd.srcDValue=D;nd.dstNzC0Stride=16;nd.dstNzNStride=1;nd.srcNdMatrixStride=0;nd.dstNzMatrixStride=0;DataCopy(ql,q[head*4*D],nd);Mutex::Unlock<PIPE_MTE2>(0);Mutex::Lock<PIPE_MTE1>(0);
 int tiles=(valid+N-1)/N;
 Mutex::Lock<PIPE_MTE2>(4);copy_pages(kv,k,table,start,imin(N,valid),head);Mutex::Unlock<PIPE_MTE2>(4);
 for(int tile=0;tile<tiles;tile++){
  int buf=tile%2,ev=4+buf;
  if(tile+1<tiles){int nb=(tile+1)%2;Mutex::Lock<PIPE_MTE2>(4+nb);copy_pages(kv[nb*N*D],k,table,start+(tile+1)*N,imin(N,valid-(tile+1)*N),head);Mutex::Unlock<PIPE_MTE2>(4+nb);}
  auto kl=kv[buf*N*D];Mutex::Lock<PIPE_MTE1>(ev);Mutex::Lock<PIPE_M>(3);
  for(int sub=0;sub<N;sub+=128){
   auto param=fa_base_matmul::MakeMMParam(4,N,D,false,true);Mutex::Lock<PIPE_MTE1>(2);fa_base_matmul::LoadDataToL0A(a0,ql,param,0,D,4);fa_base_matmul::LoadDataToL0B(b0,kl,param,sub*16,D,128);Mutex::Unlock<PIPE_MTE1>(2);Mutex::Lock<PIPE_M>(2);
   MmadParams mm;mm.m=4;mm.n=128;mm.k=D;mm.cmatrixInitVal=true;Mmad(c0[sub*16],a0,b0,mm);Mutex::Unlock<PIPE_M>(2);
  }
  Mutex::Unlock<PIPE_MTE1>(ev);Mutex::Unlock<PIPE_M>(3);Mutex::Lock<PIPE_FIX>(3);fix(c0,tile*2*N*4,N);Mutex::Unlock<PIPE_FIX>(3);
 }
 CrossCoreSetFlag<4,PIPE_FIX>(2);CrossCoreSetFlag<4,PIPE_FIX>(18);
 CrossCoreSetFlag<4,PIPE_MTE1>(0);CrossCoreSetFlag<4,PIPE_MTE1>(16);CrossCoreSetFlag<4,PIPE_MTE1>(3);CrossCoreSetFlag<4,PIPE_MTE1>(19);
 Mutex::Lock<PIPE_MTE2>(4);copy_pages(kv,v,table,start,imin(N,valid),head);Mutex::Unlock<PIPE_MTE2>(4);
 Mutex::Lock<PIPE_M>(3);
 for(int tile=0;tile<tiles;tile++){
  int slot=tile%2,ev=4+slot,pflag=slot?4:1,qflag=slot?3:0;
  if(tile+1<tiles){int nb=(tile+1)%2;Mutex::Lock<PIPE_MTE2>(4+nb);copy_pages(kv[nb*N*D],v,table,start+(tile+1)*N,imin(N,valid-(tile+1)*N),head);Mutex::Unlock<PIPE_MTE2>(4+nb);}
  CrossCoreWaitFlag<4,PIPE_MTE1>(pflag);CrossCoreWaitFlag<4,PIPE_MTE1>(pflag+16);Mutex::Lock<PIPE_MTE1>(ev);auto kl=kv[slot*N*D];
  for(int sub=0;sub<N;sub+=128){
   auto param=fa_base_matmul::MakeMMParam(4,D,N,false,false);Mutex::Lock<PIPE_MTE1>(2);fa_base_matmul::LoadDataToL0A(a0,pl,param,slot*16*N+sub*16,128,4);fa_base_matmul::LoadDataToL0B(b0,kl,param,sub*16,128,D);Mutex::Unlock<PIPE_MTE1>(2);Mutex::Lock<PIPE_M>(2);
   MmadParams mm;mm.m=4;mm.n=D;mm.k=128;mm.cmatrixInitVal=(tile==0&&sub==0);Mmad(c0,a0,b0,mm);Mutex::Unlock<PIPE_M>(2);
  }
  Mutex::Unlock<PIPE_MTE1>(ev);CrossCoreSetFlag<4,PIPE_MTE1>(qflag);CrossCoreSetFlag<4,PIPE_MTE1>(qflag+16);
 }
 Mutex::Unlock<PIPE_M>(3);Mutex::Lock<PIPE_FIX>(3);fix(c0,PVUB,D);Mutex::Unlock<PIPE_FIX>(3);
 CrossCoreSetFlag<4,PIPE_FIX>(2);CrossCoreSetFlag<4,PIPE_FIX>(18);Mutex::Unlock<PIPE_MTE1>(0);
}
''' + f[stop:]
begin=f.index('__simd_vf__ void softmax(');stop=f.index('__simd_vf__ void finish(',begin)
f=f[:begin]+r'''__simd_vf__ void softmax_all(__ubuf__ float*x,__ubuf__ bfloat16_t*pout,__ubuf__ float*state,int valid,int tiles){
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();auto p=R::CreateMask<float,R::MaskPattern::ALL>();
 for(uint16_t row=0;row<2;row++){
  R::RegTensor<float>mx,v,neg,sum;R::RegTensor<int32_t>ix;R::RegTensor<bfloat16_t>bf;R::Duplicate(mx,-1.e30f);R::Duplicate(neg,-1.e30f);
  for(uint16_t tile=0;tile<tiles;tile++)for(uint16_t j=0;j<N/64;j++){
   int offset=(tile*2+row)*N+j*64;R::LoadAlign(v,x+offset);R::Muls(v,v,.0625f,p);R::Arange(ix,tile*N+j*64);R::MaskReg keep;R::CompareScalar<int32_t,CMPMODE::LT>(keep,ix,valid,p);R::Select(v,v,neg,keep);R::Max(mx,mx,v,p);R::StoreAlign(x+offset,v,p);
  }
  R::Reduce<R::ReduceType::MAX>(mx,mx,p);R::Duplicate(mx,mx,p);R::Duplicate(sum,0.0f);R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
  for(uint16_t tile=0;tile<tiles;tile++)for(uint16_t j=0;j<N/64;j++){
   int offset=(tile*2+row)*N+j*64;R::LoadAlign(v,x+offset);R::Sub(v,v,mx,p);R::Exp(v,v,p);R::Add(sum,sum,v,p);R::Cast<bfloat16_t,float,down>(bf,v,p);R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(pout+offset,bf,p);
  }
  R::Reduce<R::ReduceType::SUM>(sum,sum,p);R::Duplicate(sum,sum,p);R::StoreAlign(state+row*64,mx,p);R::StoreAlign(state+(2+row)*64,sum,p);
 }
}
''' + f[stop:]
begin=f.index('__aicore__ inline void vec(');stop=f.index('\n#endif',begin)
f=f[:begin]+r'''__aicore__ inline void vec(TPipe &pipe,GM_ADDR pp,GM_ADDR yp,GM_ADDR lp,int width,int logical){
 int task=logical/2,part=task/4,head=task%4,sub=GetSubBlockIdx(),start=part*width;GlobalTensor<int64_t>pos;pos.SetGlobalBuffer((__gm__ int64_t*)pp);int valid=imin(width,imax(0,(int)pos.GetValue(0)+1-start)),tiles=(valid+N-1)/N;
 TBuf<QuePosition::VECCALC>reserved,pb,sb,ob,yb,lb;pipe.InitBuffer(reserved,PVUB+2*D*4);pipe.InitBuffer(pb,2*MAXS*2);pipe.InitBuffer(sb,8*64*4);pipe.InitBuffer(ob,2*D*4);pipe.InitBuffer(yb,2*D*2);pipe.InitBuffer(lb,64);
 LocalTensor<float>qk(TPosition::VECIN,0,2*MAXS),pv(TPosition::VECIN,PVUB,2*D);LocalTensor<bfloat16_t>pl(TPosition::A1,0,2*16*N);
 auto p=pb.Get<bfloat16_t>();auto state=sb.Get<float>();auto out=ob.Get<float>();auto y=yb.Get<bfloat16_t>();auto lse=lb.Get<float>();
 init((__ubuf__ float*)state.GetPhyAddr(),(__ubuf__ float*)out.GetPhyAddr());
 if(valid>0){
  CrossCoreWaitFlag<4,PIPE_V>(2);softmax_all((__ubuf__ float*)qk.GetPhyAddr(),(__ubuf__ bfloat16_t*)p.GetPhyAddr(),(__ubuf__ float*)state.GetPhyAddr(),valid,tiles);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
  for(int tile=0;tile<tiles;tile++){
   int slot=tile%2,qflag=slot?3:0,pflag=slot?4:1;CrossCoreWaitFlag<4,PIPE_MTE3>(qflag);
   for(int row=0;row<2;row++){DataCopyParams cp{N/16,1,0,15};DataCopy(pl[slot*16*N+(sub*2+row)*16],p[(tile*2+row)*N],cp);}
   CrossCoreSetFlag<4,PIPE_MTE3>(pflag);
  }
  CrossCoreWaitFlag<4,PIPE_MTE3>(0);CrossCoreWaitFlag<4,PIPE_MTE3>(3);CrossCoreWaitFlag<4,PIPE_V>(2);
  finish((__ubuf__ float*)pv.GetPhyAddr(),(__ubuf__ float*)state.GetPhyAddr(),(__ubuf__ bfloat16_t*)y.GetPhyAddr(),(__ubuf__ float*)lse.GetPhyAddr(),valid);
 }else finish((__ubuf__ float*)out.GetPhyAddr(),(__ubuf__ float*)state.GetPhyAddr(),(__ubuf__ bfloat16_t*)y.GetPhyAddr(),(__ubuf__ float*)lse.GetPhyAddr(),valid);
 GlobalTensor<bfloat16_t>yo;GlobalTensor<float>lo;yo.SetGlobalBuffer((__gm__ bfloat16_t*)yp);lo.SetGlobalBuffer((__gm__ float*)lp);SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);int h=part*16+head*4+sub*2;DataCopy(yo[h*D],y,2*D);DataCopy(lo[(h/2)*8],lse,8);
}
''' + f[stop:]
s=s[:start]+f+s[end:]
s=s.replace('int full=rd==64,width=full?5120:128;', '''int full=rd==64,width=full?5120:128;
 if(full){GlobalTensor<int64_t> dynamic_pos;dynamic_pos.SetGlobalBuffer((__gm__ int64_t*)pos);int length=(int)dynamic_pos.GetValue(0)+1;width=((length+7*384-1)/(7*384))*384;}''')
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'decode_layer_stage_prefetch.asc').write_text(s)
(out/'execution.json').write_text(json.dumps({'candidate':'balanced7_tile384','execution':'two_pass_full_attention','details':'Whole-part QK FP32 UB, global part max, BF16 probabilities, L0C PV across all tiles. Original ABI/packing.'},indent=2)+'\n')
print(hashlib.sha256(s.encode()).hexdigest())
