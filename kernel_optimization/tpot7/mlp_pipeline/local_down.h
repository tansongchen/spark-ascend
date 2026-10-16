// W2 is transposed BF16 [10240,2560]; scratch is 56*2560 FP32 values.
__simd_vf__ void outer_accum(__ubuf__ bfloat16_t*x,__ubuf__ bfloat16_t*w,__ubuf__ float*out,int count,bool first){
 auto p=R::CreateMask<float,R::MaskPattern::ALL>();
 for(uint16_t j=0;j<5;j++){
  R::RegTensor<float>a,b;R::RegTensor<bfloat16_t>ab,bb;
  R::RegTensor<float>s0;if(first)R::Duplicate(s0,0.0f);else R::LoadAlign(s0,out+j*512+0);
  R::RegTensor<float>s1;if(first)R::Duplicate(s1,0.0f);else R::LoadAlign(s1,out+j*512+64);
  R::RegTensor<float>s2;if(first)R::Duplicate(s2,0.0f);else R::LoadAlign(s2,out+j*512+128);
  R::RegTensor<float>s3;if(first)R::Duplicate(s3,0.0f);else R::LoadAlign(s3,out+j*512+192);
  R::RegTensor<float>s4;if(first)R::Duplicate(s4,0.0f);else R::LoadAlign(s4,out+j*512+256);
  R::RegTensor<float>s5;if(first)R::Duplicate(s5,0.0f);else R::LoadAlign(s5,out+j*512+320);
  R::RegTensor<float>s6;if(first)R::Duplicate(s6,0.0f);else R::LoadAlign(s6,out+j*512+384);
  R::RegTensor<float>s7;if(first)R::Duplicate(s7,0.0f);else R::LoadAlign(s7,out+j*512+448);
  for(uint16_t k=0;k<count;k++){
   R::LoadAlign<bfloat16_t,R::LoadDist::DIST_BRC_B16>(ab,x+k);R::Cast<float,bfloat16_t,up>(a,ab,p);
   R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bb,w+k*2560+j*512+0);R::Cast<float,bfloat16_t,up>(b,bb,p);R::MulAddDst(s0,a,b,p);
   R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bb,w+k*2560+j*512+64);R::Cast<float,bfloat16_t,up>(b,bb,p);R::MulAddDst(s1,a,b,p);
   R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bb,w+k*2560+j*512+128);R::Cast<float,bfloat16_t,up>(b,bb,p);R::MulAddDst(s2,a,b,p);
   R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bb,w+k*2560+j*512+192);R::Cast<float,bfloat16_t,up>(b,bb,p);R::MulAddDst(s3,a,b,p);
   R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bb,w+k*2560+j*512+256);R::Cast<float,bfloat16_t,up>(b,bb,p);R::MulAddDst(s4,a,b,p);
   R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bb,w+k*2560+j*512+320);R::Cast<float,bfloat16_t,up>(b,bb,p);R::MulAddDst(s5,a,b,p);
   R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bb,w+k*2560+j*512+384);R::Cast<float,bfloat16_t,up>(b,bb,p);R::MulAddDst(s6,a,b,p);
   R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bb,w+k*2560+j*512+448);R::Cast<float,bfloat16_t,up>(b,bb,p);R::MulAddDst(s7,a,b,p);
  }
  R::StoreAlign(out+j*512+0,s0,p);
  R::StoreAlign(out+j*512+64,s1,p);
  R::StoreAlign(out+j*512+128,s2,p);
  R::StoreAlign(out+j*512+192,s3,p);
  R::StoreAlign(out+j*512+256,s4,p);
  R::StoreAlign(out+j*512+320,s5,p);
  R::StoreAlign(out+j*512+384,s6,p);
  R::StoreAlign(out+j*512+448,s7,p);
 }
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
}
__aicore__ inline void local_down(TPipe &pipe,GM_ADDR midgm,GM_ADDR wgm,GM_ADDR scratch,int logical){
 constexpr int B=16,N=2560,SLOT=B*N;
 TBuf<QuePosition::VECCALC>wb,xb,ob;pipe.InitBuffer(wb,2*SLOT*2);pipe.InitBuffer(xb,384);pipe.InitBuffer(ob,2560*4);
 auto w=wb.Get<bfloat16_t>();auto x=xb.Get<bfloat16_t>();auto out=ob.Get<float>();
 GlobalTensor<bfloat16_t>mg,wg;GlobalTensor<float>sg;mg.SetGlobalBuffer((__gm__ bfloat16_t*)midgm);wg.SetGlobalBuffer((__gm__ bfloat16_t*)wgm);wg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);sg.SetGlobalBuffer((__gm__ float*)scratch);
 int begin=(2560*logical/56)*4,end=(2560*(logical+1)/56)*4,count=end-begin;
 DataCopyExtParams cp{1,(uint32_t)(count*2),0,0,0};DataCopyPad(x,mg[begin],cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
 SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 int first=count<B?count:B;DataCopy(w,wg[(int64_t)begin*N],first*N);SetFlag<HardEvent::MTE2_V>(4);
 int tiles=(count+B-1)/B;
 for(int i=0;i<tiles;i++){
  int slot=i&1,other=slot^1,off=i*B,n=count-off;if(n>B)n=B;
  if(i>0)WaitFlag<HardEvent::V_MTE2>(4+other);
  if(i+1<tiles){int nn=count-off-B;if(nn>B)nn=B;DataCopy(w[other*SLOT],wg[(int64_t)(begin+off+B)*N],nn*N);SetFlag<HardEvent::MTE2_V>(4+other);}
  WaitFlag<HardEvent::MTE2_V>(4+slot);
  outer_accum((__ubuf__ bfloat16_t*)x[off].GetPhyAddr(),(__ubuf__ bfloat16_t*)w[slot*SLOT].GetPhyAddr(),(__ubuf__ float*)out.GetPhyAddr(),n,i==0);
  SetFlag<HardEvent::V_MTE2>(4+slot);
 }
 WaitFlag<HardEvent::V_MTE2>(4+((tiles-1)&1));SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(sg[logical*2560],out,2560);
}
__simd_vf__ void reduce56(__ubuf__ float*x,__ubuf__ bfloat16_t*y){
 auto p=R::CreateMask<float,R::MaskPattern::ALL>();R::RegTensor<float>a,b;R::RegTensor<bfloat16_t>out;R::Duplicate(a,0.0f);
 for(uint16_t i=0;i<56;i++){R::LoadAlign(b,x+i*64);R::Add(a,a,b,p);}
 R::Cast<bfloat16_t,float,down>(out,a,p);R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y,out,p);
}
__aicore__ inline void reduce_down(TPipe &pipe,GM_ADDR scratch,GM_ADDR raw){
 int core=GetBlockIdx();if(core>=40)return;
 TBuf<QuePosition::VECCALC>xb,yb;pipe.InitBuffer(xb,56*256);pipe.InitBuffer(yb,128);auto x=xb.Get<float>();auto y=yb.Get<bfloat16_t>();GlobalTensor<float>sg;GlobalTensor<bfloat16_t>rg;sg.SetGlobalBuffer((__gm__ float*)scratch);rg.SetGlobalBuffer((__gm__ bfloat16_t*)raw);
 DataCopyExtParams cp{56,256,9984,0,0};DataCopyPad(x,sg[core*64],cp,DataCopyPadExtParams<float>{false,0,0,0});SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 reduce56((__ubuf__ float*)x.GetPhyAddr(),(__ubuf__ bfloat16_t*)y.GetPhyAddr());SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(rg[core*64],y,64);
}
