#pragma once
// FP8 is a storage codec only. Cube QK/PV and softmax still consume BF16.
// The softmax/accumulator math below retains the v10b row contract;
// KV movement and the Cube/AIV schedule are new, with 512-position tiles.
#include "decode_cube_matmul.h"
#include "w4a16_reborn_kv8.h"
namespace Full8 {
using namespace AscendC;
using namespace AscendC::Impl::Detail;
namespace R=AscendC::Reg;
constexpr int N=512,D=256,Half=N/2,PL1=16*N*2,QL1=PL1,KL1=QL1+16*D*2;
constexpr R::CastTrait up{R::RegLayout::ZERO,R::SatMode::UNKNOWN,R::MaskMergeMode::ZEROING,RoundMode::UNKNOWN};
constexpr R::CastTrait down{R::RegLayout::ZERO,R::SatMode::NO_SAT,R::MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
__aicore__ inline int imin(int a,int b){return a<b?a:b;}
__aicore__ inline int imax(int a,int b){return a>b?a:b;}
#ifndef __DAV_C310_CUBE__
__simd_vf__ void init(__ubuf__ float*state,__ubuf__ float*out){
 auto p=R::CreateMask<float,R::MaskPattern::ALL>();R::RegTensor<float>z,m;R::Duplicate(z,0.0f);R::Duplicate(m,-1.e30f);
 for(uint16_t i=0;i<2;i++)R::StoreAlign(state+i*64,m,p);
 for(uint16_t i=2;i<6;i++)R::StoreAlign(state+i*64,z,p);
 for(uint16_t i=0;i<8;i++)R::StoreAlign(out+i*64,z,p);
}
__simd_vf__ void softmax(__ubuf__ float*x,__ubuf__ bfloat16_t*pout,__ubuf__ float*state,int valid,int lower){
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();auto p=R::CreateMask<float,R::MaskPattern::ALL>();
 for(uint16_t row=0;row<2;row++){
  R::RegTensor<float>mx,v,neg,oldm,olds,alpha,sum,z,ns;R::RegTensor<int32_t>ix;R::RegTensor<bfloat16_t>bf;R::Duplicate(mx,-1.e30f);R::Duplicate(neg,-1.e30f);
  for(uint16_t j=0;j<N/64;j++){R::LoadAlign(v,x+row*N+j*64);R::Muls(v,v,.0625f,p);R::Arange(ix,j*64);R::MaskReg keep;R::CompareScalar<int32_t,CMPMODE::LT>(keep,ix,valid,p);R::Select(v,v,neg,keep);R::CompareScalar<int32_t,CMPMODE::GE>(keep,ix,lower,p);R::Select(v,v,neg,keep);R::Max(mx,mx,v,p);R::StoreAlign(x+row*N+j*64,v,p);}
  R::Reduce<R::ReduceType::MAX>(mx,mx,p);R::Duplicate(mx,mx,p);R::LoadAlign(oldm,state+row*64);R::LoadAlign(olds,state+(2+row)*64);R::Max(mx,mx,oldm,p);R::Sub(alpha,oldm,mx,p);R::Exp(alpha,alpha,p);R::Duplicate(sum,0.0f);
  R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
  for(uint16_t j=0;j<N/64;j++){R::LoadAlign(v,x+row*N+j*64);R::Sub(v,v,mx,p);R::Exp(v,v,p);R::Add(sum,sum,v,p);R::Cast<bfloat16_t,float,down>(bf,v,p);R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(pout+row*N+j*64,bf,p);}
  R::Reduce<R::ReduceType::SUM>(sum,sum,p);R::Duplicate(sum,sum,p);R::Mul(ns,olds,alpha,p);R::Add(ns,ns,sum,p);R::StoreAlign(state+row*64,mx,p);R::StoreAlign(state+(2+row)*64,ns,p);R::StoreAlign(state+(4+row)*64,alpha,p);
 }
}
__simd_vf__ void accum(__ubuf__ float*x,__ubuf__ float*out,__ubuf__ float*state){
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();auto p=R::CreateMask<float,R::MaskPattern::ALL>();R::RegTensor<float>a,b,c;
 for(uint16_t row=0;row<2;row++){R::LoadAlign(c,state+(4+row)*64);for(uint16_t j=0;j<4;j++){R::LoadAlign(a,out+row*256+j*64);R::LoadAlign(b,x+row*256+j*64);R::Mul(a,a,c,p);R::Add(a,a,b,p);R::StoreAlign(out+row*256+j*64,a,p);}}
}
__simd_vf__ void finish(__ubuf__ float*out,__ubuf__ float*state,__ubuf__ bfloat16_t*y,__ubuf__ float*lse,int valid){
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();auto p=R::CreateMask<float,R::MaskPattern::ALL>();R::RegTensor<float>a,s,m,l;R::RegTensor<bfloat16_t>b;uint32_t one=1;auto p1=R::UpdateMask<float>(one);
 for(uint16_t row=0;row<2;row++){
  R::LoadAlign(s,state+(2+row)*64);R::LoadAlign(m,state+row*64);if(valid==0)R::Duplicate(s,1.0f);R::Ln(l,s,p);R::Add(l,l,m,p);R::StoreAlign(lse+row*8,l,p1);
  for(uint16_t j=0;j<4;j++){R::LoadAlign(a,out+row*D+j*64);R::Div(a,a,s,p);R::Cast<bfloat16_t,float,down>(b,a,p);R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+row*D+j*64,b,p);}
 }
 // Pack the two scalar LSE values without changing their FP32 arithmetic.
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
 uint32_t two=2;auto p2=R::UpdateMask<float>(two);
 R::RegTensor<int32_t>indices;R::RegTensor<float>packed;
 R::Arange(indices,0);R::Muls(indices,indices,8,p2);
 R::Gather(packed,lse,(R::RegTensor<uint32_t>&)indices,p2);
 R::StoreAlign(lse,packed,p2);
}

#endif

#ifdef __DAV_C310_CUBE__
static constexpr FixpipeConfig cfg{CO2Layout::ROW_MAJOR,true};
__aicore__ inline void drain(LocalTensor<float> src,int addr,int cols) {
    LocalTensor<float> dst(TPosition::VECIN,addr,2*cols);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    p.nSize=cols;p.mSize=4;p.srcStride=16;p.dstStride=cols;p.dualDstCtl=1;
    p.params.ndNum=1;p.params.srcNdStride=0;p.params.dstNdStride=0;
    Fixpipe<float,float,cfg>(dst,src,p);
}
__aicore__ inline void cube(GM_ADDR qp,GM_ADDR pp,int width,int logical) {
    int part=logical/4,head=logical%4,start=part*width;
    GlobalTensor<int64_t> pos;pos.SetGlobalBuffer((__gm__ int64_t*)pp);
    int valid=imin(width,imax(0,(int)pos.GetValue(0)+1-start));if(!valid)return;
    GlobalTensor<bfloat16_t> q;q.SetGlobalBuffer((__gm__ bfloat16_t*)qp);
    LocalTensor<bfloat16_t> pl(TPosition::A1,0,16*N),ql(TPosition::A1,QL1,16*D),kv(TPosition::B1,KL1,N*D);
    LocalTensor<bfloat16_t> a(TPosition::A2,0,16*D),b(TPosition::B2,0,128*D);
    LocalTensor<float> c(TPosition::CO1,0,16*N);
    Mutex::Lock<PIPE_MTE2>(0);
    Nd2NzParams nd;nd.ndNum=1;nd.nValue=4;nd.dValue=D;nd.srcDValue=D;
    nd.dstNzC0Stride=16;nd.dstNzNStride=1;nd.srcNdMatrixStride=0;nd.dstNzMatrixStride=0;
    DataCopy(ql,q[head*4*D],nd);Mutex::Unlock<PIPE_MTE2>(0);Mutex::Lock<PIPE_MTE1>(0);
    for(int off=0;off<valid;off+=N) {
        CrossCoreWaitFlag<4,PIPE_MTE1>(8);CrossCoreWaitFlag<4,PIPE_MTE1>(24);
        Mutex::Lock<PIPE_M>(3);
        for(int sub=0;sub<N;sub+=128) {
            auto param=fa_base_matmul::MakeMMParam(4,N,D,false,true);
            Mutex::Lock<PIPE_MTE1>(2);
            fa_base_matmul::LoadDataToL0A(a,ql,param,0,D,4);
            fa_base_matmul::LoadDataToL0B(b,kv,param,sub*16,D,128);
            Mutex::Unlock<PIPE_MTE1>(2);Mutex::Lock<PIPE_M>(2);
            MmadParams mm;mm.m=4;mm.n=128;mm.k=D;mm.cmatrixInitVal=true;Mmad(c[sub*16],a,b,mm);
            Mutex::Unlock<PIPE_M>(2);
        }
        CrossCoreSetFlag<4,PIPE_MTE1>(9);CrossCoreSetFlag<4,PIPE_MTE1>(25);
        Mutex::Unlock<PIPE_M>(3);Mutex::Lock<PIPE_FIX>(3);drain(c,0,N);Mutex::Unlock<PIPE_FIX>(3);
        CrossCoreSetFlag<4,PIPE_FIX>(0);CrossCoreSetFlag<4,PIPE_FIX>(16);
        CrossCoreWaitFlag<4,PIPE_MTE1>(8);CrossCoreWaitFlag<4,PIPE_MTE1>(24);
        CrossCoreWaitFlag<4,PIPE_MTE1>(1);CrossCoreWaitFlag<4,PIPE_MTE1>(17);
        Mutex::Lock<PIPE_M>(3);
        for(int sub=0;sub<N;sub+=128) {
            auto param=fa_base_matmul::MakeMMParam(4,D,N,false,false);
            Mutex::Lock<PIPE_MTE1>(2);
            fa_base_matmul::LoadDataToL0A(a,pl,param,sub*16,128,4);
            fa_base_matmul::LoadDataToL0B(b,kv,param,sub*16,128,D);
            Mutex::Unlock<PIPE_MTE1>(2);Mutex::Lock<PIPE_M>(2);
            MmadParams mm;mm.m=4;mm.n=D;mm.k=128;mm.cmatrixInitVal=sub==0;Mmad(c,a,b,mm);
            Mutex::Unlock<PIPE_M>(2);
        }
        CrossCoreSetFlag<4,PIPE_MTE1>(9);CrossCoreSetFlag<4,PIPE_MTE1>(25);
        Mutex::Unlock<PIPE_M>(3);Mutex::Lock<PIPE_FIX>(3);drain(c,2*N*4,D);Mutex::Unlock<PIPE_FIX>(3);
        CrossCoreSetFlag<4,PIPE_FIX>(2);CrossCoreSetFlag<4,PIPE_FIX>(18);
    }
    Mutex::Unlock<PIPE_MTE1>(0);
}
#else
__aicore__ inline void prepare(LocalTensor<uint8_t> packed,LocalTensor<bfloat16_t> decoded,
    GlobalTensor<uint8_t> cache,GlobalTensor<int32_t> table,int first,int valid,int head,bool initial) {
    int sub=GetSubBlockIdx();
    SetFlag<HardEvent::V_MTE2>(4);WaitFlag<HardEvent::V_MTE2>(4);
    for(int j=0;j<Half;j+=128) {
        int offset=imin(sub*Half+j,((valid-1)/128)*128);
        int physical=table.GetValue((first+offset)/128);
        DataCopyParams cp{16,64,0,(Half-128)/2};
        DataCopy(packed[j*16],cache[(int64_t)physical*131072+head*32768],cp);
    }
    SetFlag<HardEvent::MTE2_V>(4);WaitFlag<HardEvent::MTE2_V>(4);
    SetFlag<HardEvent::MTE3_V>(4);WaitFlag<HardEvent::MTE3_V>(4);
    Kv8::Decode((__ubuf__ uint8_t*)packed.GetPhyAddr(),(__ubuf__ bfloat16_t*)decoded.GetPhyAddr(),Half*D);
    SetFlag<HardEvent::V_MTE3>(4);WaitFlag<HardEvent::V_MTE3>(4);
    if(!initial)CrossCoreWaitFlag<4,PIPE_MTE3>(9);
    LocalTensor<bfloat16_t> l1(TPosition::A1,KL1,N*D);
    DataCopyParams cp{16,Half,0,N-Half};DataCopy(l1[sub*Half*16],decoded,cp);
    CrossCoreSetFlag<4,PIPE_MTE3>(8);
}
__aicore__ inline void vec(TPipe& pipe,GM_ADDR kp,GM_ADDR vp,GM_ADDR tp,GM_ADDR pp,
                          GM_ADDR yp,GM_ADDR lp,int width,int logical) {
    int task=logical/2,part=task/4,head=task%4,sub=GetSubBlockIdx(),start=part*width;
    GlobalTensor<int64_t> pos;pos.SetGlobalBuffer((__gm__ int64_t*)pp);
    int valid=imin(width,imax(0,(int)pos.GetValue(0)+1-start));
    GlobalTensor<uint8_t> k,v;k.SetGlobalBuffer((__gm__ uint8_t*)kp);v.SetGlobalBuffer((__gm__ uint8_t*)vp);
    GlobalTensor<int32_t> table;table.SetGlobalBuffer((__gm__ int32_t*)tp);
    TBuf<TPosition::VECCALC> reserved,pb,sb,ob,yb,lb,kb,db;
    pipe.InitBuffer(reserved,2*N*4+2*D*4);pipe.InitBuffer(pb,2*N*2);pipe.InitBuffer(sb,6*64*4);
    pipe.InitBuffer(ob,2*D*4);pipe.InitBuffer(yb,2*D*2);pipe.InitBuffer(lb,64);
    pipe.InitBuffer(kb,Half*D);pipe.InitBuffer(db,Half*D*2);
    LocalTensor<float> qk(TPosition::VECIN,0,2*N),pv(TPosition::VECIN,2*N*4,2*D);
    LocalTensor<bfloat16_t> pl(TPosition::A1,0,16*N);
    auto p=pb.Get<bfloat16_t>(),y=yb.Get<bfloat16_t>(),decoded=db.Get<bfloat16_t>();
    auto state=sb.Get<float>(),out=ob.Get<float>(),lse=lb.Get<float>();auto packed=kb.Get<uint8_t>();
    init((__ubuf__ float*)state.GetPhyAddr(),(__ubuf__ float*)out.GetPhyAddr());
    if(valid)prepare(packed,decoded,k,table,start,imin(N,valid),head,true);
    for(int off=0;off<valid;off+=N) {
        int count=imin(N,valid-off);
        prepare(packed,decoded,v,table,start+off,count,head,false);
        CrossCoreWaitFlag<4,PIPE_V>(0);
        softmax((__ubuf__ float*)qk.GetPhyAddr(),(__ubuf__ bfloat16_t*)p.GetPhyAddr(),
                (__ubuf__ float*)state.GetPhyAddr(),count,0);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        for(int row=0;row<2;++row) {
            DataCopyParams cp{N/16,1,0,15};DataCopy(pl[(sub*2+row)*16],p[row*N],cp);
        }
        CrossCoreSetFlag<4,PIPE_MTE3>(1);
        if(off+N<valid)prepare(packed,decoded,k,table,start+off+N,imin(N,valid-off-N),head,false);
        CrossCoreWaitFlag<4,PIPE_V>(2);
        accum((__ubuf__ float*)pv.GetPhyAddr(),(__ubuf__ float*)out.GetPhyAddr(),(__ubuf__ float*)state.GetPhyAddr());
    }
    if(valid)CrossCoreWaitFlag<4,PIPE_MTE3>(9);
    finish((__ubuf__ float*)out.GetPhyAddr(),(__ubuf__ float*)state.GetPhyAddr(),
           (__ubuf__ bfloat16_t*)y.GetPhyAddr(),(__ubuf__ float*)lse.GetPhyAddr(),valid);
    GlobalTensor<bfloat16_t> yo;GlobalTensor<float> lo;
    yo.SetGlobalBuffer((__gm__ bfloat16_t*)yp);lo.SetGlobalBuffer((__gm__ float*)lp);
    SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
    int h=part*16+head*4+sub*2;DataCopy(yo[h*D],y,2*D);DataCopy(lo[(h/2)*8],lse,8);
}
#endif
} // namespace Full8
