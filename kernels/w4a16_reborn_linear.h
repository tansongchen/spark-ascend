#pragma once
#include "kernel_operator.h"

// A new outer-product GEMV: 128 output channels live in SIMD lanes.
// INT4 * FP16 activation is accumulated in four independent FP16 registers
// for one G128 group; the original BF16 scale is applied once to its sum.
// FP32 accumulators combine groups. Inputs and final outputs remain BF16.
namespace Reborn {
using namespace AscendC;
namespace R = AscendC::Reg;
constexpr R::CastTrait widen{R::RegLayout::ZERO, R::SatMode::UNKNOWN,
    R::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr R::CastTrait narrow{R::RegLayout::ZERO, R::SatMode::NO_SAT,
    R::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template<int Groups>
__simd_vf__ void OuterProduct(__ubuf__ half* x, __ubuf__ uint8_t* packed,
                            __ubuf__ float* out) {
    auto ph = R::CreateMask<half, R::MaskPattern::ALL>();
    auto pf = R::CreateMask<float, R::MaskPattern::ALL>();
    R::RegTensor<float> total0, total1;
    R::Duplicate(total0, 0.0f); R::Duplicate(total1, 0.0f);
    for (uint16_t g = 0; g < Groups; ++g) {
        R::RegTensor<half> a, b, c, d;
        R::Duplicate(a, (half)0); R::Duplicate(b, (half)0);
        R::Duplicate(c, (half)0); R::Duplicate(d, (half)0);
        for (uint16_t i = 0; i < 128; i += 4) {
            R::RegTensor<uint8_t> bits;
            R::RegTensor<half> q, act;
#define REBORN_STEP(J, ACC) \
            R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK4_B8>(bits,packed+(g*128+i+J)*64); \
            R::Cast<half,int4x2_t,widen>(q,(R::RegTensor<int4x2_t>&)bits,ph); \
            R::LoadAlign<half,R::LoadDist::DIST_BRC_B16>(act,x+g*128+i+J); \
            R::MulAddDst(ACC,q,act,ph);
            REBORN_STEP(0,a) REBORN_STEP(1,b)
            REBORN_STEP(2,c) REBORN_STEP(3,d)
#undef REBORN_STEP
        }
        R::Add(a,a,b,ph); R::Add(c,c,d,ph); R::Add(a,a,c,ph);
        R::RegTensor<float> v0,v1,s0,s1;
        R::RegTensor<uint32_t> high;
        R::RegTensor<bfloat16_t> scale;
        R::Cast<float,half,widen>(v0,a,pf);
        R::ShiftRights(high,(R::RegTensor<uint32_t>&)a,(int16_t)16,pf);
        R::Cast<float,half,widen>(v1,(R::RegTensor<half>&)high,pf);
        R::LoadAlign(scale,(__ubuf__ bfloat16_t*)(packed+Groups*8192)+g*128);
        R::Cast<float,bfloat16_t,widen>(s0,scale,pf);
        R::ShiftRights(high,(R::RegTensor<uint32_t>&)scale,(int16_t)16,pf);
        R::Cast<float,bfloat16_t,widen>(s1,(R::RegTensor<bfloat16_t>&)high,pf);
        R::MulAddDst(total0,v0,s0,pf); R::MulAddDst(total1,v1,s1,pf);
    }
    R::StoreAlign(out,total0,pf); R::StoreAlign(out+64,total1,pf);
}

template<int K,int Groups>
__aicore__ inline void Linear(TPipe& pipe,GM_ADDR x,GM_ADDR weights,GM_ADDR partial,int n,int cores) {
    constexpr int bytes = Groups*(8192+256);
    constexpr int splits = K/(Groups*128);
    GlobalTensor<bfloat16_t> xg; xg.SetGlobalBuffer((__gm__ bfloat16_t*)x);
    GlobalTensor<uint8_t> wg; wg.SetGlobalBuffer((__gm__ uint8_t*)weights);
    GlobalTensor<float> pg; pg.SetGlobalBuffer((__gm__ float*)partial);
    TBuf<TPosition::VECCALC> xb,xf,xh,wb,yb;
    pipe.InitBuffer(xb,K*2); pipe.InitBuffer(xf,K*4); pipe.InitBuffer(xh,K*2);
    pipe.InitBuffer(wb,bytes*2); pipe.InitBuffer(yb,512);
    auto bx=xb.Get<bfloat16_t>(); auto fx=xf.Get<float>(); auto hx=xh.Get<half>();
    auto bw=wb.Get<uint8_t>(); auto by=yb.Get<float>();
    DataCopy(bx,xg,K);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(fx,bx,RoundMode::CAST_NONE,K); Cast(hx,fx,RoundMode::CAST_RINT,K);
    int jobs=(n/128)*splits, core=GetBlockIdx();
    if(core>=jobs) return;
    DataCopy(bw,wg[core*bytes],bytes);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for(int job=core,turn=0;job<jobs;job+=cores,++turn) {
        int bank=turn&1;
        if(job+cores<jobs) {
            if(turn) {SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);}
            DataCopy(bw[(1-bank)*bytes],wg[(job+cores)*bytes],bytes);
            SetFlag<HardEvent::MTE2_V>(bank?EVENT_ID0:EVENT_ID1);
        }
        WaitFlag<HardEvent::MTE2_V>(bank?EVENT_ID1:EVENT_ID0);
        OuterProduct<Groups>((__ubuf__ half*)hx.GetPhyAddr()+(job%splits)*Groups*128,
            (__ubuf__ uint8_t*)bw.GetPhyAddr()+bank*bytes,(__ubuf__ float*)by.GetPhyAddr());
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(pg[job*128],by,128);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}
} // namespace Reborn
