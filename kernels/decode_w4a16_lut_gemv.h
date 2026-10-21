#pragma once
#include "kernel_operator.h"

// Quantization-specific 16-value BF16 tables replace per-weight conversions.
// Each group table is exactly BF16(q * original_BF16_scale), q=-8..7.
// Packed INT4 remains resident; no full dequantized weight matrix is allocated.
namespace W4Lut {
namespace R = AscendC::Reg;
constexpr R::CastTrait up{R::RegLayout::ZERO, R::SatMode::UNKNOWN,
                          R::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr R::CastTrait down{R::RegLayout::ZERO, R::SatMode::NO_SAT,
                            R::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

#define W4_ACCUMULATOR(I) R::RegTensor<float> s##I; if constexpr(M>I) R::Duplicate(s##I,0.0f);
#ifdef W4_DEFER_HIGH
#define W4_HIGH_REGISTER(I) R::RegTensor<float> h##I;
#define W4_HIGH_ALIAS(I) R::RegTensor<float>& w1=h##I;
#define W4_HIGH_NOW(I)
#define W4_HIGH_LATER(I) if constexpr(M>I){R::MulAddDst(s##I,a1,h##I,p);}
#else
#define W4_HIGH_REGISTER(I)
#define W4_HIGH_ALIAS(I)
#define W4_HIGH_NOW(I) R::MulAddDst(s##I,a1,w1,p);
#define W4_HIGH_LATER(I)
#endif
#ifdef W4_COMPACT_CONVERT
constexpr R::CastTrait int4_low{R::RegLayout::ZERO, R::SatMode::UNKNOWN,
                               R::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
#define W4_TABLE(I)
#define W4_LOOKUP(I) \
  R::Cast<bfloat16_t,int4x2_t,int4_low>(b0,(R::RegTensor<int4x2_t>&)packed,p16); \
  R::LoadAlign<bfloat16_t,R::LoadDist::DIST_BRC_B16>(bt,(__ubuf__ bfloat16_t*)table+(g*M+I)*2+1); \
  R::Mul(b0,b0,bt,p16); \
  R::ShiftRights(hi,(R::RegTensor<uint32_t>&)b0,(int16_t)16,p); \
  R::Cast<float,bfloat16_t,up>(w0,b0,p);R::Cast<float,bfloat16_t,up>(w1,(R::RegTensor<bfloat16_t>&)hi,p);
#elif defined(W4_HARDWARE_CONVERT)
constexpr R::CastTrait int4_low{R::RegLayout::ZERO, R::SatMode::UNKNOWN,
                               R::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
#define W4_TABLE(I)
#define W4_LOOKUP(I) \
  R::Cast<bfloat16_t,int4x2_t,int4_low>(b0,(R::RegTensor<int4x2_t>&)packed,p16); \
  R::ShiftRights(hi,packed,(int16_t)4,p); \
  R::Cast<bfloat16_t,int4x2_t,int4_low>(b1,(R::RegTensor<int4x2_t>&)hi,p16); \
  R::LoadAlign<bfloat16_t,R::LoadDist::DIST_BRC_B16>(bt,(__ubuf__ bfloat16_t*)table+(g*M+I)*2+1); \
  R::Mul(b0,b0,bt,p16);R::Mul(b1,b1,bt,p16); \
  R::Cast<float,bfloat16_t,up>(w0,b0,p);R::Cast<float,bfloat16_t,up>(w1,b1,p);
#elif defined(W4_DIRECT_GEMV)
constexpr R::CastTrait int_to_float{
    R::RegLayout::ZERO, R::SatMode::UNKNOWN,
    R::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
#define W4_TABLE(I)
#define W4_LOOKUP(I) \
  R::ShiftLefts(lo,packed,(int16_t)28,p); \
  R::ShiftLefts(hi,packed,(int16_t)24,p); \
  R::ShiftRights((R::RegTensor<int32_t>&)lo,(R::RegTensor<int32_t>&)lo,(int16_t)28,p); \
  R::ShiftRights((R::RegTensor<int32_t>&)hi,(R::RegTensor<int32_t>&)hi,(int16_t)28,p); \
  R::Cast<float,int32_t,int_to_float>(w0,(R::RegTensor<int32_t>&)lo,p); \
  R::Cast<float,int32_t,int_to_float>(w1,(R::RegTensor<int32_t>&)hi,p); \
  R::LoadAlign<float,R::LoadDist::DIST_BRC_B32>(ft,table+g*M+I); \
  R::Mul(w0,w0,ft,p);R::Mul(w1,w1,ft,p); \
  R::Cast<bfloat16_t,float,down>(b0,w0,p);R::Cast<bfloat16_t,float,down>(b1,w1,p); \
  R::Cast<float,bfloat16_t,up>(w0,b0,p);R::Cast<float,bfloat16_t,up>(w1,b1,p);
#elif defined(W4_FLOAT_LUT)
#define W4_TABLE(I)
#define W4_LOOKUP(I) \
  R::Gather(w0,table+(g*M+I)*16,lo,p);R::Gather(w1,table+(g*M+I)*16,hi,p);
#elif defined(W4_REGISTER_LUT)
#define W4_TABLE(I) \
 if constexpr(M>I){ \
  R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bt,table+(g*M+I)*16); \
  R::Cast<float,bfloat16_t,up>(ft,bt,p); \
 }
#define W4_LOOKUP(I) \
  if constexpr(I%4){R::Adds(lo,lo,(uint32_t)((I%4)*16),p);R::Adds(hi,hi,(uint32_t)((I%4)*16),p);} \
  R::Gather(w0,ft,lo);R::Gather(w1,ft,hi);
#else
#define W4_TABLE(I)
#define W4_LOOKUP(I) \
  R::Gather(b0,table+(g*M+I)*16,(R::RegTensor<uint16_t>&)lo,p); \
  R::Gather(b1,table+(g*M+I)*16,(R::RegTensor<uint16_t>&)hi,p); \
  R::Cast<float,bfloat16_t,up>(w0,b0,p);R::Cast<float,bfloat16_t,up>(w1,b1,p);
#endif
#ifdef W4_DIRECT_GEMV
#define W4_INDICES
#else
#define W4_INDICES R::And(lo,packed,mask15,p);R::ShiftRights(hi,packed,(int16_t)4,p);
#endif
#define W4_ROW(I) \
 if constexpr(M>I){ \
  W4_HIGH_ALIAS(I) \
  R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK4_B8>((R::RegTensor<uint8_t>&)packed,src+(g*M+I)*64); \
  W4_INDICES \
  W4_LOOKUP(I) \
  R::MulAddDst(s##I,a0,w0,p);W4_HIGH_NOW(I) \
 }
#define W4_REDUCE(I) if constexpr(M>I){R::Reduce<R::ReduceType::SUM>(s##I,s##I,p);R::StoreAlign(tmp+I*64,s##I,p1);}

template<int K,int M>
__simd_vf__ void gemv(__ubuf__ float* x,__ubuf__ uint8_t* src,
                     __ubuf__ float* tmp,__ubuf__ bfloat16_t* y) {
 auto p=R::CreateMask<float,R::MaskPattern::ALL>();
 auto p16=R::CreateMask<bfloat16_t,R::MaskPattern::ALL>();
 uint32_t one=1;auto p1=R::UpdateMask<float>(one);
 R::RegTensor<uint32_t>packed,lo,hi,mask15;
 R::RegTensor<float>a0,a1,w0,w1,ft;
 R::RegTensor<bfloat16_t>b0,b1,bt;
 R::Duplicate(mask15,15u);
#if defined(W4_FLOAT_LUT) || defined(W4_DIRECT_GEMV)
 __ubuf__ float* table=(__ubuf__ float*)(src+K*M/2);
#else
 __ubuf__ bfloat16_t* table=(__ubuf__ bfloat16_t*)(src+K*M/2);
#endif
 W4_ACCUMULATOR(0) W4_ACCUMULATOR(1) W4_ACCUMULATOR(2) W4_ACCUMULATOR(3)
 W4_ACCUMULATOR(4) W4_ACCUMULATOR(5) W4_ACCUMULATOR(6) W4_ACCUMULATOR(7)
 W4_HIGH_REGISTER(0) W4_HIGH_REGISTER(1) W4_HIGH_REGISTER(2) W4_HIGH_REGISTER(3)
 W4_HIGH_REGISTER(4) W4_HIGH_REGISTER(5) W4_HIGH_REGISTER(6) W4_HIGH_REGISTER(7)
 for(uint16_t g=0;g<K/128;g++){
  R::LoadAlign(a0,x+g*128);R::LoadAlign(a1,x+g*128+64);
  W4_TABLE(0)
  W4_ROW(0) W4_ROW(1) W4_ROW(2) W4_ROW(3)
  W4_TABLE(4)
  W4_ROW(4) W4_ROW(5) W4_ROW(6) W4_ROW(7)
  // Preserve each output's K accumulation order while separating its two FMAs.
  W4_HIGH_LATER(0) W4_HIGH_LATER(1) W4_HIGH_LATER(2) W4_HIGH_LATER(3)
  W4_HIGH_LATER(4) W4_HIGH_LATER(5) W4_HIGH_LATER(6) W4_HIGH_LATER(7)
 }
 W4_REDUCE(0) W4_REDUCE(1) W4_REDUCE(2) W4_REDUCE(3)
 W4_REDUCE(4) W4_REDUCE(5) W4_REDUCE(6) W4_REDUCE(7)
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
 uint32_t count=M;auto pm=R::UpdateMask<float>(count);
 R::RegTensor<int32_t>ix;R::RegTensor<float>out;R::RegTensor<bfloat16_t>ob;
 R::Arange(ix,0);R::Muls(ix,ix,64,pm);R::Gather(out,tmp,(R::RegTensor<uint32_t>&)ix,pm);
 R::Cast<bfloat16_t,float,down>(ob,out,pm);
 R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y,ob,pm);
}
#undef W4_ACCUMULATOR
#undef W4_HIGH_REGISTER
#undef W4_HIGH_ALIAS
#undef W4_HIGH_NOW
#undef W4_HIGH_LATER
#undef W4_ROW
#undef W4_INDICES
#undef W4_LOOKUP
#undef W4_TABLE
#undef W4_REDUCE
}
