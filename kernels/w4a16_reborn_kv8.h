#pragma once
#include "kernel_operator.h"
namespace Kv8 {
using namespace AscendC;
namespace R=AscendC::Reg;
constexpr R::CastTrait up{R::RegLayout::ZERO,R::SatMode::UNKNOWN,R::MaskMergeMode::ZEROING,RoundMode::UNKNOWN};
constexpr R::CastTrait bf{R::RegLayout::ZERO,R::SatMode::NO_SAT,R::MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
constexpr R::CastTrait fp8{R::RegLayout::ZERO,R::SatMode::SAT,R::MaskMergeMode::ZEROING,RoundMode::CAST_RINT};
__simd_vf__ void Encode(__ubuf__ bfloat16_t* src,__ubuf__ uint8_t* dst,int count) {
    auto m=R::CreateMask<float,R::MaskPattern::ALL>();
    R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
    for(uint16_t i=0;i<count/64;++i) {
        R::RegTensor<bfloat16_t>b;R::RegTensor<float>f;R::RegTensor<fp8_e4m3fn_t>q;
        R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,src+i*64);
        R::Cast<float,bfloat16_t,up>(f,b,m);
        // SAT alone still emitted NaN for finite FP8 overflow on this build.
        // Clamp explicitly so large BF16 K/V cannot poison attention.
        R::Maxs(f,f,-448.f,m);R::Mins(f,f,448.f,m);
        R::Cast<fp8_e4m3fn_t,float,fp8>(q,f,m);
        R::StoreAlign<fp8_e4m3fn_t,R::StoreDist::DIST_PACK4_B32>((__ubuf__ fp8_e4m3fn_t*)dst+i*64,q,m);
    }
}
__simd_vf__ void Decode(__ubuf__ uint8_t* src,__ubuf__ bfloat16_t* dst,int count) {
    auto m=R::CreateMask<float,R::MaskPattern::ALL>();
    for(uint16_t i=0;i<count/64;++i) {
        R::RegTensor<uint8_t>q;R::RegTensor<float>f;R::RegTensor<bfloat16_t>b;
        R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK4_B8>(q,src+i*64);
        R::Cast<float,fp8_e4m3fn_t,up>(f,(R::RegTensor<fp8_e4m3fn_t>&)q,m);
        R::Cast<bfloat16_t,float,bf>(b,f,m);
        R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(dst+i*64,b,m);
    }
}
} // namespace Kv8
