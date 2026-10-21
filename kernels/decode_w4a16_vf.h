#pragma once
#include "kernel_operator.h"

// INT4-G128 tile format:
// [K/128, M, 64] bytes, first 64 K lanes in low nibbles, next 64 in high.
// Followed by [K/128, M] FP32 scales (exact expansion of checkpoint BF16).
// Decode into the original stage_prefetch [K/64, M, 64] BF16 UB layout.
namespace W4A16 {
namespace R = AscendC::Reg;
constexpr R::CastTrait to_float{R::RegLayout::ZERO, R::SatMode::UNKNOWN,
                               R::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
constexpr R::CastTrait to_bf16{R::RegLayout::ZERO, R::SatMode::NO_SAT,
                              R::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
constexpr R::CastTrait int4_low{R::RegLayout::ZERO, R::SatMode::UNKNOWN,
                               R::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

// An independent probe uses the same path as the direct INT4-convert GEMV.
// FP32 scales contain the exact BF16 bits in their upper 16 bits.
template<int K, int M>
__simd_vf__ void decode_hardware(__ubuf__ uint8_t* src, __ubuf__ bfloat16_t* dst) {
    auto p = R::CreateMask<float, R::MaskPattern::ALL>();
    auto p16 = R::CreateMask<bfloat16_t, R::MaskPattern::ALL>();
    R::RegTensor<uint32_t> packed, shifted;
    R::RegTensor<bfloat16_t> lo, hi, scale;
    for (uint16_t g = 0; g < K / 128; ++g) {
        for (uint16_t row = 0; row < M; ++row) {
            R::LoadAlign<uint8_t, R::LoadDist::DIST_UNPACK4_B8>(
                (R::RegTensor<uint8_t>&)packed, src + (g * M + row) * 64);
#ifdef W4_COMPACT_CONVERT
            // The layout probe shows two BF16 outputs for each selected byte.
            R::Cast<bfloat16_t, int4x2_t, int4_low>(lo, (R::RegTensor<int4x2_t>&)packed, p16);
            R::LoadAlign<bfloat16_t, R::LoadDist::DIST_BRC_B16>(
                scale, (__ubuf__ bfloat16_t*)(src + K * M / 2) + (g * M + row) * 2 + 1);
            R::Mul(lo, lo, scale, p16);
            R::RegTensor<float> f0, f1;
            R::ShiftRights(shifted, (R::RegTensor<uint32_t>&)lo, (int16_t)16, p);
            R::Cast<float, bfloat16_t, int4_low>(f0, lo, p);
            R::Cast<float, bfloat16_t, int4_low>(f1, (R::RegTensor<bfloat16_t>&)shifted, p);
            R::Cast<bfloat16_t, float, to_bf16>(lo, f0, p);
            R::Cast<bfloat16_t, float, to_bf16>(hi, f1, p);
#else
            R::Cast<bfloat16_t, int4x2_t, int4_low>(lo, (R::RegTensor<int4x2_t>&)packed, p16);
            // RegLayout::ONE does not select the high nibble of each byte.
            R::ShiftRights(shifted, packed, (int16_t)4, p);
            R::Cast<bfloat16_t, int4x2_t, int4_low>(hi, (R::RegTensor<int4x2_t>&)shifted, p16);
            R::LoadAlign<bfloat16_t, R::LoadDist::DIST_BRC_B16>(
                scale, (__ubuf__ bfloat16_t*)(src + K * M / 2) + (g * M + row) * 2 + 1);
            R::Mul(lo, lo, scale, p16);
            R::Mul(hi, hi, scale, p16);
#endif
            R::StoreAlign<bfloat16_t, R::StoreDist::DIST_PACK_B32>(
                dst + (2 * g * M + row) * 64, lo, p);
            R::StoreAlign<bfloat16_t, R::StoreDist::DIST_PACK_B32>(
                dst + ((2 * g + 1) * M + row) * 64, hi, p);
        }
    }
    R::LocalMemBar<R::MemType::VEC_STORE, R::MemType::VEC_LOAD>();
}
template<int K, int M>
__simd_vf__ void decode(__ubuf__ uint8_t* src, __ubuf__ bfloat16_t* dst) {
    auto p = R::CreateMask<uint32_t, R::MaskPattern::ALL>();
    R::RegTensor<uint32_t> packed, low, high;
    R::RegTensor<int32_t> signed_low, signed_high;
    R::RegTensor<float> scale, lo, hi;
    R::RegTensor<bfloat16_t> blo, bhi;
    for (uint16_t g = 0; g < K / 128; ++g) {
        for (uint16_t row = 0; row < M; ++row) {
            R::LoadAlign<uint8_t, R::LoadDist::DIST_UNPACK4_B8>(
                (R::RegTensor<uint8_t>&)packed, src + (g * M + row) * 64);
            R::ShiftLefts(low, packed, (int16_t)28, p);
            R::ShiftLefts(high, packed, (int16_t)24, p);
            R::ShiftRights(signed_low, (R::RegTensor<int32_t>&)low, (int16_t)28, p);
            R::ShiftRights(signed_high, (R::RegTensor<int32_t>&)high, (int16_t)28, p);
            R::Cast<float, int32_t, to_float>(lo, signed_low, p);
            R::Cast<float, int32_t, to_float>(hi, signed_high, p);
            R::LoadAlign<float, R::LoadDist::DIST_BRC_B32>(
                scale, (__ubuf__ float*)(src + K * M / 2) + g * M + row);
            R::Mul(lo, lo, scale, p);
            R::Mul(hi, hi, scale, p);
            R::Cast<bfloat16_t, float, to_bf16>(blo, lo, p);
            R::Cast<bfloat16_t, float, to_bf16>(bhi, hi, p);
            R::StoreAlign<bfloat16_t, R::StoreDist::DIST_PACK_B32>(
                dst + (2 * g * M + row) * 64, blo, p);
            R::StoreAlign<bfloat16_t, R::StoreDist::DIST_PACK_B32>(
                dst + ((2 * g + 1) * M + row) * 64, bhi, p);
        }
    }
    R::LocalMemBar<R::MemType::VEC_STORE, R::MemType::VEC_LOAD>();
}
}
