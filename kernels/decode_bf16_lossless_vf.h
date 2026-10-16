/**
 * BF16 无损存储解码：所有计算均为整数位运算，输出保留原始 16 位码点。
 *
 * 输入 tile（小端）：[32B header][E 个 sign/mantissa 字节]
 *                  [E/2 个 exponent-nibble 字节][256 个 4B repair pair]。
 * header uint32[8] = {repair_count, exponent_base, magic, E, 0, 0, 0, 0}，
 * magic=0x42463132；repair pair 为 uint16 索引 + uint16 原始 BF16 位模式。
 * 每个指数字节低/高 nibble 分别对应偶数/奇数元素；异常项最后精确覆盖。
 *
 * 融合入口使用 E=20480 / 40960、base=109、LOSSLESS_REPAIRS=256，
 * 对应输入 31776 / 62496 bytes，输出 E*2 bytes；输入输出位于独立 UB 区域。
 * 调用方须保证 MTE2 完成、头部与索引合法；本 VF 不检查不可信编码数据。
 * Python pack_weight 会拒绝超过 256 个异常项，绝不使用有损回退。
 *
 * 整理自已验证 lossless_fast_vf.h，原头 SHA256：
 * 6150bb5973b205eccb9261e36a85c0c752701a9a7a5c8f4c551821e60c700d75
 */
#pragma once
#include "kernel_operator.h"
#ifndef LOSSLESS_REPAIRS
#define LOSSLESS_REPAIRS 256
#endif
namespace LosslessBF16 {
namespace R=AscendC::Reg;
// Decode two BF16 values per uint32 lane; every operation is integer and exact.
template<int E> __simd_vf__ void decode(__ubuf__ uint8_t*src,__ubuf__ uint16_t*dst){
 auto p=R::CreateMask<uint32_t,R::MaskPattern::ALL>();
 auto p16=R::CreateMask<uint16_t,R::MaskPattern::ALL>();
 R::RegTensor<uint16_t> low,m7,m80,man,sign,body;
 R::RegTensor<uint32_t> ex,base,basepair,m15,mask16,a,b,out;
 R::Duplicate(m7,(uint16_t)127);R::Duplicate(m80,(uint16_t)128);R::Duplicate(m15,15u);R::Duplicate(mask16,65535u);
 R::LoadAlign<uint32_t,R::LoadDist::DIST_BRC_B32>(base,(__ubuf__ uint32_t*)src+1);
 R::ShiftLefts(a,base,(int16_t)7,p);R::ShiftLefts(b,base,(int16_t)23,p);R::Or(basepair,a,b,p);
 for(uint16_t j=0;j<E/128;j++){
  R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK_B8>((R::RegTensor<uint8_t>&)low,src+32+j*128);
  R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK4_B8>((R::RegTensor<uint8_t>&)ex,src+32+E+j*64);
  R::And(man,low,m7,p16);R::And(sign,low,m80,p16);R::ShiftLefts(sign,sign,(int16_t)8,p16);R::Or(body,man,sign,p16);
  R::And(a,ex,m15,p);R::ShiftLefts(a,a,(int16_t)7,p);
  R::ShiftRights(b,ex,(int16_t)4,p);R::ShiftLefts(b,b,(int16_t)23,p);R::Or(out,a,b,p);
  R::Or(out,out,(R::RegTensor<uint32_t>&)body,p);R::Add(out,out,basepair,p);
  R::StoreAlign((__ubuf__ uint32_t*)dst+j*64,out,p);
 }
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
 R::RegTensor<uint32_t> repairs,ix,values,lanes,count;
 R::LoadAlign<uint32_t,R::LoadDist::DIST_BRC_B32>(count,(__ubuf__ uint32_t*)src);
 for(uint16_t chunk=0;chunk<(LOSSLESS_REPAIRS+63)/64;chunk++){
  R::LoadAlign(repairs,(__ubuf__ uint32_t*)(src+32+E*3/2)+chunk*64);
  R::And(ix,repairs,mask16,p);R::ShiftRights(values,repairs,(int16_t)16,p);
  R::Arange((R::RegTensor<int32_t>&)lanes,chunk*64);
  R::MaskReg valid;R::Compare<uint32_t,AscendC::CMPMODE::LT>(valid,lanes,count,p);
  R::Scatter<uint16_t,uint16_t>(dst,(R::RegTensor<uint16_t>&)values,(R::RegTensor<uint16_t>&)ix,valid);
 }
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
}
}
