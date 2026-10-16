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
