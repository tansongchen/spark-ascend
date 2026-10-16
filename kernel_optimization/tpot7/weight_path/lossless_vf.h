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
 R::RegTensor<uint32_t> low,ex,base,m7,m80,m7hi,m80hi,m15,mask16,a,b,c,d,out;
 R::Duplicate(m7,127u);R::Duplicate(m80,128u);R::Duplicate(m7hi,0x7f00u);R::Duplicate(m80hi,0x8000u);R::Duplicate(m15,15u);R::Duplicate(mask16,65535u);
 R::LoadAlign<uint32_t,R::LoadDist::DIST_BRC_B32>(base,(__ubuf__ uint32_t*)src+1);
 for(uint16_t j=0;j<E/128;j++){
  R::LoadAlign<uint16_t,R::LoadDist::DIST_UNPACK_B16>((R::RegTensor<uint16_t>&)low,(__ubuf__ uint16_t*)(src+32)+j*64);
  R::LoadAlign<uint8_t,R::LoadDist::DIST_UNPACK4_B8>((R::RegTensor<uint8_t>&)ex,src+32+E+j*64);
  R::And(a,low,m7,p);R::And(b,low,m80,p);R::ShiftLefts(b,b,(int16_t)8,p);R::Or(out,a,b,p);
  R::And(a,low,m7hi,p);R::ShiftLefts(a,a,(int16_t)8,p);R::Or(out,out,a,p);
  R::And(a,low,m80hi,p);R::ShiftLefts(a,a,(int16_t)16,p);R::Or(out,out,a,p);
  R::And(a,ex,m15,p);R::Add(a,a,base,p);R::ShiftLefts(a,a,(int16_t)7,p);R::Or(out,out,a,p);
  R::ShiftRights(b,ex,(int16_t)4,p);R::Add(b,b,base,p);R::ShiftLefts(b,b,(int16_t)23,p);R::Or(out,out,b,p);
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
