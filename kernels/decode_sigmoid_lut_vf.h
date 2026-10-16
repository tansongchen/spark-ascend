#pragma once
#ifndef __DAV_C310_CUBE__
namespace SigmoidLookup {
namespace R=AscendC::Reg;
__simd_vf__ void run(__ubuf__ uint16_t*x,__ubuf__ uint16_t*lut,__ubuf__ uint16_t*out,int n){
 uint32_t count=n;auto p=R::UpdateMask<uint32_t>(count);
 R::RegTensor<uint32_t>bits,abs,ix,sign,tmp,mask,extended,value,large,half,saved;
 R::MaskReg test,special;
 R::LoadAlign<uint16_t,R::LoadDist::DIST_UNPACK_B16>((R::RegTensor<uint16_t>&)bits,x);
 R::Duplicate(mask,32767u);R::And(abs,bits,mask,p);
 R::Adds(ix,abs,(uint32_t)(-0x3b00),p);R::Duplicate(mask,2047u);R::And(ix,ix,mask,p);
 R::ShiftRights(sign,bits,(int16_t)4,p);R::Duplicate(mask,2048u);R::And(sign,sign,mask,p);R::Or(ix,ix,sign,p);
 R::Duplicate(mask,127u);R::And(extended,bits,mask,p);R::ShiftRights(tmp,bits,(int16_t)8,p);R::Duplicate(mask,128u);R::And(tmp,tmp,mask,p);R::Add(extended,extended,tmp,p);R::Adds(extended,extended,4096u,p);
 R::CompareScalar<uint32_t,AscendC::CMPMODE::GE>(special,abs,0x7f80u,p);R::Select(ix,extended,ix,special);
 R::Gather((R::RegTensor<uint16_t>&)value,lut,(R::RegTensor<uint16_t>&)ix,p);saved=value;
 R::Duplicate(half,0x3f00u);R::CompareScalar<uint32_t,AscendC::CMPMODE::LT>(test,abs,0x3b00u,p);R::Select(value,half,value,test);
 R::Duplicate(large,0x3f80u);R::Duplicate(tmp,0u);R::CompareScalar<uint32_t,AscendC::CMPMODE::GE>(test,bits,32768u,p);R::Select(large,tmp,large,test);
 R::CompareScalar<uint32_t,AscendC::CMPMODE::GE>(test,abs,0x4300u,p);R::Select(value,large,value,test);R::Select(value,saved,value,special);
 R::StoreAlign<uint16_t,R::StoreDist::DIST_PACK_B32>(out,(R::RegTensor<uint16_t>&)value,p);
 R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
}
}
#endif
