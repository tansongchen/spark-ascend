#pragma once
#include "w4a16_reborn_linear.h"
#include "w4a16_reborn_kv8.h"

namespace Reborn {
// All offsets are bytes. Keep stage outputs compatible with the CPU oracle.
constexpr int Scratch=131072, NormInput=300032, GateParts=305152;
constexpr int QkvBytes=6144*2560*33/64;
constexpr SyncAllConfig syncConfig{PIPE_MTE3,PIPE_MTE2};
__aicore__ inline void Barrier(TPipe& pipe) {
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID7);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID7);
    SyncAll<true,syncConfig>(); pipe.Reset();
}

template<int Splits,int Tiles>
__simd_vf__ void MergeLinear(__ubuf__ float* p,__ubuf__ bfloat16_t* y) {
    auto m=R::CreateMask<float,R::MaskPattern::ALL>();
    for(uint16_t t=0;t<Tiles;++t) for(uint16_t j=0;j<2;++j) {
        R::RegTensor<float> v,a; R::RegTensor<bfloat16_t> b;
        R::Duplicate(v,0.f);
        for(uint16_t s=0;s<Splits;++s) {
            R::LoadAlign(a,p+(t*Splits+s)*128+j*64); R::Add(v,v,a,m);
        }
        R::Cast<bfloat16_t,float,narrow>(b,v,m);
        R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+t*128+j*64,b,m);
    }
}

__simd_vf__ void Rope(__ubuf__ bfloat16_t* y,__ubuf__ bfloat16_t* cs,int rd) {
    R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
    uint32_t length=rd/2; auto m=R::UpdateMask<float>(length);
    for(uint16_t j=0;j<rd/2;j+=64) {
        R::RegTensor<bfloat16_t> ab,bb,cb,sb,ob;
        R::RegTensor<float>a,b,c,s,x,z,tmp;
        R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(ab,y+j);
        R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(bb,y+rd/2+j);
        R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(cb,cs+j);
        R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(sb,cs+rd/2+j);
        R::Cast<float,bfloat16_t,widen>(a,ab,m); R::Cast<float,bfloat16_t,widen>(b,bb,m);
        R::Cast<float,bfloat16_t,widen>(c,cb,m); R::Cast<float,bfloat16_t,widen>(s,sb,m);
        R::Mul(x,a,c,m); R::Mul(tmp,b,s,m); R::Sub(x,x,tmp,m);
        R::Mul(z,b,c,m); R::Mul(tmp,a,s,m); R::Add(z,z,tmp,m);
        R::Cast<bfloat16_t,float,narrow>(ob,x,m);
        R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+j,ob,m);
        R::Cast<bfloat16_t,float,narrow>(ob,z,m);
        R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+rd/2+j,ob,m);
    }
}

__aicore__ inline void QkvPost(TPipe& pipe,GM_ADDR x,GM_ADDR wq,GM_ADDR pos,
    GM_ADDR cs,GM_ADDR slots,GM_ADDR kc,GM_ADDR vc,GM_ADDR nk,GM_ADDR nv,
    GM_ADDR ws,int rd,int slot32) {
    int id=GetBlockIdx();
    if(id<24) {
        TBuf<TPosition::VECCALC> pb,yb,cb;
        pipe.InitBuffer(pb,2048); pipe.InitBuffer(yb,512); pipe.InitBuffer(cb,512);
        auto p=pb.Get<float>(); auto y=yb.Get<bfloat16_t>();auto c=cb.Get<bfloat16_t>();
        GlobalTensor<float> pg; pg.SetGlobalBuffer((__gm__ float*)(ws+Scratch));
        GlobalTensor<bfloat16_t> cg,yg;cg.SetGlobalBuffer((__gm__ bfloat16_t*)cs);
        yg.SetGlobalBuffer((__gm__ bfloat16_t*)ws);
        GlobalTensor<int64_t> pp;pp.SetGlobalBuffer((__gm__ int64_t*)pos);
        int position=(int)pp.GetValue(0);
        DataCopy(p,pg[id*512],512);DataCopy(c,cg[position*rd],rd);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        MergeLinear<2,2>((__ubuf__ float*)p.GetPhyAddr(),(__ubuf__ bfloat16_t*)y.GetPhyAddr());
        if(id<20) Rope((__ubuf__ bfloat16_t*)y.GetPhyAddr(),(__ubuf__ bfloat16_t*)c.GetPhyAddr(),rd);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(yg[id*256],y,256);
        if(id>=16) {
            int64_t slot;
            if(slot32){GlobalTensor<int32_t> t;t.SetGlobalBuffer((__gm__ int32_t*)slots);slot=t.GetValue(0);}
            else {GlobalTensor<int64_t> t;t.SetGlobalBuffer((__gm__ int64_t*)slots);slot=t.GetValue(0);}
            int head=(id-16)%4;
            GlobalTensor<bfloat16_t> cache,nz;
            cache.SetGlobalBuffer((__gm__ bfloat16_t*)(id<20?kc:vc));
            DataCopy(cache[slot*1024+head*256],y,256);
            if(rd==64) {
                auto encoded=pb.Get<uint8_t>();
                Kv8::Encode((__ubuf__ bfloat16_t*)y.GetPhyAddr(),(__ubuf__ uint8_t*)encoded.GetPhyAddr(),256);
                SetFlag<HardEvent::V_S>(1);WaitFlag<HardEvent::V_S>(1);
                auto words=pb.Get<uint64_t>();
                // Pad each 16-byte dimension fragment to one aligned UB block.
                for(int dim=15;dim>=0;--dim) {
                    uint64_t a=words.GetValue(dim*2),b=words.GetValue(dim*2+1);
                    words.SetValue(dim*4,a);words.SetValue(dim*4+1,b);
                }
                SetFlag<HardEvent::S_MTE3>(1);WaitFlag<HardEvent::S_MTE3>(1);
                GlobalTensor<uint8_t> cache8;cache8.SetGlobalBuffer((__gm__ uint8_t*)(id<20?nk:nv));
                DataCopyExtParams cp{16,16,0,2032,0};
                DataCopyPad(cache8[(slot/128)*131072+head*32768+(slot%128)*16],encoded,cp);
            }
        }
    } else if(id<40) {
        int head=id-24;
        TBuf<TPosition::VECCALC> xb,wb,xf,wf,tmp,yb;
        pipe.InitBuffer(xb,5120);pipe.InitBuffer(wb,5120);
        pipe.InitBuffer(xf,10240);pipe.InitBuffer(wf,10240);
        pipe.InitBuffer(tmp,512);pipe.InitBuffer(yb,32);
        GlobalTensor<bfloat16_t> xg,wg;GlobalTensor<float> yg;
        xg.SetGlobalBuffer((__gm__ bfloat16_t*)x);wg.SetGlobalBuffer((__gm__ bfloat16_t*)(wq+QkvBytes));
        yg.SetGlobalBuffer((__gm__ float*)(ws+GateParts));
        auto a=xb.Get<bfloat16_t>(),b=wb.Get<bfloat16_t>();
        auto af=xf.Get<float>(),bf=wf.Get<float>(),t=tmp.Get<float>(),y=yb.Get<float>();
        DataCopy(a,xg,2560);DataCopy(b,wg[head*2560],2560);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        Cast(af,a,RoundMode::CAST_NONE,2560);Cast(bf,b,RoundMode::CAST_NONE,2560);
        Mul(af,af,bf,2560);ReduceSum(y,af,t,2560);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(yg[head*8],y,8);
    }
}

__simd_vf__ void AttentionMerge(__ubuf__ bfloat16_t* part,__ubuf__ float* lse,
    __ubuf__ float* gates,__ubuf__ bfloat16_t* y,__ubuf__ bfloat16_t* gateOut,int head) {
    auto m=R::CreateMask<float,R::MaskPattern::ALL>();uint32_t eight=8;auto m8=R::UpdateMask<float>(eight);
    R::RegTensor<int32_t> ix;R::RegTensor<float> w,mx,sum,g,one;
    R::RegTensor<bfloat16_t> b;
    R::Arange(ix,0);R::Muls(ix,ix,64,m8);R::Adds(ix,ix,(head/2)*8+head%2,m8);
    R::Gather(w,lse,(R::RegTensor<uint32_t>&)ix,m8);R::Reduce<R::ReduceType::MAX>(mx,w,m8);R::Duplicate(mx,mx,m8);
    R::Sub(w,w,mx,m8);R::Exp(w,w,m8);R::Reduce<R::ReduceType::SUM>(sum,w,m8);
    R::Duplicate(sum,sum,m8);R::Div(w,w,sum,m8);R::StoreAlign(lse,w,m8);
    R::LoadAlign<float,R::LoadDist::DIST_BRC_B32>(g,gates+head*8);
    R::Cast<bfloat16_t,float,narrow>(b,g,m);R::Cast<float,bfloat16_t,widen>(g,b,m);
    R::Muls(g,g,-1.f,m);R::Exp(g,g,m);R::Adds(g,g,1.f,m);R::Duplicate(one,1.f);R::Div(g,one,g,m);
    R::Cast<bfloat16_t,float,narrow>(b,g,m);R::Cast<float,bfloat16_t,widen>(g,b,m);
    R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
    for(uint16_t j=0;j<4;++j) {
        R::RegTensor<float>a,s,acc;R::Duplicate(acc,0.f);
        for(uint16_t p=0;p<8;++p) {
            R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,part+p*256+j*64);
            R::Cast<float,bfloat16_t,widen>(a,b,m);R::LoadAlign<float,R::LoadDist::DIST_BRC_B32>(s,lse+p);
            R::MulAddDst(acc,a,s,m);
        }
        R::Cast<bfloat16_t,float,narrow>(b,acc,m);R::Cast<float,bfloat16_t,widen>(acc,b,m);
        R::Mul(acc,acc,g,m);R::Cast<bfloat16_t,float,narrow>(b,acc,m);
        R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(y+j*64,b,m);
    }
    if(head==0) {
        uint32_t sixteen=16;auto m16=R::UpdateMask<float>(sixteen);
        R::Arange(ix,0);R::Muls(ix,ix,8,m16);R::Gather(g,gates,(R::RegTensor<uint32_t>&)ix,m16);
        R::Cast<bfloat16_t,float,narrow>(b,g,m16);
        R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(gateOut,b,m16);
    }
}

__aicore__ inline void MergeAttention(TPipe& pipe,GM_ADDR ws) {
    int head=GetBlockIdx();if(head>=16)return;
    TBuf<TPosition::VECCALC> pb,lb,gb,yb;
    pipe.InitBuffer(pb,4096);pipe.InitBuffer(lb,2048);pipe.InitBuffer(gb,512);pipe.InitBuffer(yb,544);
    auto p=pb.Get<bfloat16_t>(),y=yb.Get<bfloat16_t>();auto l=lb.Get<float>(),g=gb.Get<float>();
    GlobalTensor<bfloat16_t> pg,yg;GlobalTensor<float> lg,gg;
    pg.SetGlobalBuffer((__gm__ bfloat16_t*)(ws+12544));yg.SetGlobalBuffer((__gm__ bfloat16_t*)ws);
    lg.SetGlobalBuffer((__gm__ float*)(ws+78080));gg.SetGlobalBuffer((__gm__ float*)(ws+GateParts));
    DataCopyExtParams cp{8,512,7680,0,0};
    DataCopyPad(p,pg[head*256],cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
    DataCopy(l,lg,512);DataCopy(g,gg,128);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    AttentionMerge((__ubuf__ bfloat16_t*)p.GetPhyAddr(),(__ubuf__ float*)l.GetPhyAddr(),
        (__ubuf__ float*)g.GetPhyAddr(),(__ubuf__ bfloat16_t*)y.GetPhyAddr(),(__ubuf__ bfloat16_t*)y.GetPhyAddr()+256,head);
    SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(yg[80128/2+head*256],y,256);if(head==0)DataCopy(yg[12288/2],y[256],16);
}

template<int Splits>
__aicore__ inline void Norm(TPipe& pipe,GM_ADDR partial,GM_ADDR residual,GM_ADDR gamma,
    GM_ADDR raw,GM_ADDR norm,GM_ADDR resout,float eps) {
    if(GetBlockIdx()!=0)return;
    TBuf<TPosition::VECCALC> pb,bb,fb,tb;
    pipe.InitBuffer(pb,2560*Splits*4);pipe.InitBuffer(bb,2560*2*4);
    pipe.InitBuffer(fb,2560*4*3);pipe.InitBuffer(tb,512);
    auto p=pb.Get<float>();auto b=bb.Get<bfloat16_t>();auto f=fb.Get<float>();auto t=tb.Get<float>();
    GlobalTensor<float> pg;pg.SetGlobalBuffer((__gm__ float*)partial);
    GlobalTensor<bfloat16_t> rg,gg,rawg,ng,og;
    rg.SetGlobalBuffer((__gm__ bfloat16_t*)residual);gg.SetGlobalBuffer((__gm__ bfloat16_t*)gamma);
    rawg.SetGlobalBuffer((__gm__ bfloat16_t*)raw);ng.SetGlobalBuffer((__gm__ bfloat16_t*)norm);
    og.SetGlobalBuffer((__gm__ bfloat16_t*)resout);
    DataCopy(p,pg,2560*Splits);DataCopy(b[2560],rg,2560);DataCopy(b[5120],gg,2560);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    MergeLinear<Splits,20>((__ubuf__ float*)p.GetPhyAddr(),(__ubuf__ bfloat16_t*)b.GetPhyAddr());
    Cast(f,b,RoundMode::CAST_NONE,2560);Cast(f[2560],b[2560],RoundMode::CAST_NONE,2560);
    Add(f,f,f[2560],2560);Cast(b[7680],f,RoundMode::CAST_RINT,2560);
    Mul(f[2560],f,f,2560);ReduceSum(t,f[2560],t[8],2560);
    Muls(t,t,1.f/2560.f,1);Adds(t,t,eps,1);Rsqrt(t,t,1);
    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    float inv=t.GetValue(0);
    Cast(f[5120],b[5120],RoundMode::CAST_NONE,2560);
    Muls(f,f,inv,2560);Mul(f,f,f[5120],2560);Cast(b[2560],f,RoundMode::CAST_RINT,2560);
    SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(rawg,b,2560);DataCopy(ng,b[2560],2560);DataCopy(og,b[7680],2560);
}

__simd_vf__ void GeluProduct(__ubuf__ bfloat16_t* gate,__ubuf__ bfloat16_t* up,
                           __ubuf__ bfloat16_t* out) {
    R::LocalMemBar<R::MemType::VEC_STORE,R::MemType::VEC_LOAD>();
    auto m=R::CreateMask<float,R::MaskPattern::ALL>();
    for(uint16_t j=0;j<2;++j) {
        R::RegTensor<bfloat16_t>b;R::RegTensor<float>x,u,t,z,one;
        R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,gate+j*64);R::Cast<float,bfloat16_t,widen>(x,b,m);
        R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(b,up+j*64);R::Cast<float,bfloat16_t,widen>(u,b,m);
        R::Mul(t,x,x,m);R::Muls(t,t,.044715f,m);R::Adds(t,t,1.f,m);R::Mul(t,t,x,m);
        R::Muls(t,t,-1.5957691216f,m);R::Exp(t,t,m);R::Adds(t,t,1.f,m);R::Div(z,x,t,m);
        R::Cast<bfloat16_t,float,narrow>(b,z,m);R::Cast<float,bfloat16_t,widen>(z,b,m);
        R::Mul(z,z,u,m);R::Cast<bfloat16_t,float,narrow>(b,z,m);
        R::StoreAlign<bfloat16_t,R::StoreDist::DIST_PACK_B32>(out+j*64,b,m);
    }
}
__aicore__ inline void UpPost(TPipe& pipe,GM_ADDR ws) {
    TBuf<TPosition::VECCALC> pb,bb;pipe.InitBuffer(pb,2048);pipe.InitBuffer(bb,768);
    auto p=pb.Get<float>();auto b=bb.Get<bfloat16_t>();
    GlobalTensor<float> pg;GlobalTensor<bfloat16_t> yg;
    pg.SetGlobalBuffer((__gm__ float*)(ws+Scratch));yg.SetGlobalBuffer((__gm__ bfloat16_t*)(ws+93440));
    for(int tile=GetBlockIdx();tile<80;tile+=56) {
        DataCopy(p,pg[tile*256],256);DataCopy(p[256],pg[(tile+80)*256],256);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        MergeLinear<2,2>((__ubuf__ float*)p.GetPhyAddr(),(__ubuf__ bfloat16_t*)b.GetPhyAddr());
        GeluProduct((__ubuf__ bfloat16_t*)b.GetPhyAddr(),(__ubuf__ bfloat16_t*)b.GetPhyAddr()+128,
                    (__ubuf__ bfloat16_t*)b.GetPhyAddr()+256);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(yg[tile*128],b[256],128);
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
    }
}
} // namespace Reborn
