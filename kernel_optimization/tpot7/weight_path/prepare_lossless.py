"""Generate the exact BF16 storage-decoding candidate; QKV/O remain plain BF16."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--repairs',type=int,default=256);p.add_argument('--fast',action='store_true');a=p.parse_args()
assert a.repairs in (32,64,128,256)
src=ROOT/'kernels/decode_layer_stage_prefetch.asc';s=src.read_text()
begin=s.index('namespace FfnStage {');end=s.index('namespace FullStage {',begin)
f=s[begin:end]
f=f.replace('template<int K,int M> __simd_vf__ void gemv', 'template<int K,int M,bool DIRECT=false> __simd_vf__ void gemv')
f=f.replace('  R::LoadAlign(a,x+j*64);','  if constexpr(DIRECT){R::LoadAlign<bfloat16_t,R::LoadDist::DIST_UNPACK_B16>(ab,(__ubuf__ bfloat16_t*)x+j*64);R::Cast<float,bfloat16_t,up>(a,ab,p);}else R::LoadAlign(a,x+j*64);')
f=f.replace('weights,yb,xb,tb;', 'weights,yb,xb,tb,decoded;')
f=f.replace('weights,yb,xb,tb,lb,rb,gb;', 'weights,yb,xb,tb,lb,rb,gb,decoded;')
assert f.count('pipe.InitBuffer(weights,163840);')==2
f=f.replace('pipe.InitBuffer(weights,163840);','pipe.InitBuffer(weights,131072);pipe.InitBuffer(decoded,81920);')
f=f.replace('pipe.InitBuffer(xb,K*6);','pipe.InitBuffer(xb,K==10240?K*2:K*6);')
f=f.replace('auto xf=xb.Get<float>()[K/2];','auto xf=xb.Get<float>()[K==10240?0:K/2];')
f=f.replace(' expand_input((__ubuf__ bfloat16_t*)x.GetPhyAddr(),(__ubuf__ float*)xf.GetPhyAddr(),K);',' if constexpr(K!=10240)expand_input((__ubuf__ bfloat16_t*)x.GetPhyAddr(),(__ubuf__ float*)xf.GetPhyAddr(),K);')
f=f.replace('first_slot*40960','first_slot*32768').replace('other*40960','other*32768').replace('slot*40960','slot*32768')
split=f.index('__simd_vf__ void activate160')
linear,up=f[:split],f[split:]
linear=linear.replace(' const int start=', ' constexpr int WE=K*M,WL=K==10240?WE*3/4+80:WE;\n const int start=',1)
linear=linear.replace('b[(int64_t)start*K],M*K', 'b[(int64_t)(start/M)*WL],WL')
linear=linear.replace('b[(int64_t)(row+M)*K],M*K', 'b[(int64_t)((row+M)/M)*WL],WL')
linear=linear.replace('  gemv<K,M>(', '  if constexpr(K==10240){LosslessBF16::decode<40960>((__ubuf__ uint8_t*)w.GetPhyAddr(),(__ubuf__ uint16_t*)decoded.Get<bfloat16_t>().GetPhyAddr());w=decoded.Get<bfloat16_t>();}\n  gemv<K,M,(K==10240)>(')
up=up.replace('b[(int64_t)(row+M)*K],M*K','b[(int64_t)((row+M)/M)*(M*K*3/4+80)],M*K*3/4+80')
up=up.replace('  gemv<K,M>(', '  LosslessBF16::decode<20480>((__ubuf__ uint8_t*)w.GetPhyAddr(),(__ubuf__ uint16_t*)decoded.Get<bfloat16_t>().GetPhyAddr());w=decoded.Get<bfloat16_t>();\n  gemv<K,M>(')
f=linear+up
assert f.count('nw[(int64_t)next_start*NK],NM*NK')==2
f=f.replace('nw[(int64_t)next_start*NK],NM*NK','nw[(int64_t)(next_start/NM)*(NM*NK*3/4+80)],NM*NK*3/4+80')
f=f.replace('3/4+80',f'3/4+{16+a.repairs*2}')
f=f.replace('two 80 KiB slots','two 64 KiB packed slots')
s=s[:begin]+f+s[end:]
s=f'#define LOSSLESS_REPAIRS {a.repairs}\n#include "lossless_vf.h"\n'+s
# Reuse the independent entry point in the same binary for bit-level verification.
s+='\n'+(ROOT/'kernel_optimization/tpot7/weight_path/codec_probe.asc').read_text()
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'decode_layer_stage_prefetch.asc').write_text(s)
(a.output/'lossless_vf.h').write_bytes((ROOT/('kernel_optimization/tpot7/weight_path/lossless_fast_vf.h' if a.fast else 'kernel_optimization/tpot7/weight_path/lossless_vf.h')).read_bytes())
(a.output/'execution.json').write_text(json.dumps(dict(candidate='lossless_mlp_fast' if a.fast else 'lossless_mlp',weight_encoding='bf16_lossless12',workspace_bytes=131072,codec_base=109,codec_repairs=a.repairs,unchanged_abi=True,unchanged_qkv_o_packing=True,baseline_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),generated_source_sha256=hashlib.sha256(s.encode()).hexdigest(),decoder_header_sha256=hashlib.sha256((a.output/'lossless_vf.h').read_bytes()).hexdigest(),ub_allocation_bytes=dict(out=239808,up=250304,down=234880),ub_header_limit_bytes=253952),indent=2)+'\n')
