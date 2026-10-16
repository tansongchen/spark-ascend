"""Generate isolated scheduling/cache candidates without changing weight packing."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser(__doc__)
p.add_argument('--candidate', choices=['baseline', 'cyclic_mlp', 'cyclic_all', 'cache_all', 'cache_mlp'], required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
src = ROOT / 'kernels/decode_layer_stage_prefetch.asc'
s = src.read_text()
if a.candidate.startswith('cache'):
    if a.candidate == 'cache_all':
        s = s.replace('.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)', '.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL)')
    else:
        before, rest = s.split('namespace FfnStage {', 1)
        ffn, after = rest.split('namespace FullStage {', 1)
        ffn = ffn.replace('.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)', '.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL)')
        s = before + 'namespace FfnStage {' + ffn + 'namespace FullStage {' + after
elif a.candidate.startswith('cyclic'):
    begin = s.index('template<int K,int M> __aicore__ inline int linear_stage')
    stop = s.index('__aicore__ inline void norm_final', begin)
    ffn = s[begin:stop]
    old = 'const int start=((N/M)*logical/56)*M,stride=M,end=((N/M)*(logical+1)/56)*M;'
    assert ffn.count(old) == 2
    condition = 'true' if a.candidate == 'cyclic_all' else 'K==10240'
    linear = f'constexpr bool cyclic={condition};const int start=cyclic?logical*M:((N/M)*logical/56)*M,stride=cyclic?56*M:M,end=cyclic?N:((N/M)*(logical+1)/56)*M;'
    ffn = ffn.replace(old, linear, 1)
    ffn = ffn.replace(old, 'const int start=logical*M,stride=56*M,end=N;', 1)
    ffn = ffn.replace('const int tiles=(end-start)/M;', 'const int tiles=(end-start+stride-1)/stride;')
    ffn = ffn.replace('row=start+i*M', 'row=start+i*stride')
    ffn = ffn.replace('(row+M)*K', '(row+stride)*K')
    ffn = ffn.replace('const int next_start=((NN/NM)*logical/56)*NM;', 'const int next_start=logical*NM;')
    ffn = ffn.replace('(uint16_t)((end-start)/stride)', '(uint16_t)tiles')
    ffn = ffn.replace('(end-start)/2);', 'tiles*4);')
    ffn = ffn.replace('DataCopyExtParams cp{(uint16_t)((end-start)/8),8,0,0,0};', 'DataCopyExtParams cp{(uint16_t)tiles,8,0,(uint32_t)((stride/2-4)*2),0};')
    s = s[:begin] + ffn + s[stop:]
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'decode_layer_stage_prefetch.asc').write_text(s)
(a.output / 'execution.json').write_text(json.dumps(dict(candidate=a.candidate, source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), unchanged_abi=True, unchanged_packing=True), indent=2)+'\n')
