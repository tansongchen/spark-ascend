"""Generate isolated BF16 attention candidates; never modify the supplied source."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--candidate', choices=['baseline', 'balanced7_all'], required=True)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
s = (ROOT / 'kernels/decode_layer_stage_prefetch.asc').read_text()
changes = []
if a.candidate == 'balanced7_all':
    old = 'int full=rd==64,width=full?5120:128;'
    new = '''int full=rd==64,width=full?5120:128;
 // A page-aligned seven-way split uses exactly 28 AIC tasks. Keep the
 // eighth partial slot (invalid), the eight-way merge, and the public ABI.
 // All supported contexts use seven balanced partitions.
 if(full){GlobalTensor<int64_t> dynamic_pos;dynamic_pos.SetGlobalBuffer((__gm__ int64_t*)pos);
  int length=(int)dynamic_pos.GetValue(0)+1;
  width=((length+7*128-1)/(7*128))*128;
 }'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    changes = ['global attention: width=ceil(length/896)*128 for all supported contexts',
               'retain eight partial slots, invalid eighth slot masked by merge']
(out / 'decode_layer_stage_prefetch.asc').write_text(s)
(out / 'execution.json').write_text(json.dumps({'execution': 'fused_attention_candidate',
                                              'candidate': a.candidate, 'changes': changes}, indent=2)+'\n')
print(out / 'decode_layer_stage_prefetch.asc')
