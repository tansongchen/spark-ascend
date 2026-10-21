"""Read unprofiled measurements and device samples; no device execution."""
import argparse
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--fused-graph-comparison',action='store_true')
p.add_argument('--bf16-optimization-comparison',action='store_true')
p.add_argument('--input-tokens',type=int,default=32768)
p.add_argument('--output-tokens',type=int,default=4096)
a=p.parse_args();root=a.directory
if a.fused_graph_comparison and a.bf16_optimization_comparison:
    p.error('两种对照模式不能同时使用')
names=['fused-eager','fused-graph'] if a.fused_graph_comparison else ['baseline-eager','fused-eager','baseline-graph']
candidate='fused-graph' if a.fused_graph_comparison else 'fused-eager'
baselines=['fused-eager'] if a.fused_graph_comparison else ['baseline-eager','baseline-graph']
if a.bf16_optimization_comparison:
    names=['fused-eager','fused-optimized'];candidate='fused-optimized';baselines=['fused-eager']
data={name:json.loads((root/f'{name}.json').read_text()) for name in names}
out={'candidate':candidate,'modes':{},'comparisons':{},'health':{},'same_npu':len({d['settings']['npu'] for d in data.values()})==1}
for name,d in data.items():
    if d.get('profiled') or d.get('validation_overhead'):
        raise RuntimeError('Profiled or validation result must not be used as performance evidence')
    if any((r['input_tokens'],r['output_tokens'])!=(a.input_tokens,a.output_tokens) for r in d['runs']):
        raise RuntimeError('Input/output lengths differ')
    if len(d['runs'])!=3 or d['settings']['warmup']!=1:
        raise RuntimeError('Expected one full warmup and three measurements')
    if a.fused_graph_comparison and bool(d['settings'].get('fused_graph'))!=(name=='fused-graph'):
        raise RuntimeError('Fused graph mode settings differ from result label')
    out['modes'][name]={'summary':d['summary'],'tpot_runs_ms':[r['tpot_ms'] for r in d['runs']],
                        'ttft_runs_ms':[r['ttft_ms'] for r in d['runs']],
                        'unique_output_hashes':len({r['token_ids_sha256'] for r in d['runs']})}
    if 'fused_worker_status' in d:
        out['modes'][name]['worker_status']=d['fused_worker_status']
    samples=[]
    for line in (root/f'{name}-health.jsonl').read_text().splitlines():
        row=json.loads(line);fields={key.strip():value.strip() for key,value in re.findall(r'^\s*([^:\n]+):\s*(.*)$',row['common'],re.M)}
        try:
            samples.append({'util':int(fields['Aicore Usage Rate(%)']),
                            'freq':int(fields['Aicore curFreq(MHZ)']),
                            'temp':int(fields['Temperature(C)'])})
        except (KeyError,ValueError):pass
    active=[s for s in samples if s['util']>=50]
    out['health'][name]={'sample_count':len(samples),'active_samples':len(active),
                          'active_freq_mhz':sorted({s['freq'] for s in active}),
                          'active_below_1650_count':sum(s['freq']<1650 for s in active),
                          'temperature_range_c':[min(s['temp'] for s in samples),max(s['temp'] for s in samples)] if samples else None,
                          'scope':'whole process including load and warmup; active means utilization >=50%'}
for base in baselines:
    b=data[base]['summary']['tpot_ms']['mean'];f=data[candidate]['summary']['tpot_ms']['mean']
    out['comparisons'][base]={'generation_speedup':b/f,'tpot_reduction_percent':100*(1-f/b),
                             'e2e_reduction_percent':100*(1-data[candidate]['summary']['e2e_ms']['mean']/data[base]['summary']['e2e_ms']['mean'])}
    pairs=[]
    for br,fr in zip(data[base]['runs'],data[candidate]['runs']):
        bs,fs=br['token_ids'],fr['token_ids']
        prefix=next((i for i,(x,y) in enumerate(zip(bs,fs)) if x!=y),min(len(bs),len(fs)))
        pairs.append({'exact':bs==fs,'common_prefix_tokens':prefix,
                      'same_index_match_fraction':sum(x==y for x,y in zip(bs,fs))/len(bs)})
    out['comparisons'][base]['greedy_output_pairs']=pairs
out['limitations']=['sequential mode order, not randomized/interleaved',
                    'synthetic repeated text; not a model-quality benchmark',
                    'BF16 TP=1 batch=1; outer engine eager, optional internal fused graph',
                    'observed same-device results, not a calibrated roofline']
(root/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
