#!/usr/bin/env python3
"""汇总 Spark 基准及 torch-npu profiler 导出的原始 kernel CSV。"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def quantile(values, q):
    a = sorted(values)
    if not a:
        return None
    x = (len(a)-1)*q
    lo = int(x)
    return a[lo] + (a[min(lo+1,len(a)-1)]-a[lo])*(x-lo)


def union_length(intervals):
    end = None
    total = 0.
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop-start
        elif stop > end:
            total += stop-end
        end = max(stop, end) if end is not None else stop
    return total


def analyze_csv(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    def key(*names):
        return next((n for n in names if n in columns), None)
    name = key('Name', 'Op Name', 'OP Name')
    duration = key('Duration(us)', 'Duration (us)', 'Task Duration(us)')
    start = key('Start Time(us)', 'Start Time (us)', 'Task Start Time(us)')
    if name is None or duration is None:
        return {'path':str(path), 'columns':columns, 'error':'unknown schema'}
    groups = defaultdict(list)
    intervals = []
    invalid_duration = 0
    time_origin = None
    for row in rows:
        try:
            d = float(row[duration])
            if not math.isfinite(d) or d < 0:
                raise ValueError()
        except (ValueError, TypeError):
            invalid_duration += 1
            continue
        groups[row[name]].append(d)
        if start:
            try:
                timestamp = Decimal(row[start].strip())
                if time_origin is None:
                    time_origin = timestamp
                t = float(timestamp - time_origin)
                if math.isfinite(t):
                    intervals.append((t,t+d))
            except (ValueError, TypeError):
                pass
    ops = [{'name':n,'count':len(ds),'sum_us':sum(ds),'mean_us':sum(ds)/len(ds)} for n,ds in groups.items()]
    ops.sort(key=lambda x:x['sum_us'],reverse=True)
    out = {'path':str(path),'columns':columns,'row_count':len(rows),'invalid_durations':invalid_duration,'ops':ops,'task_sum_us':sum(o['sum_us'] for o in ops)}
    if intervals:
        span = max(t[1] for t in intervals)-min(t[0] for t in intervals)
        busy = union_length(intervals)
        out.update(span_us=span,union_busy_us=busy,uncovered_us=span-busy,valid_intervals=len(intervals))
    counts = {n:len(ds) for n,ds in groups.items()}
    steps = counts.get('aclnnEmbedding_GatherV2AiCore_GatherV2',0)
    expected = {'FusedInferAttentionScore':36*steps,
                'aclnnMatmul_MatMulV3Common_MatMulV3':181*steps,
                '_triton_rope':36*steps,
                'aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2':steps}
    out['reconciliation'] = {'embedding_steps':steps,'expected':expected,
        'actual':{n:counts.get(n,0) for n in expected},
        'passed':steps==16 and all(counts.get(n,0)==c for n,c in expected.items())}
    shapes = defaultdict(list)
    for r in rows:
        if r[name]=='aclnnMatmul_MatMulV3Common_MatMulV3':
            shapes[r.get('Input Shapes','')].append(r)
    def detail(rs):
        ds = [float(r[duration]) for r in rs]
        d = {'count':len(rs),'sum_us':sum(ds),'mean_us':sum(ds)/len(ds)}
        for base, metrics in [('aicore_time(us)',['aic_mac_time(us)','aic_mte2_time(us)','aic_scalar_time(us)']),
                              ('aiv_time(us)',['aiv_vec_time(us)','aiv_mte2_time(us)','aiv_scalar_time(us)'])]:
            def total(field):
                try:
                    vals=[float(r[field]) for r in rs]
                    return sum(vals) if all(math.isfinite(v) for v in vals) else None
                except (KeyError,ValueError,TypeError):
                    return None
            den=total(base)
            for m in metrics:
                num=total(m)
                d[m+'/sum_'+base] = num/den if num is not None and den is not None and den>0 else None
        return d
    out['matmul_shapes'] = {s:detail(rs) for s,rs in shapes.items()}
    attention = [r for r in rows if r[name]=='FusedInferAttentionScore']
    if start:
        attention.sort(key=lambda r:Decimal(r[start].strip()))
    if len(attention)==36*steps and steps:
        out['attention_by_layer_order'] = {
            'full_every_fourth':detail([r for i,r in enumerate(attention) if i%4==3]),
            'sliding_others':detail([r for i,r in enumerate(attention) if i%4!=3])}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    a = parser.parse_args()
    out = {'benchmarks':{},'profiles':[], 'health':{}}
    for p in sorted(a.root.glob('*-health.jsonl')):
        samples = [json.loads(line) for line in p.read_text().splitlines()]
        parsed = []
        for row in samples:
            fields = {}
            for line in row['common'].splitlines():
                if ':' in line:
                    k,v = line.split(':',1)
                    try:
                        fields[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
            parsed.append(fields)
        active = [r for r in parsed if r.get('Aicore Usage Rate(%)',0)>0]
        out['health'][p.stem] = {'sample_count':len(samples),'active_count':len(active),
            'query_failures':sum(r['returncode']!=0 for r in samples),
            'active_current_frequencies_mhz':sorted({r['Aicore curFreq(MHZ)'] for r in active if 'Aicore curFreq(MHZ)' in r}),
            'ranges':{k:[min(r[k] for r in parsed if k in r),max(r[k] for r in parsed if k in r)]
                      for k in ['Temperature(C)','NPU Real-time Power(W)'] if any(k in r for r in parsed)}}
    for name in ['eager','graph','profile-eager','profile-graph']:
        p = a.root/(name+'.json')
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        result = {'summary':data['summary'],'profiled':data.get('profiled',False),'runs':[]}
        for run in data['runs']:
            arrivals = run.get('token_arrivals_s',[])
            itl = [(y-x)*1000 for x,y in zip(arrivals,arrivals[1:])]
            result['runs'].append({'tpot_ms':run['tpot_ms'],'token_ids_sha256':run.get('token_ids_sha256'),'itl_p50_ms':quantile(itl,.5),'itl_p95_ms':quantile(itl,.95),'itl_p99_ms':quantile(itl,.99)})
        out['benchmarks'][name] = result
    for p in sorted(a.root.glob('profile-*/**/kernel_details.csv')):
        out['profiles'].append(analyze_csv(p))
    (a.root/'analysis.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    for name,data in out['benchmarks'].items():
        print(name, json.dumps(data,ensure_ascii=False))
    for data in out['profiles']:
        print('PROFILE',data['path'],{k:v for k,v in data.items() if k not in ['ops','columns','path']})
        for op in data.get('ops',[])[:20]:
            print(op)


if __name__ == '__main__':
    main()
