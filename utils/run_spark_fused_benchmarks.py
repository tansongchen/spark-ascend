"""Sequential same-device performance comparisons with device sampling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--directory',type=Path,default=Path('/data1/t00995472/spark-fused-benchmark/performance'))
p.add_argument('--npu',type=int,default=2)
p.add_argument('--fused-graph-comparison',action='store_true',help='比较融合 eager 与融合内部图重放')
p.add_argument('--candidate-library',type=Path,help='比较原融合 eager 与兼容的 BF16 候选库')
p.add_argument('--input-tokens',type=int,default=32768)
p.add_argument('--output-tokens',type=int,default=4096)
a=p.parse_args()
if a.candidate_library and (a.fused_graph_comparison or not a.candidate_library.is_file()):
    p.error('候选库须存在，且不与 --fused-graph-comparison 同用')
if a.npu not in range(1,8):p.error('Only physical NPU 1..7')
root=a.directory;root.mkdir(parents=True,exist_ok=True)
if (root/'execution.json').exists():p.error('目录已有结果，请用 --directory 指定新目录')
env=os.environ.copy()
for key in ['ASCEND_LAUNCH_BLOCKING','SPARK_BENCH_DEBUG','SPARK_LOGITS_DIR','SPARK_FORCE_TOKENS','SPARK_FUSED_LIBRARY']:
    env.pop(key,None)
modes=[('baseline-eager',['--enforce-eager']),('fused-eager',['--enforce-eager','--fused-decode']),('baseline-graph',[])]
if a.fused_graph_comparison:
    modes=[('fused-eager',['--enforce-eager','--fused-decode']),
           ('fused-graph',['--enforce-eager','--fused-decode','--fused-graph'])]
if a.candidate_library:
    modes=[('fused-eager',['--enforce-eager','--fused-decode','--fused-library',
                          '/data1/t00995472/spark-kernel-validation/build/libdecode_layer_stage_prefetch.so']),
           ('fused-optimized',['--enforce-eager','--fused-decode','--fused-library',str(a.candidate_library.resolve())])]
summary=[]
for name,flags in modes:
    cmd=[sys.executable,'-B',str(Path(__file__).resolve().parents[1]/'benchmark.py'),'--npu',str(a.npu),
         '--input-tokens',str(a.input_tokens),'--output-tokens',str(a.output_tokens),
         '--result',str(root/f'{name}.json'),*flags]
    print('START',name,flush=True)
    started=time.time()
    with (root/f'{name}.log').open('w') as log,(root/f'{name}-health.jsonl').open('w') as health:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        while proc.poll() is None:
            check=subprocess.run(['npu-smi','info','-t','common','-i',str(a.npu)],capture_output=True,text=True)
            health.write(json.dumps({'utc':time.time(),'returncode':check.returncode,'common':check.stdout,'stderr':check.stderr})+'\n');health.flush()
            time.sleep(2)
    row={'mode':name,'returncode':proc.returncode,'start_utc':started,'end_utc':time.time(),'command':cmd}
    target=root/f'{name}.json'
    if target.exists():row['summary']=json.loads(target.read_text())['summary']
    summary.append(row);(root/'execution.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('END',json.dumps(row),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)
