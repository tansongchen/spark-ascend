"""Run independent cases sequentially, with per-process timeouts and durable results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--output',default='/data1/t00995472/spark-kernel-validation/suite')
p.add_argument('--model',required=True,help='显式 BF16 checkpoint')
p.add_argument('--build',default='/data1/t00995472/spark-kernel-validation/build')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
cases=[(0,n) for n in [1,127,128,129,512,513,514,32768]]
cases += [(3,n) for n in [1,127,128,129,5120,5121,32768,36864,40960]]
rows=[]
for i,(layer,length) in enumerate(cases):
    label=f'l{layer}-n{length}'
    target=out/f'{label}.json'
    cmd=[sys.executable,str(Path(__file__).with_name('validate_kernel.py')),'--build',a.build,
         '--model',a.model,'--layer',str(layer),'--length',str(length),'--output',str(target)]
    if i%2:
        cmd+=['--slot32']
    started=time.monotonic()
    with (out/f'{label}.log').open('w') as log:
        # GNU timeout escalates TERM to KILL if a device teardown cannot finish.
        result=subprocess.run(['timeout','-k','5s','100s',*cmd],stdout=log,stderr=subprocess.STDOUT)
    row={'layer':layer,'length':length,'slot32':bool(i%2),'returncode':result.returncode,
         'elapsed_s':time.monotonic()-started,'result':str(target)}
    if target.exists():
        data=json.loads(target.read_text())
        row.update(passed=data['pass'],y_nrmse=data['stages']['y'].get('nrmse'),
                   failed_stages=[k for k,v in data['stages'].items() if not v['pass']])
    else:
        row['passed']=False
    rows.append(row)
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    # A runtime fault can leave a device task pending; do not pile up more work.
    if result.returncode in (124,137,-9):
        print('STOP: device case timed out',flush=True)
        break
raise SystemExit(0 if len(rows)==len(cases) and all(r['passed'] for r in rows) else 1)
