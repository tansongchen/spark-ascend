"""Use the shared CPU reference and AB/BA benchmark, with sustained windows."""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
s = (ROOT/'kernel_optimization/bench_pair.py').read_text()
s = s.replace('ROOT=Path(__file__).resolve().parents[1]', f'ROOT=Path({str(ROOT)!r})')
s = s.replace('range(100)', 'range(3000)').replace('range(8)', 'range(6)').replace('range(250)', 'range(3000)').replace('*1000/250', '*1000/3000')
rotations=int(os.environ.get('WEIGHT_ROTATIONS','1'))
if rotations > 1:
    old="""    methods={'baseline':lambda:baseline(stream.npu_stream,bptr,*common),
             'null':lambda:baseline(stream.npu_stream,bptr,*common),
             'candidate':lambda:fn(stream.npu_stream,ptrs,*common)}"""
    new="""    rotating_weights=[[dev[j].clone() for j in (2,3,4,5)] for _ in range(ROTATIONS)]
    rotating_ptrs=[]
    for tensors in rotating_weights:
        pp=(ctypes.c_void_p*14)(*[t.data_ptr() for t in dev])
        for j,t in zip((2,3,4,5),tensors):pp[j]=t.data_ptr()
        rotating_ptrs.append(pp)
    stream.synchronize()
    counts={name:0 for name in ('baseline','null','candidate')}
    def rotated(name,launch):
        pp=rotating_ptrs[counts[name]%ROTATIONS]
        counts[name]+=1
        launch(stream.npu_stream,pp,*common)
    methods={'baseline':lambda:rotated('baseline',baseline),
             'null':lambda:rotated('null',baseline),
             'candidate':lambda:rotated('candidate',fn)}""".replace('ROTATIONS',str(rotations))
    assert old in s
    s=s.replace(old,new).replace("'event_start_submitted_before_raw_launch':True",f"'rotating_weight_allocations':{rotations},'event_start_submitted_before_raw_launch':True")
samples=[]
stop=threading.Event()
physical=os.environ['ASCEND_RT_VISIBLE_DEVICES']
assert physical == '7', physical
def telemetry():
    while not stop.is_set():
        r=subprocess.run(['npu-smi','info','-i',physical,'-t','common'],capture_output=True,text=True)
        samples.append(dict(time=time.time(),returncode=r.returncode,text=r.stdout))
        stop.wait(.5)
th=threading.Thread(target=telemetry,daemon=True)
th.start()
try:
    exec(compile(s,str(ROOT/'kernel_optimization/bench_pair.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'kernel_optimization/bench_pair.py')})
finally:
    stop.set();th.join()
    out=Path(sys.argv[sys.argv.index('--output')+1]+'.health.json')
    out.write_text(json.dumps(samples,indent=2)+'\n')
