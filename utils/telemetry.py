"""Save device telemetry alongside experiments without interpreting sparse samples."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import threading
import time


@contextmanager
def record_device(physical, output, interval=0.5):
    samples, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            try:
                r = subprocess.run(['npu-smi','info','-i',str(physical),'-t','common'],
                                   capture_output=True, text=True, timeout=10)
                samples.append({'time':time.time(),'returncode':r.returncode,'text':r.stdout})
            except subprocess.TimeoutExpired:
                samples.append({'time':time.time(),'error':'npu-smi timeout'})
            stop.wait(interval)
    if interval <= 0:
        yield
        return
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set(); thread.join()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(samples, indent=2)+'\n')
