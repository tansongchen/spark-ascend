"""Streaming GM->UB probe, with tiny observable per-tile writes and sample hash."""
import ctypes
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=Path('/data1/t00995472/spark-bf16-opt/bandwidth-build'))
    parser.add_argument('--output',type=Path,required=True,help='全新结果目录，避免覆盖历史证据')
    args=parser.parse_args()
    physical=os.environ.get('ASCEND_RT_VISIBLE_DEVICES','')
    if physical not in [str(i) for i in range(1,8)]:
        raise RuntimeError('请显式暴露一张已确认空闲的物理 NPU 1..7，禁止 NPU 0')
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(12)
    torch.npu.set_device(0)
    model=Path('/data1/t00995472/models/Spark-X2.5-4B')
    with safe_open(str(model/'model-00001-of-00005.safetensors'),framework='pt') as f:
        cpu=f.get_tensor('model.embedding.weight').view(torch.int32).reshape(-1)
    inputs=[cpu.npu() for _ in range(3)]
    total=cpu.numel()
    sampled=torch.empty(total//4096,8,dtype=torch.int32,device='npu')
    lib=ctypes.CDLL(str(args.build/'libstream_read.so'))
    launch=lib.launch_stream_read
    launch.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint32]*3
    launch.restype=None
    stream=torch.npu.current_stream()
    def call(i,cores,tile):
        launch(stream.npu_stream,inputs[i%3].data_ptr(),sampled.data_ptr(),cores,total,tile)
    hashes={}
    results=[]
    samples=[]
    stop=threading.Event()
    def telemetry():
        while not stop.is_set():
            r=subprocess.run(['npu-smi','info','-i',physical,'-t','common'],capture_output=True,text=True)
            samples.append(dict(time=time.time(),returncode=r.returncode,text=r.stdout))
            stop.wait(.5)
    th=threading.Thread(target=telemetry,daemon=True)
    th.start()
    try:
        for tile_bytes in (16384,32768,65536,131072):
            tile=tile_bytes//4
            assert total%tile==0
            for cores in (28,56,112):
                torch.npu.synchronize()
                call(0,cores,tile)
                torch.npu.synchronize()
                got=sampled[:total//tile].cpu()
                expected=cpu.reshape(-1,tile)[:,:8].contiguous()
                assert torch.equal(got,expected),(cores,tile_bytes)
                hashes[f'{cores}-{tile_bytes}']=hashlib.sha256(got.numpy().tobytes()).hexdigest()
                for i in range(60):
                    call(i,cores,tile)
                torch.npu.synchronize()
                timings=[]
                # Sustained >1s windows to permit loaded-frequency observations.
                for r in range(3):
                    start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                    start.record();start.synchronize()
                    for i in range(3000):
                        call(i,cores,tile)
                    end.record();end.synchronize()
                    timings.append(start.elapsed_time(end)*1000/3000)
                mean=statistics.mean(timings)
                row=dict(cores=cores,tile_bytes=tile_bytes,timings_us=timings,mean_us=mean,
                         read_bytes=total*4,output_bytes=(total//tile)*32,
                         achieved_read_TB_s=(total*4)/(mean*1e6))
                results.append(row)
                print(json.dumps(row),flush=True)
    finally:
        stop.set();th.join()
        (out/'health.json').write_text(json.dumps(samples,indent=2)+'\n')
    result=dict(physical_npu=int(physical),total_read_bytes=total*4,rotating_allocations=3,
                sample_hashes=hashes,results=results,
                limitation='Probe-specific achieved access rate, not a hardware peak or proof of optimality; data validation samples 32 bytes of every complete DMA tile.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
