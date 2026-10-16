"""Actual model weight shapes, full byte verification and rotating allocation timing."""
import argparse,ctypes,hashlib,json,os,statistics,subprocess,threading,time
from pathlib import Path
import torch
import torch_npu
from safetensors import safe_open

p=argparse.ArgumentParser(__doc__)
p.add_argument('--build',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='7'
torch.set_num_threads(12);torch.npu.set_device(0)
lib=ctypes.CDLL(str(a.build/'libread_shapes.so'));fn=lib.launch_read_shapes
fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint32]*5;fn.restype=None
model=Path('/data1/t00995472/models/Spark-X2.5-4B');idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
stream=torch.npu.current_stream();samples=[];stop=threading.Event()
def telemetry():
 while not stop.is_set():
  r=subprocess.run(['npu-smi','info','-i','7','-t','common'],capture_output=True,text=True)
  samples.append(dict(time=time.time(),text=r.stdout,returncode=r.returncode));stop.wait(.5)
thread=threading.Thread(target=telemetry,daemon=True);thread.start();results=[]
try:
 for name,keys,tile_bytes in [('up',['mlp.gate_proj.weight','mlp.up_proj.weight'],40960),('down',['mlp.down_proj.weight'],81920),('out',['self_attn.out_proj.weight'],65536)]:
  chunks=[]
  for key in keys:
   key='model.layers.0.'+key
   with safe_open(model/idx[key],framework='pt') as f:chunks.append(f.get_tensor(key).flatten())
  cpu=torch.cat(chunks).contiguous().view(torch.int32);total=cpu.numel();tile=tile_bytes//4
  assert total%tile==0
  inputs=[cpu.npu() for _ in range(3)];output=torch.empty_like(inputs[0]);stream.synchronize()
  def call(i,cyclic,cached,full=0):fn(stream.npu_stream,inputs[i%3].data_ptr(),output.data_ptr(),total,tile,cyclic,cached,full)
  for cyclic in (0,1):
   for cached in (0,1):
    for repeat in range(3):
     output.fill_(-1);stream.synchronize();call(repeat,cyclic,cached,1);stream.synchronize()
     assert torch.equal(output.cpu(),cpu),(name,cyclic,cached,repeat)
    call(0,cyclic,cached);stream.synchronize();sample=output[:total//tile*8].cpu().reshape(-1,8)
    assert torch.equal(sample,cpu.reshape(-1,tile)[:,:8])
  for i in range(1000):call(i,0,0)
  stream.synchronize();obs=[]
  methods=[(0,0),(1,0),(0,1),(1,1)]
  for r in range(6):
   for cyclic,cached in methods[::1 if r%2==0 else -1]:
    start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
    start.record();stream.synchronize()
    for i in range(5000):call(i,cyclic,cached)
    end.record();end.synchronize()
    row=dict(round=r,cyclic=cyclic,cached=cached,us=start.elapsed_time(end)*1000/5000)
    obs.append(row);print(name,json.dumps(row),flush=True)
  results.append(dict(shape=name,bytes=cpu.numel()*4,tile_bytes=tile_bytes,full_byte_verification_repeats=3,rotating_allocations=3,observations=obs,means={f'cyclic{c}_cache{k}':statistics.mean(x['us'] for x in obs if x['cyclic']==c and x['cached']==k) for c,k in methods}))
  (a.output/'result.json').write_text(json.dumps(results,indent=2)+'\n')
  del inputs,output,chunks,cpu
finally:
 stop.set();thread.join();(a.output/'health.json').write_text(json.dumps(samples,indent=2)+'\n')
