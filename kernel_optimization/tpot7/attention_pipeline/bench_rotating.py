"""Nine KV banks for a more streaming-like global-attention screening workload.

Still one real-weight layer, not full-model TPOT. Shared validation stays unchanged.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
s=(ROOT/'kernel_optimization/bench_pair.py').read_text()
s=s.replace('ROOT=Path(__file__).resolve().parents[1]',f'ROOT=Path({str(ROOT)!r})')
old="""    methods={'baseline':lambda:baseline(stream.npu_stream,bptr,*common),
             'null':lambda:baseline(stream.npu_stream,bptr,*common),
             'candidate':lambda:fn(stream.npu_stream,ptrs,*common)}"""
new="""    cache_banks=[caches]+[[t.clone() for t in caches] for _ in range(8)]
    torch.npu.synchronize()
    cursors={'baseline':0,'null':0,'candidate':0}
    def call_rotating(name):
        bank=cache_banks[cursors[name]%len(cache_banks)];cursors[name]+=1
        launch=fn if name=='candidate' else baseline
        pointers=ptrs if name=='candidate' else bptr
        launch(stream.npu_stream,pointers,*[c.data_ptr() for c in bank],ws.data_ptr(),y.data_ptr(),rout.data_ptr(),rd,int(args.slot32),1e-6,1e-6)
    methods={name:(lambda name=name:call_rotating(name)) for name in cursors}"""
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("'scope':'same-allocation single layer, screening only; not full-model speed'", "'scope':'nine rotating KV banks, same real-weight layer; screening only, not full-model speed','cache_banks':9")
exec(compile(s,str(ROOT/'kernel_optimization/bench_pair.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'kernel_optimization/bench_pair.py')})
