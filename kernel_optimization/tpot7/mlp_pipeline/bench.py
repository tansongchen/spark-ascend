"""Real-weight candidate validation with interleaved baseline/null/device events.

Uses the existing CPU reference; repeated single-layer operands are a screening
workload, not full-model speed evidence. All raw launches follow a submitted start
event, and baseline/candidate share the same input/cache/output allocations.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
s=(ROOT/'kernel_validation/validate.py').read_text()
s=s.replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(ROOT)!r})')
s=s.replace('inputs[4]=tile_weights(inputs[4],8)','inputs[4]=tile_weights(inputs[4],args.up_tile)')
s=s.replace('inputs[5]=tile_weights(inputs[5],4)','inputs[5]=inputs[5].T.contiguous()')
s=s.replace('128*1024+4096','131072+56*2560*4+4096').replace('workspace[128*1024:]','workspace[131072+56*2560*4:]')
s=s.replace('    width=5120 if rd==64 else 128', '''    width=5120 if rd==64 else 128
    manifest=Path(args.build)/'execution.json'
    candidate=json.loads(manifest.read_text()).get('candidate') if manifest.exists() else None
    if rd==64 and (candidate=='balanced7_all' or (candidate=='balanced7' and args.length<=35840)):
        width=((args.length+895)//896)*128
    elif rd==64 and candidate=='balanced7_tile384':
        width=((args.length+2687)//2688)*384''')
injection='''
    import statistics
    baseline_lib=ctypes.CDLL(args.baseline_library)
    baseline=baseline_lib.launch_decode_layer_stage_prefetch
    baseline.argtypes=fn.argtypes;baseline.restype=None
    b1=tile_weights(w1,8).npu()
    b2=tile_weights(w['mlp.down_proj.weight'],4).npu()
    bptr=(ctypes.c_void_p*14)(*[t.data_ptr() for t in dev])
    bptr[4]=b1.data_ptr();bptr[5]=b2.data_ptr()
    torch.npu.synchronize()
    stream=torch.npu.current_stream()
    common=(*[c.data_ptr() for c in caches],ws.data_ptr(),y.data_ptr(),rout.data_ptr(),rd,int(args.slot32),1e-6,1e-6)
    methods={'baseline':lambda:baseline(stream.npu_stream,bptr,*common),
             'null':lambda:baseline(stream.npu_stream,bptr,*common),
             'candidate':lambda:fn(stream.npu_stream,ptrs,*common)}
    snap={}
    for name in ('baseline','candidate'):
        methods[name]();stream.synchronize()
        snap[name]={'y':y[:2560].cpu(),'resout':rout[:2560].cpu()}
    comparisons={name:{'bit_equal':torch.equal(snap['baseline'][name],snap['candidate'][name]),
                       **stats(snap['candidate'][name],snap['baseline'][name])} for name in ('y','resout')}
    for _ in range(3000):methods['baseline']()
    stream.synchronize()
    if args.profile:
        from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        from torch_npu.profiler import _ExperimentalConfig, ProfilerLevel, AiCMetrics
        with profile(activities=[ProfilerActivity.NPU],
                     on_trace_ready=tensorboard_trace_handler(args.profile),
                     experimental_config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
                                                              aic_metrics=AiCMetrics.PipeUtilization)):
            for method in ('baseline','candidate'):
                for _ in range(3): methods[method]()
                stream.synchronize()
    observations=[]
    orders=[['baseline','null','candidate'],['candidate','null','baseline']]
    for r in range(8):
        row={'round':r,'order':orders[r%2]}
        for name in row['order']:
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            start.record();stream.synchronize()
            for _ in range(1000):methods[name]()
            end.record();end.synchronize()
            row[name+'_us']=start.elapsed_time(end)*1000/1000
        observations.append(row)
    means={name:statistics.mean(r[name+'_us'] for r in observations) for name in methods}
    micro={'physical_npu':os.environ['ASCEND_RT_VISIBLE_DEVICES'],'layer':layer,'length':args.length,
           'mean_us':means,'candidate_reduction_percent':100*(1-means['candidate']/means['baseline']),
           'null_reduction_percent':100*(1-means['null']/means['baseline']),
           'baseline_library':args.baseline_library,'candidate_library':str(libpath),
           'up_tile':args.up_tile,'down_tile':args.down_tile,'comparisons':comparisons,'observations':observations,
           'scope':'same-allocation single layer, screening only; not full-model speed',
           'event_start_submitted_before_raw_launch':True}
    Path(str(args.output)+'.benchmark.json').write_text(json.dumps(micro,indent=2)+'\\n')
    print('PAIRED_BENCH',json.dumps(micro),flush=True)
    methods['candidate']();stream.synchronize()
'''
needle='    elapsed = time.monotonic()-started'
assert s.count(needle)==1
s=s.replace(needle,injection+'\n'+needle)
s=s.replace("    p.add_argument('--variant',", "    p.add_argument('--profile')\n    p.add_argument('--baseline-library',default='/data1/t00995472/spark-kernel-validation/build/libdecode_layer_stage_prefetch.so')\n    p.add_argument('--up-tile',type=int,default=8)\n    p.add_argument('--down-tile',type=int,default=4)\n    p.add_argument('--variant',",1)
exec(compile(s,str(ROOT/'kernel_validation/validate.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'kernel_validation/validate.py')})
