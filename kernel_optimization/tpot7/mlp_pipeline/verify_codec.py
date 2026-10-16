"""Independent raw-bit coverage for the lossless BF16 storage codec.

Exhaustive BF16 codes are distributed across tiles obeying the bounded repair
contract. CPU header validation and capacity refusal are tested separately.
Optional native VF verification uses its public decode-only export.
"""
import argparse,ctypes,hashlib,json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from kernel_optimization.tpot7.lossless.codec import pack_weight,unpack_weight,tile_stride,REPAIRS


def digest(t): return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def corpus(elements):
    allbits=torch.arange(65536,dtype=torch.int32)
    exp=(allbits>>7)&255
    common=allbits[(exp>=109)&(exp<125)]
    rare=allbits[(exp<109)|(exp>=125)].reshape(-1,REPAIRS)
    regular=common[torch.arange(elements)%common.numel()]
    source=regular.repeat(len(rare),1).to(torch.int16)
    positions=torch.tensor([0,1,2,3,31,32,63,64,65,127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048,2049,4095,4096,4097,elements-5,elements-4,elements-3,elements-2,elements-1])
    positions=torch.tensor(list(dict.fromkeys(positions.tolist()+list(range(elements)))))[:REPAIRS]
    source[:,positions]=rare.to(torch.int16)
    # A no-repair tile and each intermediate repair count exercise the masked scatter.
    tails=regular.repeat(REPAIRS,1).to(torch.int16)
    for i in range(REPAIRS): tails[i,positions[:i]]=rare[i%len(rare),:i].to(torch.int16)
    source=torch.cat((source,tails),dim=0)
    coverage=torch.bincount((source.to(torch.int32)&65535).flatten(),minlength=65536)
    assert bool((coverage>0).all())
    return source,positions


def expect_reject(fn):
    try:fn()
    except ValueError as e:return str(e)
    raise AssertionError('invalid encoding accepted')


def verify_negative(source,packed,elements):
    negatives={}
    overflowing=source[:1].clone();bits=overflowing[0].to(torch.int32)&65535;ex=(bits>>7)&255;extra=int(((ex>=109)&(ex<125)).nonzero()[0]);overflowing[0,extra]=0
    negatives[f'{REPAIRS+1}_repairs']=expect_reject(lambda:pack_weight(overflowing.view(torch.bfloat16),elements,1))
    for name,column,value in [('bad_magic',2,0),('bad_elements',3,elements+64),('count_negative',0,-1),('count_excess',0,REPAIRS+1),('base_negative',1,-1),('base_241',1,241)]:
        bad=packed[:1].clone();bad[:,:32].view(torch.int32)[0,column]=value
        negatives[name]=expect_reject(lambda:unpack_weight(bad,elements,1))
    bad=packed[:1].clone();offset=32+elements*3//2;bad[0,offset]=elements&255;bad[0,offset+1]=elements>>8
    negatives['index_oob']=expect_reject(lambda:unpack_weight(bad,elements,1))
    return negatives


def verify_native(source,packed,elements,library):
    import torch_npu
    physical=os.environ.get('ASCEND_RT_VISIBLE_DEVICES','')
    if physical not in [str(i) for i in range(1,8)]: raise RuntimeError('explicit checked physical NPU1..7 required')
    torch.npu.set_device(0);stream=torch.npu.current_stream()
    lib=ctypes.CDLL(library);fn=lib.launch_decode_tile
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32];fn.restype=None
    acl=ctypes.CDLL('libascendcl.so');acl.aclrtSetOpExecuteTimeOut.argtypes=[ctypes.c_uint32];assert acl.aclrtSetOpExecuteTimeOut(10)==0
    rows=0;guards=True;exact=True
    for begin in range(0,len(source),64):
        inp=packed[begin:begin+64].npu();n=len(inp);out=torch.full((n,elements+64),-23131,dtype=torch.int16,device='npu');torch.npu.synchronize()
        for i in range(n):fn(stream.npu_stream,inp.data_ptr()+i*tile_stride(elements),out.data_ptr()+(i*(elements+64)+32)*2,elements)
        stream.synchronize();got=out.cpu()
        exact &= torch.equal(got[:,32:32+elements],source[begin:begin+n]);guards &= bool((got[:,:32]==-23131).all() and (got[:,-32:]==-23131).all())
        assert exact and guards,('native mismatch',elements,begin)
        assert torch.equal(inp.cpu(),packed[begin:begin+n]),'input changed'
        rows+=n
    # Reuse buffers/flags with poisoned outputs; include zero, one, capacity-1 and capacity repair counts.
    reuse_idx=torch.tensor([0,len(source)-REPAIRS,len(source)-REPAIRS+1,len(source)-1]);inp=packed[reuse_idx].npu();out=torch.full((4,elements+64),-23131,dtype=torch.int16,device='npu')
    for repetition in range(10):
        out.fill_(-23131);stream.synchronize()
        for i in range(4):fn(stream.npu_stream,inp.data_ptr()+i*tile_stride(elements),out.data_ptr()+(i*(elements+64)+32)*2,elements)
        stream.synchronize();got=out.cpu();assert torch.equal(got[:,32:32+elements],source[reuse_idx]);assert bool((got[:,:32]==-23131).all() and (got[:,-32:]==-23131).all())
    return dict(physical_npu=physical,tiles=rows,reuse_launches=40,all_raw_bits_exact=exact,guards=guards,input_unchanged=True,library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--library');a=p.parse_args()
    torch.set_num_threads(8);result={'scope':'independent full raw-bit coverage, no numerical BF16 comparisons','all_codes':65536,'repair_capacity':REPAIRS,'rows':[],'codec_sha256':hashlib.sha256((ROOT/'kernel_optimization/tpot7/lossless/codec.py').read_bytes()).hexdigest()}
    for elements in (20480,40960):
        started=time.monotonic();source,positions=corpus(elements);packed=pack_weight(source.view(torch.bfloat16),elements,1);unpacked=unpack_weight(packed,elements,1).view(torch.int16)
        assert torch.equal(source,unpacked),'CPU bits differ'
        expected_count=torch.cat((torch.full((61440//REPAIRS,),REPAIRS),torch.arange(REPAIRS))).to(torch.int32)
        assert torch.equal(packed[:,:32].view(torch.int32)[:,0],expected_count)
        row=dict(elements=elements,tiles=len(source),source_sha256=digest(source),encoded_sha256=digest(packed),decoded_sha256=digest(unpacked),all_raw_bits_exact=True,counts_covered=list(range(REPAIRS+1)),exception_positions=positions.tolist(),negative_cases=verify_negative(source,packed,elements))
        del unpacked
        if a.library:row['native']=verify_native(source,packed,elements,a.library)
        row['base_windows']=[]
        for base in (0,1,108,124,240):
            codes=torch.arange(65536,dtype=torch.int32);exponents=(codes>>7)&255
            regular=codes[(exponents>=base)&(exponents<base+16)]
            special=regular[torch.arange(elements)%len(regular)].to(torch.int16).reshape(1,-1)
            coded=pack_weight(special.view(torch.bfloat16),elements,1,base=base)
            decoded=unpack_weight(coded,elements,1).view(torch.int16)
            assert torch.equal(special,decoded)
            row['base_windows'].append({'base':base,'raw_bits_exact':True,'sha256':digest(decoded)})
        row['wall_seconds']=time.monotonic()-started;result['rows'].append(row);print(json.dumps(row),flush=True)
        del source,packed
    result['pass']=True;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
