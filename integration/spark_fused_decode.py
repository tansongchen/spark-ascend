"""Opt-in Spark single-request decode adapter for the supplied Ascend C kernel.

Loaded by the local Spark plugin only when SPARK_FUSED_DECODE=1. Prefill and
unsupported batches use the original implementation. The outer engine is eager;
SPARK_FUSED_GRAPH=1 captures the 36 fused layers into an internal NPU graph.
"""
import ctypes
import json
import math
import os
from pathlib import Path
import types

import torch
from vllm.forward_context import get_forward_context
from vllm_spark2_5_plugin.spark2_5 import Spark2_5ForCausalLM
from kernel_validation.validate import make_luts, pack_qkv, tile_weights


def fused_status(model):
    state=getattr(model.model,'_spark_fused_state',None)
    return {'enabled':state is not None,'ready':bool(state and state.ready),
            'decode_steps':state.steps if state else 0,
            'layers':len(state.entries) if state and state.ready else 0,
            'capacity':state.capacity if state else 0,
            'graph_enabled':bool(state and state.graph_enabled),
            'library':state.library_path if state and state.ready else None,
            'graph_captures':state.graph_captures if state else 0,
            'graph_replays':state.graph_replays if state else 0}


class SparkFusedWorkerExtension:
    def spark_fused_status(self):
        return fused_status(self.get_model())


class FusedState:
    def __init__(self, model):
        self.model = model
        self.original = model.forward
        self.ready = False
        self.history_ready = False
        self.steps = 0
        self.graph_enabled = os.environ.get('SPARK_FUSED_GRAPH') == '1'
        self.graph = None
        self.graph_captures = 0
        self.graph_replays = 0
        self.last_length = None
        self.capacity = int(os.environ.get('SPARK_FUSED_CAPACITY', '40960'))
        if not 1 <= self.capacity <= 40960:
            raise ValueError('Fused context capacity must be 1..40960')

    def prepare(self):
        if self.ready:
            return
        model = self.model
        if len(model.layers) != 36 or model.start_layer != 0 or model.end_layer != 36:
            raise ValueError('Fused path requires the complete 36-layer model, TP=PP=1')
        self.pages = math.ceil(self.capacity/128)
        device = model.layers[0].self_attn.q_k_v_proj.weight.device
        self.ws = torch.empty(128*1024, dtype=torch.uint8, device=device)
        self.output = [torch.empty((1,2560),dtype=torch.bfloat16,device=device) for _ in range(2)]
        self.residual = [torch.empty_like(self.output[0]) for _ in range(2)]
        if self.graph_enabled:
            self.graph_hidden = torch.empty_like(self.output[0])
            self.graph_residual = torch.empty_like(self.output[0])
            self.capture_stream = torch.npu.Stream()
        self.table = torch.arange(self.pages,dtype=torch.int32,device=device)
        self.slot = torch.zeros(1,dtype=torch.int64,device=device)
        lg,ls = make_luts()
        self.lg,self.ls=lg.to(device),ls.to(device)
        self.cos_sin={}
        for rd,theta in [(64,5000000),(256,10000)]:
            inv=1./(theta**(torch.arange(0,rd,2).float()/rd))
            angles=torch.arange(self.capacity).float()[:,None]*inv
            self.cos_sin[rd]=torch.cat((angles.cos(),angles.sin()),-1).bfloat16().to(device)
        self.entries=[]
        for i,layer in enumerate(model.layers):
            attn=layer.self_attn
            if attn.q_k_v_proj.weight.shape != (6144,2560) or attn.q_k_v_proj.weight.dtype!=torch.bfloat16:
                raise ValueError('Only unquantized BF16 TP=1 weights are supported')
            rd=64 if (i+1)%4==0 else 256
            # Pack on CPU; do not depend on a device's internal weight format.
            wq=pack_qkv(attn.q_k_v_proj.weight.cpu(),attn.g_proj.weight.cpu(),rd)
            wq=torch.cat((tile_weights(wq[:6144],8),wq[6144:])).to(device)
            gu=layer.mlp.gate_up_proj.weight.cpu()
            gate,up=gu.chunk(2,dim=0)
            w1=torch.stack((gate.reshape(-1,4,2560),up.reshape(-1,4,2560)),1).reshape(20480,2560)
            packed=[wq,tile_weights(attn.out_proj.weight.cpu(),8).to(device),
                    tile_weights(w1,8).to(device),tile_weights(layer.mlp.down_proj.weight.cpu(),4).to(device)]
            g2=model.layers[i+1].input_layernorm.weight if i<35 else model.norm.weight
            # Own request-local logical page cache: independent of vLLM page IDs.
            kc=torch.zeros((self.pages,128,4,256),dtype=torch.bfloat16,device=device)
            vc=torch.zeros_like(kc)
            nk=torch.zeros((self.pages,4,16,128,16),dtype=torch.bfloat16,device=device) if rd==64 else kc
            nv=torch.zeros_like(nk) if rd==64 else vc
            tensors=[None,None,*packed,self.lg,self.ls,layer.post_attention_layernorm.weight,g2,
                     None,self.cos_sin[rd],self.slot,self.table]
            ptrs=(ctypes.c_void_p*14)(*[0 if t is None else t.data_ptr() for t in tensors])
            self.entries.append(dict(layer=layer,rd=rd,tensors=tensors,ptrs=ptrs,caches=[kc,vc,nk,nv]))
        path=os.environ.get('SPARK_FUSED_LIBRARY','/data1/t00995472/spark-kernel-validation/build/libdecode_layer_stage_prefetch.so')
        self.library_path=path
        self.library=ctypes.CDLL(path)
        self.launch=self.library.launch_decode_layer_stage_prefetch
        self.launch.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*7+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float]
        self.launch.restype=None
        torch.npu.synchronize()
        self.ready=True
        print(f'SPARK_FUSED_READY layers=36 capacity={self.capacity} pages={self.pages} library={path}',flush=True)

    def metadata(self):
        ctx=get_forward_context()
        metadata=ctx.attn_metadata
        if not isinstance(metadata,dict) or not metadata:
            return None
        return metadata

    def transfer_history(self, metadata, length):
        count=math.ceil(length/128)
        for entry in self.entries:
            attn=entry['layer'].self_attn.core_attention
            md=metadata[attn.layer_name]
            source=attn.kv_cache
            # Current vLLM Ascend binds a tensor [2,pages,128,4,256].
            if isinstance(source,(list,tuple)):
                source=source[0] if len(source)==1 else source
            sk,sv=source[0],source[1]
            if sk.shape[1:] != (128,4,256) or sk.dtype!=torch.bfloat16:
                raise ValueError(f'Unsupported vLLM cache shape: {sk.shape}')
            ids=md.block_tables[0,:count].long()
            kc,vc,nk,nv=entry['caches']
            # SWA may reclaim old pages: only retain the pages the kernel reads.
            begin=max(0,(length-1)-512)//128 if entry['rd']==256 else 0
            kc[begin:count].copy_(sk.index_select(0,ids[begin:count]))
            vc[begin:count].copy_(sv.index_select(0,ids[begin:count]))
            # MMAD still touches padding V rows: 0 * NaN must not poison PV.
            tail=length%128
            if tail:
                kc[count-1,tail:].zero_();vc[count-1,tail:].zero_()
            if entry['rd']==64:
                nk[:count].copy_(kc[:count].reshape(count,128,4,16,16).permute(0,2,3,1,4))
                nv[:count].copy_(vc[:count].reshape(count,128,4,16,16).permute(0,2,3,1,4))
        self.history_ready=True
        print(f'SPARK_FUSED_HISTORY length={length} physical_npu={os.environ.get("ASCEND_RT_VISIBLE_DEVICES")}',flush=True)

    def launch_layers(self, stream, hidden, residual, positions):
        for i,entry in enumerate(self.entries):
            ptrs=entry['ptrs'];ptrs[0]=hidden.data_ptr();ptrs[1]=residual.data_ptr();ptrs[10]=positions.data_ptr()
            out=self.output[i%2];resout=self.residual[i%2]
            self.launch(stream,ptrs,*[c.data_ptr() for c in entry['caches']],self.ws.data_ptr(),
                        out.data_ptr(),resout.data_ptr(),entry['rd'],0,1e-6,1e-6)
            hidden,residual=out,resout
        return hidden

    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
        self.prepare()
        md=self.metadata()
        first=next(iter(md.values())) if md else None
        decode=(first is not None and getattr(first.attn_state,'name','')=='DecodeOnly'
                and first.num_decodes==1 and first.num_prefills==0
                and positions.numel()==1 and len(first.seq_lens_list)==1)
        if not decode:
            if self.history_ready and first is not None and getattr(first.attn_state,'name','')=='DecodeOnly':
                raise RuntimeError('Cannot fall back after fused decode without restoring its cache')
            self.history_ready=False
            self.last_length=None
            return self.original(input_ids,positions,intermediate_tensors,inputs_embeds)
        length=first.seq_lens_list[0]
        if length>self.capacity:
            raise ValueError('Fused request exceeds prepared capacity; refuse stale-cache fallback')
        if not self.history_ready:
            self.transfer_history(md,length)
            self.request_steps=0
            self.request_first_length=length
        elif length!=self.last_length+1:
            raise RuntimeError(f'Nonsequential decode {self.last_length} -> {length}')
        hidden=inputs_embeds if inputs_embeds is not None else self.model.embed_input_ids(input_ids)
        residual=hidden
        hidden=self.model.layers[0].input_layernorm(hidden)
        # Logical slots match absolute positions; no device-to-host read needed.
        self.slot.copy_(positions.reshape(1))
        if self.graph_enabled:
            self.graph_hidden.copy_(hidden)
            self.graph_residual.copy_(residual)
        # Torch's asynchronous task queue must submit copies before raw ACL calls.
        current=torch.npu.current_stream()
        current.synchronize()
        stream=current.npu_stream
        if self.graph_enabled:
            if self.graph is None:
                # Initialize ACL kernel resources before capture. Repeating the same
                # token overwrites the same KV slot with identical values.
                self.launch_layers(stream,self.graph_hidden,self.graph_residual,self.slot)
                current.synchronize()
                self.graph = torch.npu.NPUGraph()
                with torch.npu.graph(self.graph,stream=self.capture_stream):
                    self.launch_layers(self.capture_stream.npu_stream,self.graph_hidden,self.graph_residual,self.slot)
                self.graph_captures += 1
            self.graph.replay()
            self.graph_replays += 1
            hidden = self.output[35%2]
        else:
            hidden=self.launch_layers(stream,hidden,residual,positions)
        self.last_length=length
        self.steps+=1
        self.request_steps+=1
        if self.steps in (1,16,4096):
            print(f'SPARK_FUSED_STEPS={self.steps} length={length} launches_per_step=36',flush=True)
        if self.request_steps==self.capacity-self.request_first_length:
            print(f'SPARK_FUSED_REQUEST_COMPLETE decode_steps={self.request_steps} last_length={length}',flush=True)
        return hidden


class SparkFusedForCausalLM(Spark2_5ForCausalLM):
    def __init__(self,*,vllm_config,prefix=''):
        super().__init__(vllm_config=vllm_config,prefix=prefix)
        enabled=os.environ.get('SPARK_FUSED_DECODE')=='1'
        if enabled and (vllm_config.scheduler_config.max_num_seqs!=1
                        or not vllm_config.model_config.enforce_eager
                        or vllm_config.model_config.max_model_len>40960
                        or vllm_config.cache_config.enable_prefix_caching
                        or vllm_config.speculative_config is not None):
            raise ValueError('Fused adapter requires eager, max_num_seqs=1, max_model_len<=40960, no prefix cache/speculation')
        config=vllm_config.model_config.hf_config
        if enabled and (config.sliding_window!=512 or config.rms_norm_eps!=1e-6):
            raise ValueError('Kernel is specialized for sliding_window=512, rms_norm_eps=1e-6')
        state=FusedState(self.model) if enabled else None
        if state is not None:
            self.model._spark_fused_state=state
        original=self.model.forward
        self._validation_active=False
        self._validation_step=0
        directory=os.environ.get('SPARK_LOGITS_DIR')
        self._logit_dir=Path(directory) if directory else None
        if self._logit_dir:
            self._logit_dir.mkdir(parents=True,exist_ok=True)
        forced=os.environ.get('SPARK_FORCE_TOKENS')
        self._forced=json.loads(Path(forced).read_text()) if forced else None
        def forward(model,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
            if self._logit_dir or self._forced:
                metadata=get_forward_context().attn_metadata
                first=next(iter(metadata.values())) if isinstance(metadata,dict) and metadata else None
                self._validation_active=first is not None
                if first is not None:
                    if getattr(first.attn_state,'name','')=='DecodeOnly':
                        self._validation_step+=1
                    else:
                        self._validation_step=0
            fn=state.forward if state is not None else original
            return fn(input_ids,positions,intermediate_tensors,inputs_embeds)
        self.model.forward=types.MethodType(forward,self.model)

    def compute_logits(self,hidden_states):
        logits=super().compute_logits(hidden_states)
        step=self._validation_step
        if self._validation_active and logits is not None:
            if self._logit_dir and step<128:
                torch.save(logits.detach().float().cpu(),self._logit_dir/f'step-{step:04d}.pt')
            if self._forced is not None and step<len(self._forced):
                logits.fill_(-float('inf'))
                logits[...,self._forced[step]]=0
        return logits
