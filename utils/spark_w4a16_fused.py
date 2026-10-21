"""W4A16-G128 fusion reuses the validated BF16 cache and layer control flow."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
from utils.spark_fused_decode import FusedState
from utils.w4a16_packing import (
    FORMAT, LUT_FORMAT, FLOAT_LUT_FORMAT, OUTER_FORMAT, unpack_checkpoint, pack_tile, pack_qkv, interleave_gate_up, pack_outer,
)


class W4A16FusedState(FusedState):
    # Explicit selection avoids silently choosing an experimental binary.
    default_library = ""

    def prepare(self):
        was_ready = self.ready
        if not was_ready:
            path = Path(os.environ.get("SPARK_FUSED_LIBRARY", ""))
            manifest = json.loads((path.parent / "execution.json").read_text())
            self.native_rope_cache = manifest.get("native_rope_cache", False)
            self.outer = manifest.get('weight_encoding') == OUTER_FORMAT
            self.quantized_head = manifest.get('lm_head_w4a16', False)
            self.kv8 = manifest.get('kv_encoding') == 'fp8_e4m3fn_full_attention'
            self.lut = ("float32" if manifest.get("weight_encoding") == FLOAT_LUT_FORMAT
                        else manifest.get("weight_encoding") == LUT_FORMAT)
            self.workspace_bytes = manifest["workspace_bytes"]
            if self.workspace_bytes not in (131072, 262144, 524288):
                raise ValueError("Unsupported W4A16 workspace layout")
        super().prepare()
        if not was_ready and self.quantized_head:
            self.prepare_head()
        if not was_ready and self.kv8:
            self.kv_codec = self.library.launch_reborn_kv8_codec
            self.kv_codec.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2
            self.kv_codec.restype=None
            for entry in self.entries:
                if entry['rd'] == 64:
                    entry['bf16_nz']=entry['caches'][2:]
                    entry['fp8_nz']=[torch.empty_like(t,dtype=torch.uint8) for t in entry['bf16_nz']]
                    entry['caches'][2:]=entry['fp8_nz']
        if not was_ready and self.native_rope_cache:
            self.rope_cache_audit = {}
            for entry in self.entries:
                cache = entry["layer"].self_attn.rotary_emb.cos_sin_cache
                rd = entry["rd"]
                if (cache.dtype != torch.bfloat16 or cache.shape[1] != rd
                        or cache.shape[0] < self.capacity or not cache.is_contiguous()):
                    raise ValueError("Native RoPE cache is incompatible with fused ABI")
                if rd not in self.rope_cache_audit:
                    old = entry["tensors"][11][:self.capacity].cpu()
                    new = cache[:self.capacity].cpu()
                    self.rope_cache_audit[rd] = {
                        "different_values": int((old != new).sum()),
                        "values": old.numel(),
                        "max_abs": float((old.float() - new.float()).abs().max()),
                    }
                entry["tensors"][11] = cache
                entry["ptrs"][11] = cache.data_ptr()
                self.cos_sin[rd] = cache
            torch.npu.synchronize()
            print("SPARK_W4A16_NATIVE_ROPE " + json.dumps(self.rope_cache_audit), flush=True)

    def transfer_history(self,metadata,length):
        if not self.kv8:
            return super().transfer_history(metadata,length)
        for entry in self.entries:
            if entry['rd']==64:
                entry['caches'][2:]=entry['bf16_nz']
        try:
            super().transfer_history(metadata,length)
            torch.npu.synchronize()
            count=((length+127)//128)*131072
            stream=torch.npu.current_stream().npu_stream
            for entry in self.entries:
                if entry['rd']==64:
                    for src,dst in zip(entry['bf16_nz'],entry['fp8_nz']):
                        self.kv_codec(stream,src.data_ptr(),dst.data_ptr(),count,0)
            torch.npu.synchronize()
        finally:
            for entry in self.entries:
                if entry['rd']==64:
                    entry['caches'][2:]=entry['fp8_nz']

    def prepare_head(self):
        from utils.quantize_spark_w4a16 import quantize
        weight = self.model.embedding.weight
        if weight.shape != (131072,2560) or weight.dtype != torch.bfloat16:
            raise ValueError('Reborn head requires the tied BF16 131072x2560 embedding')
        words,scales,relative_l2 = quantize(weight.cpu(),128)
        q,s = unpack_checkpoint(words,scales)
        packed = pack_outer(q,s)
        self.head_weight = packed.to(weight.device)
        self.head_partial = torch.empty((131072*2,),dtype=torch.float32,device=weight.device)
        self.head_output = torch.empty((1,131072),dtype=torch.bfloat16,device=weight.device)
        self.head_launch = self.library.launch_w4a16_reborn_head
        self.head_launch.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int]
        self.head_launch.restype = None
        self.head_calls = 0
        self.head_audit = {'enabled': True,'algorithm':'symmetric absmax RTN G128',
            'packed_bytes':packed.numel(),'activation_dtype':'bfloat16',
            'source_weight_dtype':'bfloat16','relative_weight_l2':relative_l2,
            'packed_sha256':hashlib.sha256(packed.numpy().tobytes()).hexdigest(),
            'scope':'decode LM head only; embedding and prefill head retain BF16'}
        print('SPARK_REBORN_HEAD '+json.dumps(self.head_audit),flush=True)

    def head_logits(self,hidden):
        stream = torch.npu.current_stream()
        if self.graph_enabled:
            stream.synchronize()
        # Raw layer launches precede this entry on this stream. The output tensor
        # is kept alive with the state until asynchronous sampling has completed.
        self.head_launch(stream.npu_stream,hidden.data_ptr(),self.head_weight.data_ptr(),
                         self.head_partial.data_ptr(),self.head_output.data_ptr(),131072)
        self.head_calls += 1
        return self.head_output

    def pack_layer(self, layer, rd):
        attn = layer.self_attn
        device = attn.q_k_v_proj.weight.device
        def read(mod):
            scheme = getattr(mod.quant_method, "quant_method", None)
            if getattr(scheme, "group_size", None) != 128:
                raise ValueError("Fused W4A16 requires group_size=128")
            return unpack_checkpoint(mod.weight.cpu(), mod.weight_scale.cpu(), runtime=True)
        q, s = read(attn.q_k_v_proj)
        if q.shape != (6144, 2560):
            raise ValueError("Unexpected W4A16 QKV dimensions")
        if self.outer:
            def outer(q,s): return pack_outer(q,s).to(device)
            wq = torch.cat((pack_outer(q,s),attn.g_proj.weight.cpu().contiguous().view(torch.uint8).flatten())).to(device)
            wo = outer(*read(attn.out_proj))
            w1 = outer(*read(layer.mlp.gate_up_proj))
            w2 = outer(*read(layer.mlp.down_proj))
            return [wq,wo,w1,w2]
        wq = pack_qkv(q, s, attn.g_proj.weight.cpu(), rd, lut=self.lut).to(device)
        q, s = read(attn.out_proj)
        wo = pack_tile(q, s, 8, lut=self.lut).to(device)
        q, s = read(layer.mlp.gate_up_proj)
        gate, up = q.chunk(2, 0)
        gs, us = s.chunk(2, 0)
        w1 = pack_tile(interleave_gate_up(gate, up), interleave_gate_up(gs, us), 8, lut=self.lut).to(device)
        q, s = read(layer.mlp.down_proj)
        w2 = pack_tile(q, s, 4, lut=self.lut).to(device)
        return [wq, wo, w1, w2]

    def load_library(self, path):
        if not path:
            raise ValueError("W4A16 fusion requires an explicit SPARK_FUSED_LIBRARY")
        path = Path(path).resolve(strict=True)
        manifest = json.loads((path.parent / "execution.json").read_text())
        encoding = OUTER_FORMAT if self.outer else FLOAT_LUT_FORMAT if self.lut == "float32" else LUT_FORMAT if self.lut else FORMAT
        if manifest.get("weight_encoding") != encoding or manifest.get("workspace_bytes") != self.workspace_bytes:
            raise ValueError("Incompatible W4A16 fused library manifest")
        if (bool(manifest.get('lm_head_w4a16',False)) != self.quantized_head
                or (manifest.get('kv_encoding') == 'fp8_e4m3fn_full_attention') != self.kv8):
            raise ValueError('Library head/KV encoding does not match the prepared state')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in getattr(self, "_library_hashes", {}) and self._library_hashes[str(path)] != digest:
            raise RuntimeError("A loaded library was overwritten")
        library = ctypes.CDLL(str(path))
        launch = library.launch_decode_layer_w4a16
        launch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_void_p]*7 + [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
        launch.restype = None
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Library changed while loading")
        if not hasattr(self, "_libraries"):
            self._libraries, self._library_hashes = {}, {}
        self._libraries[str(path)], self._library_hashes[str(path)] = library, digest
        self.library_path, self.library, self.launch = str(path), library, launch
        self.library_sha256, self.weight_encoding = digest, encoding
        if hasattr(self,'head_weight'):
            self.head_launch=library.launch_w4a16_reborn_head
            self.head_launch.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]
            self.head_launch.restype=None
        if hasattr(self,'kv_codec'):
            self.kv_codec=library.launch_reborn_kv8_codec
            self.kv_codec.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2
            self.kv_codec.restype=None
