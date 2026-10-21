"""W4A16-G128 fusion reuses the validated BF16 cache and layer control flow."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
from utils.spark_fused_decode import FusedState
from utils.w4a16_packing import (
    FORMAT, LUT_FORMAT, FLOAT_LUT_FORMAT, unpack_checkpoint, pack_tile, pack_qkv, interleave_gate_up,
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
            self.lut = ("float32" if manifest.get("weight_encoding") == FLOAT_LUT_FORMAT
                        else manifest.get("weight_encoding") == LUT_FORMAT)
            self.workspace_bytes = manifest["workspace_bytes"]
            if self.workspace_bytes not in (131072, 262144):
                raise ValueError("Unsupported W4A16 workspace layout")
        super().prepare()
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
        encoding = FLOAT_LUT_FORMAT if self.lut == "float32" else LUT_FORMAT if self.lut else FORMAT
        if manifest.get("weight_encoding") != encoding or manifest.get("workspace_bytes") != self.workspace_bytes:
            raise ValueError("Incompatible W4A16 fused library manifest")
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
