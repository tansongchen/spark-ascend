"""Lossless layout conversion of upstream compressed-tensors INT4, no quantizer."""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32


class Checkpoint:
    """Read either the upstream single safetensors file or a sharded checkpoint."""
    def __init__(self, model):
        self.path = Path(model)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.index = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.path / "model.safetensors", framework="pt") as f:
                self.index = {key: "model.safetensors" for key in f.keys()}

    def read(self, name):
        with safe_open(self.path / self.index[name], framework="pt") as f:
            return f.get_tensor(name)


def int4_scales(scales):
    """Upstream down_proj stores BF16-exact scales in FP32 containers."""
    if scales.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Expected BF16 or BF16-exact FP32 INT4 scales")
    bf16 = scales.bfloat16()
    if not torch.equal(scales.float(), bf16.float()):
        raise ValueError("INT4 scale is not exactly representable in BF16; refusing precision loss")
    if not torch.isfinite(bf16).all() or not (bf16 > 0).all():
        raise ValueError("INT4 scales must be finite and positive")
    return bf16


def unpack_int4(words, scales, zero_points):
    scales = int4_scales(scales)
    n, groups = scales.shape
    k = groups * 32
    if (words.dtype != torch.int32 or words.shape != (n, k // 8)
            or scales.dtype != torch.bfloat16
            or zero_points.dtype != torch.int32 or zero_points.shape != (n // 8, groups)):
        raise ValueError("Expected upstream INT4 [N,K/8], BF16 G32 scales and packed [N/8,K/32] zero points")
    q = unpack_from_int32(words, 4, torch.Size((n, k)))
    zp = unpack_from_int32(zero_points, 4, scales.shape, packed_dim=0)
    return q, zp


def repack_int4(words, scales, zero_points):
    """Offset-binary K packing -> signed N packing; retain every code/scale/zp."""
    scales = int4_scales(scales)
    q, zp = unpack_int4(words, scales, zero_points)
    k, n = q.shape[1], q.shape[0]
    codes = (q.t().contiguous().reshape(k, n // 8, 8).int() & 15)
    packed = torch.zeros((k, n // 8), dtype=torch.int32, device=q.device)
    for i in range(8):
        packed |= codes[..., i] << (4 * i)
    # Ascend antiquantization is (signed_code + offset) * scale.
    return packed, scales.t().contiguous(), (-zp.to(torch.bfloat16)).t().contiguous()
