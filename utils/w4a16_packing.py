"""Exact repacking of existing signed INT4-G128; never requantizes weights."""
import torch

FORMAT = "spark_w4a16_tile_g128_v1"
LUT_FORMAT = "spark_w4a16_tile_g128_lut_v2"
FLOAT_LUT_FORMAT = "spark_w4a16_tile_g128_float_lut_v5"


def unpack_checkpoint(words, scales, runtime=False):
    """Return signed integer [N,K] and original BF16 [N,K/128]."""
    if runtime:
        words, scales = words.t().contiguous(), scales.t().contiguous()
    if words.dtype != torch.int32 or scales.dtype != torch.bfloat16:
        raise ValueError("Expected checkpoint INT32 packed weights and BF16 scales")
    q = torch.stack([(words >> (4 * i)) & 15 for i in range(8)], 1).flatten(0, 1)
    q = torch.where(q >= 8, q - 16, q).to(torch.int8)
    if q.shape != (scales.shape[0], scales.shape[1] * 128):
        raise ValueError("Inconsistent INT4 and group scale dimensions")
    return q, scales


def pack_tile(q, scales, m, lut=False):
    n, k = q.shape
    if q.dtype != torch.int8 or n % m or k % 128 or scales.shape != (n, k // 128):
        raise ValueError("Tile requires aligned [N,K] INT4 and [N,K/128] scales")
    if int(q.min()) < -8 or int(q.max()) > 7 or not torch.isfinite(scales).all():
        raise ValueError("Invalid signed INT4 or scale")
    rows = q.reshape(n // m, m, k // 128, 128).permute(0, 2, 1, 3).int() & 15
    packed = (rows[..., :64] | (rows[..., 64:] << 4)).to(torch.uint8).flatten(1)
    tile_scales = scales.reshape(n // m, m, k // 128).permute(0, 2, 1).contiguous().float()
    if lut:
        codes = torch.arange(16)
        codes = torch.where(codes >= 8, codes - 16, codes).float()
        values = (tile_scales[..., None] * codes).bfloat16()
        if lut == "float32":
            values = values.float()  # Preserve the exact BF16 dequantization boundary.
        scale_bytes = values.view(torch.uint8).flatten(1)
    else:
        scale_bytes = tile_scales.view(torch.uint8).flatten(1)
    return torch.cat((packed, scale_bytes), 1).contiguous().flatten()


def unpack_tile(data, n, k, m):
    tiles = data.reshape(n // m, k * m * 17 // 32)
    codes = tiles[:, :k * m // 2].reshape(n // m, k // 128, m, 64)
    q = torch.cat((codes & 15, codes >> 4), -1).int()
    q = torch.where(q >= 8, q - 16, q).permute(0, 2, 1, 3).reshape(n, k).to(torch.int8)
    scales = tiles[:, k * m // 2:].contiguous().view(torch.float32)
    scales = scales.reshape(n // m, k // 128, m).permute(0, 2, 1).reshape(n, k // 128)
    return q, scales


def rope_order(rd):
    order = torch.arange(6144)
    pairs = torch.stack((torch.arange(rd // 2).reshape(-1, 4),
                         torch.arange(rd // 2, rd).reshape(-1, 4)), 1).flatten()
    for h in range(20):
        order[h * 256:h * 256 + rd] = h * 256 + pairs
    return order


def pack_qkv(q, scales, gate, rd, lut=False):
    order = rope_order(rd)
    data = pack_tile(q[order], scales[order], 8, lut=lut)
    return torch.cat((data, gate.contiguous().view(torch.uint8).flatten()))


def interleave_gate_up(gate, up):
    return torch.stack((gate.reshape(-1, 4, *gate.shape[1:]),
                        up.reshape(-1, 4, *up.shape[1:])), 1).reshape(-1, *gate.shape[1:])
