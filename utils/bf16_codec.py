"""Lossless BF16 storage encoding; reconstructed values retain all 16 bits.

The input has already been transformed by tile_weights. Each M*K-element tile:
  [32-byte header][E sign/mantissa bytes][E/2 exponent bytes][256 repair pairs]
Header uint32: [repair_count, exponent_base, magic, E, 0, 0, 0, 0].
An exponent byte contains the even index in its low nibble, odd in its high.
Each repair pair is little-endian uint16(index), uint16(original BF16 bits).
No value is rounded or discarded. More than 256 repairs refuses encoding.
"""
import torch

MAGIC = 0x42463132
REPAIRS = 256
HEADER_BYTES = 32


def tile_stride(elements):
    if elements % 64 or elements > 65536:
        raise ValueError('Tile must have a multiple of 64 elements and fit uint16 indices')
    return HEADER_BYTES + elements * 3 // 2 + REPAIRS * 4


def pack_weight(tiled_weight, K, M, base=109):
    if tiled_weight.device.type != 'cpu' or tiled_weight.dtype != torch.bfloat16:
        raise ValueError('Pass a CPU BF16 tensor after tile_weights')
    if not 0 <= base <= 240:
        raise ValueError('Invalid 16-exponent window')
    elements = K * M
    stride = tile_stride(elements)
    if tiled_weight.numel() % elements:
        raise ValueError('Incomplete tile')
    source = tiled_weight.contiguous().view(torch.int16).reshape(-1, elements)
    out = torch.zeros((source.shape[0], stride), dtype=torch.uint8)
    max_repairs, total_repairs = 0, 0
    for begin in range(0, source.shape[0], 64):
        bits = source[begin:begin+64].to(torch.int32) & 65535
        exp = (bits >> 7) & 255
        abnormal = (exp < base) | (exp >= base+16)
        counts = abnormal.sum(1)
        peak = int(counts.max())
        if peak > REPAIRS:
            raise ValueError(f'Tile requires {peak} repairs, capacity {REPAIRS}; no lossy fallback')
        max_repairs = max(max_repairs, peak)
        total_repairs += int(counts.sum())
        dst = out[begin:begin+bits.shape[0]]
        header = dst[:, :HEADER_BYTES].view(torch.int32)
        header[:, 0] = counts.to(torch.int32)
        header[:, 1] = base
        header[:, 2] = MAGIC
        header[:, 3] = elements
        dst[:, HEADER_BYTES:HEADER_BYTES+elements] = ((bits & 127) | ((bits >> 8) & 128)).to(torch.uint8)
        nibble = (exp-base).clamp(0, 15)
        dst[:, HEADER_BYTES+elements:HEADER_BYTES+elements*3//2] = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).to(torch.uint8)
        coordinates = abnormal.nonzero()
        if coordinates.numel():
            row, col = coordinates[:, 0], coordinates[:, 1]
            rank = torch.arange(len(row)) - torch.repeat_interleave(counts.cumsum(0)-counts, counts)
            repairs = dst[:, HEADER_BYTES+elements*3//2:].view(-1, REPAIRS, 4)
            raw = bits[row, col]
            repairs[row, rank, 0] = (col & 255).to(torch.uint8)
            repairs[row, rank, 1] = (col >> 8).to(torch.uint8)
            repairs[row, rank, 2] = (raw & 255).to(torch.uint8)
            repairs[row, rank, 3] = (raw >> 8).to(torch.uint8)
    return out


def unpack_weight(packed, K, M):
    if packed.device.type != 'cpu' or packed.dtype != torch.uint8:
        raise ValueError('Pass CPU encoded bytes')
    elements = K*M
    stride = tile_stride(elements)
    src = packed.contiguous().reshape(-1, stride)
    result = torch.empty((len(src), elements), dtype=torch.int16)
    for begin in range(0, len(src), 64):
        data = src[begin:begin+64]
        header = data[:, :HEADER_BYTES].view(torch.int32)
        if not bool((header[:, 2] == MAGIC).all()) or not bool((header[:, 3] == elements).all()):
            raise ValueError('Invalid header')
        count = header[:, 0].to(torch.int64)
        base = header[:, 1].to(torch.int32)
        if not bool(((count >= 0) & (count <= REPAIRS)).all()) or not bool(((base >= 0) & (base <= 240)).all()):
            raise ValueError('Invalid header range')
        low = data[:, HEADER_BYTES:HEADER_BYTES+elements].to(torch.int32)
        nibble = data[:, HEADER_BYTES+elements:HEADER_BYTES+elements*3//2].to(torch.int32)
        exp = torch.stack((nibble & 15, nibble >> 4), dim=-1).reshape(-1, elements) + base[:, None]
        bits = (low & 127) | ((low & 128) << 8) | (exp << 7)
        pairs = data[:, HEADER_BYTES+elements*3//2:].view(-1, REPAIRS, 4).to(torch.int32)
        row, entry = (torch.arange(REPAIRS)[None, :] < count[:, None]).nonzero(as_tuple=True)
        if row.numel():
            positions = pairs[row, entry, 0] | (pairs[row, entry, 1] << 8)
            values = pairs[row, entry, 2] | (pairs[row, entry, 3] << 8)
            if not bool((positions < elements).all()):
                raise ValueError('Repair index out of range')
            bits[row, positions] = values
        result[begin:begin+len(data)] = bits.to(torch.int16)
    return result.view(torch.bfloat16)
