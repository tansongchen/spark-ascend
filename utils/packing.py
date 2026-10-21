"""Shared BF16 weight layouts and activation lookup tables."""
import torch
import torch.nn.functional as F

BF = torch.bfloat16

def bf(x):
    return x.to(BF)


def pack_qkv(w, gate, rd):
    # NeoX [first half, second half] -> groups [first 4, paired second 4].
    packed = w.clone()
    for head in range(20):
        base = head * 256
        order = torch.stack((torch.arange(rd//2).reshape(-1, 4),
                             torch.arange(rd//2, rd).reshape(-1, 4)), dim=1).flatten()
        packed[base:base+rd] = w[base + order]
    return torch.cat((packed, gate)).contiguous()


def tile_weights(w, m):
    n,k=w.shape
    return w.reshape(n//m,m,k//64,64).permute(0,2,1,3).contiguous().reshape(n,k)


def make_luts():
    bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    vals = bits.view(BF).float()
    sigmoid = bf(torch.sigmoid(vals))
    pos = torch.arange(0x3b00, 0x4300, dtype=torch.int32)
    codes = torch.cat((pos, pos + 0x8000)).to(torch.uint16)
    gelu = bf(F.gelu(codes.view(BF).float(), approximate='none'))
    return gelu, sigmoid


