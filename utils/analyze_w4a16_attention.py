"""Attention-only references using captured real Q/K/V; no model loading."""
import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("directory", type=Path)
p.add_argument("--layer", type=int, required=True)
a = p.parse_args()
torch.set_num_threads(8)
base, fused = a.directory/"logits-base", a.directory/"logits-fused"
tag = f"layer-{a.layer:02d}"
c = torch.load(base/f"{tag}-cache.pt", weights_only=True)
fc = torch.load(fused/f"{tag}-cache.pt", weights_only=True)
q = torch.load(base/f"{tag}-q.pt", weights_only=True).reshape(4,4,256).double()
expected = torch.load(base/f"{tag}-attention.pt", weights_only=True).flatten().float()
length, first = c["length"], c["first_page"]*128
begin = max(0,length-1-512) if (a.layer+1)%4 else 0
lo, hi = begin-first, length-first
k = c["key"].flatten(0,1)[lo:hi].permute(1,0,2).double()
v = c["value"].flatten(0,1)[lo:hi].permute(1,0,2).double()
scores = torch.bmm(q,k.transpose(1,2))*0.0625
scores -= scores.max(-1,keepdim=True).values


def rnd(x, away):
    f = x.float().contiguous()
    if not away:
        return f.bfloat16().float().flatten()
    assert torch.isfinite(f).all()
    bits = f.view(torch.int32).long() & 0xffffffff
    return ((bits+0x8000)>>16).short().view(torch.bfloat16).float().flatten()


def metrics(x):
    x = x.float().flatten()
    return {"nrmse": float((x-expected).norm()/expected.norm()),
            "max_abs": float((x-expected).abs().max()),
            "equal_fraction": float((x == expected).float().mean())}


variants = {}
for dtype in (torch.float32, torch.float64):
    s = scores.to(dtype)
    vv = v.to(dtype)
    e = s.exp()
    total = e.sum(-1, keepdim=True)
    for name, prob, denominator in (
        ("full_precision_p", e, total),
        ("bf16_unnormalized_p", e.bfloat16().to(dtype), total),
        ("bf16_normalized_p", (e/total).bfloat16().to(dtype), torch.ones_like(total)),
    ):
        out = torch.bmm(prob, vv)/denominator
        for away in (False, True):
            variants[f"{dtype}.{name}.{'round' if away else 'rint'}"] = metrics(rnd(out, away))
    # Online softmax: this changes the BF16 P rounding point at each tile.
    for width in (128, 256, 384, 512, 640, 1024, 2048):
        m = torch.full((*s.shape[:-1],1), -1e30, dtype=dtype)
        total = torch.zeros_like(m)
        acc = torch.zeros((*s.shape[:-1],256), dtype=dtype)
        for start in range(0,s.shape[-1],width):
            chunk = s[...,start:start+width]
            next_m = torch.maximum(m,chunk.max(-1,keepdim=True).values)
            alpha = (m-next_m).exp()
            prob = (chunk-next_m).exp()
            total = total*alpha+prob.sum(-1,keepdim=True)
            acc = acc*alpha+torch.bmm(prob.bfloat16().to(dtype),vv[:,start:start+width])
            m = next_m
        for away in (False,True):
            variants[f"{dtype}.online_{width}.{'round' if away else 'rint'}"] = metrics(rnd(acc/total,away))

# Match the fused page boundaries and compare alternative split widths. This
# isolates BF16 probability rounding from errors accumulated in earlier layers.
s = torch.bmm(q, c["key"].flatten(0,1)[:hi].permute(1,0,2).double().transpose(1,2)).float()*0.0625
vv = c["value"].flatten(0,1)[:hi].permute(1,0,2).float()
s[...,:lo] = -1e30
part_widths = (128, 256, 640) if begin else (5120, ((length+895)//896)*128, length)
for part_width in part_widths:
    for width in (128, 256, 384, 640):
        outputs, maxima, sums = [], [], []
        for part in range(0,s.shape[-1],part_width):
            m = torch.full((*s.shape[:-1],1), -1e30)
            total = torch.zeros_like(m)
            acc = torch.zeros((*s.shape[:-1],256))
            for start in range(part,min(part+part_width,s.shape[-1]),width):
                end = min(start+width,part+part_width,s.shape[-1])
                chunk = s[...,start:end]
                next_m = torch.maximum(m,chunk.max(-1,keepdim=True).values)
                alpha = (m-next_m).exp()
                prob = (chunk-next_m).exp()
                total = total*alpha+prob.sum(-1,keepdim=True)
                acc = acc*alpha+torch.bmm(prob.bfloat16().float(),vv[:,start:end])
                m = next_m
            outputs.append(acc)
            maxima.append(m)
            sums.append(total)
        m = torch.stack(maxima).max(0).values
        scale = [(x-m).exp() for x in maxima]
        total = sum(x*y for x,y in zip(sums,scale))
        acc = sum(x*y for x,y in zip(outputs,scale))
        for away in (False,True):
            variants[f"page_aligned.split_{part_width}.tile_{width}.{'round' if away else 'rint'}"] = metrics(rnd(acc/total,away))

cache_diff = {}
for name in ("key","value"):
    x,y = c[name].flatten(0,1)[lo:hi],fc[name].flatten(0,1)[lo:hi]
    cache_diff[name] = {"different_values":int((x != y).sum()),"values":x.numel(),
                        "max_abs":float((x.float()-y.float()).abs().max())}
report = {"layer":a.layer, "length":length, "window_begin":begin,
          "cache_difference":cache_diff, "variants":variants,
          "reference":"Native attention output, before gate. Oracle scores use FP64 dot products of captured BF16 inputs."}
(a.directory/f"{tag}-attention-references.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"cache_difference":cache_diff,
                  "best":sorted(variants.items(),key=lambda x:x[1]["nrmse"])[:10]},indent=2))
