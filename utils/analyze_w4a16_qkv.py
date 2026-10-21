"""Compare captured native/fused projection inputs and post-RoPE Q/K/V."""
import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("directory", type=Path, help="A captured context directory")
a = p.parse_args()
torch.set_num_threads(4)
base, fused = a.directory / "logits-base", a.directory / "logits-fused"


def load(directory, layer, name):
    return torch.load(directory / f"layer-{layer:02d}-{name}.pt", weights_only=True)


def compare(actual, expected):
    x, y = actual.float().flatten(), expected.float().flatten()
    if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Invalid comparison tensors")
    return {"nrmse": float((x-y).norm()/y.norm()), "max_abs": float((x-y).abs().max()),
            "equal_fraction": float((x == y).float().mean())}


rows = []
for i in range(36):
    row = {"layer": i}
    for name in ("norm", "q", "k", "v"):
        row[name] = compare(load(fused, i, name), load(base, i, name))
    if (base / f"layer-{i:02d}-merged.pt").exists():
        expected = load(base, i, "merged")
        row["gate_reference"] = "captured native out_proj input"
    else:
        attention = load(base, i, "attention").bfloat16().reshape(16, 256)
        gate = load(base, i, "g").float().sigmoid().bfloat16().reshape(16, 1)
        expected = attention * gate
        row["gate_reference"] = "captured native attention times CPU sigmoid(BF16 g), rounded to BF16"
    row["attention_gate"] = compare(load(fused, i, "merged"), expected)
    rows.append(row)
(a.directory / "qkv-comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows[0], indent=2))
