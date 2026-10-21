#!/usr/bin/env python3
"""Export the local Spark BF16 checkpoint as symmetric per-group INT4/BF16."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "spark_int4_signed_output_pack_v1"
SUFFIXES = (
    ".self_attn.q_k_v_proj.weight", ".self_attn.out_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def quantize(weight, group_size):
    """Round-to-nearest absmax/7, with scales rounded before assigning codes."""
    n, k = weight.shape
    if n % 8 or k % group_size or weight.dtype != torch.bfloat16:
        raise ValueError(f"Unsupported matrix: {weight.shape}, {weight.dtype}")
    packed = torch.empty(n // 8, k, dtype=torch.int32)
    scales = torch.empty(n, k // group_size, dtype=torch.bfloat16)
    squared_error = squared_weight = 0.0
    for first in range(0, n, 256):
        w = weight[first:first + 256].float().reshape(-1, k // group_size, group_size)
        if not torch.isfinite(w).all():
            raise ValueError("Non-finite source weight")
        s = (w.abs().amax(-1) / 7).clamp_min(1e-12).bfloat16()
        q = (w / s.float().unsqueeze(-1)).round().clamp(-7, 7).to(torch.int32).reshape(-1, k)
        reconstructed = (q.reshape_as(w).float() * s.float().unsqueeze(-1)).bfloat16().float()
        squared_error += (w - reconstructed).square().sum().item()
        squared_weight += w.square().sum().item()
        scales[first:first + w.shape[0]] = s
        codes = q.reshape(-1, 8, k) & 15
        words = torch.zeros(codes.shape[0], k, dtype=torch.int32)
        for bit in range(8):
            words |= codes[:, bit, :] << (4 * bit)
        # Verify signed nibble packing for every exported element.
        for bit in range(8):
            restored = (words >> (4 * bit)) & 15
            restored = torch.where(restored >= 8, restored - 16, restored)
            if not torch.equal(restored, q[bit::8]):
                raise AssertionError("INT4 pack/unpack mismatch")
        packed[first // 8:(first + w.shape[0]) // 8] = words
    return packed, scales, (squared_error / squared_weight) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/data1/t00995472/models/Spark-X2.5-4B"))
    parser.add_argument("--output", type=Path, default=Path("/data1/t00995472/models/Spark-X2.5-4B-W4A16-G128"))
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be separate from the BF16 source")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; refusing to overwrite")
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "spark2_5" or not config["tie_word_embeddings"]:
        raise ValueError("Only the tied-embedding Spark model is supported")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    description = {
        "quant_method": "ascend", "spark_format": FORMAT,
        "group_size": args.group_size, "weight_bits": 4,
        "activation_dtype": "bfloat16",
    }
    report = {
        "source": str(source), "output": str(output), "format": FORMAT,
        "algorithm": "symmetric absmax RTN; BF16 scales; no calibration",
        "group_size": args.group_size, "source_sha256": {},
        "quantized": [], "retained_bf16": [], "source_parameter_bytes": index["metadata"]["total_size"],
    }
    weight_map, total_size = {}, 0
    shards = sorted(set(index["weight_map"].values()))
    for number, original in enumerate(shards, 1):
        report["source_sha256"][original] = digest(source / original)
        tensors = {}
        with safe_open(source / original, framework="pt", device="cpu") as f:
            for name in f.keys():
                weight = f.get_tensor(name)
                if name.endswith(SUFFIXES):
                    packed, scales, rel_l2 = quantize(weight, args.group_size)
                    tensors[name] = packed
                    tensors[name.removesuffix(".weight") + ".weight_scale"] = scales
                    description[name] = "SPARK_W4A16"
                    report["quantized"].append({
                        "name": name, "shape": list(weight.shape),
                        "packed_shape": list(packed.shape), "relative_weight_l2": rel_l2,
                    })
                else:
                    tensors[name] = weight
                    description[name] = "FLOAT"
                    report["retained_bf16"].append(name)
        filename = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, output / filename, metadata={"format": "pt"})
        for name, tensor in tensors.items():
            weight_map[name] = filename
            total_size += tensor.numel() * tensor.element_size()
        print(f"{number}/{len(shards)}: {filename}; {len(report['quantized'])} matrices quantized", flush=True)
        del tensors
    expected = config["num_hidden_layers"] * len(SUFFIXES)
    if len(report["quantized"]) != expected:
        raise ValueError(f"Expected {expected} matrices, got {len(report['quantized'])}")
    for file in source.iterdir():
        if file.is_file() and file.suffix in (".json", ".py", ".jinja", ".txt"):
            if file.name not in ("config.json", "model.safetensors.index.json"):
                shutil.copy2(file, output / file.name)
    if (source / "LICENSE").exists():
        shutil.copy2(source / "LICENSE", output / "LICENSE")
    config["quantization_config"] = description
    for name, data in {
        "config.json": config,
        "quant_model_description.json": description,
        "model.safetensors.index.json": {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    }.items():
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    report["output_parameter_bytes"] = total_size
    report["output_sha256"] = {p.name: digest(p) for p in sorted(output.glob("*.safetensors"))}
    report["elapsed_seconds"] = time.time() - start
    (output / "quantization_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (output / "README.md").write_text(
        "# Spark-X2.5-4B W4A16\n\n"
        "Local symmetric RTN INT4 weights, BF16 activations/scales; requires the "
        "spark-w4a16-ascend plugin and SPARK_W4A16=1. Not an AWQ/GPTQ export.\n\n"
        "Embedding/tied output head, normalization and attention gate retain BF16. "
        "See quantization_report.json for provenance and tensor coverage. "
        "No benchmark-quality accuracy or speed claim is made by this conversion.\n"
    )
    print(json.dumps({k: report[k] for k in ("output", "source_parameter_bytes", "output_parameter_bytes", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
