#!/usr/bin/env python3
"""Reconcile paired 16-step BF16/W4A16 decode captures; no device execution."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.analyze_spark_profiles import union_length

MATMUL = "aclnnMatmul_MatMulV3Common_MatMulV3"
QUANT = "aclnnWeightQuantBatchMatmulV2_WeightQuantBatchMatmulV2_WeightQuantBatchMatmulV2"
SHAPES = {
    (2560, 6144): "qkv", (4096, 2560): "attention_output",
    (2560, 20480): "mlp_gate_up", (10240, 2560): "mlp_down",
    (2560, 16): "attention_gate", (2560, 131072): "lm_head",
}


def detail(rows):
    def total(field):
        try:
            values = [float(row[field]) for row in rows]
        except (KeyError, ValueError, TypeError):
            return None
        return sum(values) if all(math.isfinite(x) for x in values) else None
    duration = total("Duration(us)")
    assert duration is not None and rows
    ratios = {}
    for base, fields in (
        ("aicore_time(us)", ("aic_mac_time(us)", "aic_mte2_time(us)")),
        ("aiv_time(us)", ("aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)")),
    ):
        denominator = total(base)
        for field in fields:
            numerator = total(field)
            ratios[field + "/" + base] = (
                numerator / denominator
                if numerator is not None and denominator is not None and denominator > 0 else None
            )
    return {"count": len(rows), "device_ms_per_step": duration / 16000, "pipe_time_ratios": ratios}


def analyze(directory, mode):
    paths = list((directory / f"profile-{mode}").rglob("kernel_details.csv"))
    assert len(paths) == 1, paths
    path = paths[0]
    with path.open() as f:
        rows = list(csv.DictReader(f))
    counts = Counter(row["Name"] for row in rows)
    expected = {
        "aclnnEmbedding_GatherV2AiCore_GatherV2": 16,
        "FusedInferAttentionScore": 576, "_triton_rope": 576,
        "aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2": 16,
        MATMUL: 592 if mode == "w4a16" else 2896,
        QUANT: 2304 if mode == "w4a16" else 0,
    }
    assert all(counts[name] == value for name, value in expected.items()), (mode, expected, counts)
    assert len(rows) == 9664
    origin = min(Decimal(row["Start Time(us)"]) for row in rows)
    intervals, groups = [], defaultdict(list)
    for row in rows:
        start = float(Decimal(row["Start Time(us)"]) - origin)
        duration = float(row["Duration(us)"])
        assert math.isfinite(duration) and duration >= 0
        intervals.append((start, start + duration))
        if row["Name"] in (MATMUL, QUANT):
            shapes = [list(map(int, shape.split(","))) for shape in row["Input Shapes"].strip('"').split(";")]
            k = shapes[0][-1]
            n = shapes[1][1] if row["Name"] == QUANT else shapes[1][0]
            groups[SHAPES[k, n]].append(row)
    attention = sorted(
        [r for r in rows if r["Name"] == "FusedInferAttentionScore"],
        key=lambda r: Decimal(r["Start Time(us)"]),
    )
    groups["attention_full"] = [r for i, r in enumerate(attention) if i % 4 == 3]
    groups["attention_sliding"] = [r for i, r in enumerate(attention) if i % 4 != 3]
    components = {name: detail(items) for name, items in groups.items()}
    task_ms = sum(float(row["Duration(us)"]) for row in rows) / 16000
    span_us = max(stop for start, stop in intervals) - min(start for start, stop in intervals)
    busy_us = union_length(intervals)
    host = defaultdict(lambda: {"count": 0, "self_ms_per_step": 0.})
    with path.with_name("operator_details.csv").open() as f:
        for row in csv.DictReader(f):
            self_us = float(row["Host Self Duration(us)"])
            assert math.isfinite(self_us) and self_us >= 0
            item = host[row["Name"]]
            item["count"] += 1
            item["self_ms_per_step"] += self_us / 16000
    for item in host.values():
        item["calls_per_step"] = item["count"] / 16
    result = json.loads((directory / f"{mode}.json").read_text())
    assert result["profiled"] and result["settings"]["warmup"] == 1
    window_start = float(origin) / 1e6
    window_end = window_start + span_us / 1e6
    overlapping_samples = []
    for line in (directory / f"{mode}-health.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if sample["utc"] <= window_end and sample["end_utc"] >= window_start:
            values = {
                key.strip(): float(value)
                for key, value in (line.split(":", 1) for line in sample["common"].splitlines() if ":" in line)
            }
            overlapping_samples.append({
                "query_start_utc": sample["utc"], "query_end_utc": sample["end_utc"],
                "current_frequency_mhz": values["Aicore curFreq(MHZ)"],
                "rated_frequency_mhz": values["Aicore Freq(MHZ)"],
            })
    return {
        "kernel_csv": str(path), "kernel_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "capture_reconciled": True, "steps": 16, "kernels_per_step": len(rows) / 16,
        "expected_counts": expected, "components": components, "host_ops": dict(host),
        "major_linear_device_ms_per_step": sum(components[n]["device_ms_per_step"] for n in ("qkv", "attention_output", "mlp_gate_up", "mlp_down")),
        "all_kernel_device_ms_per_step": task_ms,
        "other_device_ms_per_step": task_ms - sum(c["device_ms_per_step"] for c in components.values()),
        "profile_span_ms_per_step": span_us / 16000,
        "profile_uncovered_ms_per_step": (span_us - busy_us) / 16000,
        "kernel_window_utc": [window_start, window_end],
        "overlapping_device_queries": overlapping_samples,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    evidence = {
        "profiles": {mode: analyze(args.directory, mode) for mode in ("bf16", "w4a16")},
        "caveats": [
            "Profiler changes eager host overhead; profile spans are not production TPOT.",
            "Task pipeline times overlap; do not sum ratios to infer bubbles or memory bandwidth.",
            "No instruction-level trace or HBM byte counter was collected.",
            "Same-card sequential diagnostic captures are not an unprofiled paired latency benchmark.",
            "A device query overlapping the capture is not a continuous or exact-instant clock measurement; inspect per-mode frequency differences.",
        ],
    }
    (args.directory / "profile-analysis.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    for mode, report in evidence["profiles"].items():
        print(mode, json.dumps({k: v for k, v in report.items() if k not in ("host_ops", "components")}, ensure_ascii=False))
        print(json.dumps(report["components"], ensure_ascii=False, indent=2))
