#!/usr/bin/env python3
"""Audit saved official INT4/INT8 timing and device samples without executing the model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def analyze(directory):
    result = json.loads((directory / "result.json").read_text())
    execution = json.loads((directory / "execution.json").read_text())
    settings = result["settings"]
    assert execution["returncode"] == 0
    assert settings["int4"] and settings["enforce_eager"]
    assert not result["profiled"] and not result["validation_overhead"]
    assert settings["warmup"] == 1 and settings["runs"] == 3
    assert (settings["input_tokens"], settings["output_tokens"]) == (32768, 4096)
    assert result["batch_size"] == 1 and settings["npu"] != 0
    assert len(result["runs"]) == 3
    samples = [json.loads(line) for line in (directory / "health.jsonl").read_text().splitlines()]
    valid_samples = []
    for sample in samples:
        if sample.get("returncode") != 0:
            continue
        sample["values"] = {
            key.strip(): float(value.strip())
            for key, value in (line.split(":", 1) for line in sample.get("common",sample.get("stdout","")).splitlines() if ":" in line)
        }
        valid_samples.append(sample)
    rows = []
    for number, run in enumerate(result["runs"], 1):
        assert (run["input_tokens"], run["output_tokens"]) == (32768, 4096)
        arrivals = run["token_arrivals_s"]
        assert len(run["token_ids"]) == len(arrivals) == 4096
        assert all(a <= b for a, b in zip(arrivals, arrivals[1:]))
        assert run.get("first_stream_update_tokens",1) == 1
        merged = run.get("merged_stream_updates",[])
        assert sum(m["tokens"]-1 for m in merged) == sum(a == b for a,b in zip(arrivals,arrivals[1:]))
        assert hashlib.sha256(json.dumps(run["token_ids"]).encode()).hexdigest() == run["token_ids_sha256"]
        assert abs(run["ttft_ms"] - arrivals[0] * 1000) < 1e-7
        assert abs(run["tpot_ms"] - (arrivals[-1] - arrivals[0]) * 1000 / 4095) < 1e-7
        assert abs(run["e2e_ms"] - arrivals[-1] * 1000) < 1e-7
        decode = [
            sample["values"] for sample in valid_samples
            if run["first_token_utc"] <= sample.get("utc",sample.get("time"))
            and sample.get("sample_end_utc",sample.get("query_end_time")) <= run["last_token_utc"]
        ]
        assert decode, "No device sample wholly inside measured decode"
        rows.append({
            "run": number, "ttft_ms": run["ttft_ms"], "tpot_ms": run["tpot_ms"], "e2e_ms": run["e2e_ms"],
            "merged_stream_updates": len(merged),
            "decode_sample_count": len(decode),
            "decode_frequency_mhz": dict(Counter(s["Aicore curFreq(MHZ)"] for s in decode)),
            "decode_all_samples_at_rated_clock": all(s["Aicore curFreq(MHZ)"] == s["Aicore Freq(MHZ)"] for s in decode),
            "decode_temperature_c": [min(s["Temperature(C)"] for s in decode), max(s["Temperature(C)"] for s in decode)],
        })
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        assert abs(result["summary"][metric]["mean"] - statistics.mean(r[metric] for r in rows)) < 1e-7
        assert abs(result["summary"][metric]["median"] - statistics.median(r[metric] for r in rows)) < 1e-7
    return {
        "timing_audit_passed": True, "physical_npu": settings["npu"], "runs": rows,
        "summary": result["summary"],
        "all_three_outputs_equal": len({r["token_ids_sha256"] for r in result["runs"]}) == 1,
        "failed_health_samples": len(samples) - len(valid_samples),
        "runtime_quantized_layers": len(result["int4_worker_audit"][0]["quantized_linear_layers"]),
        "mean_decode_tokens_per_second": 1000 / result["summary"]["tpot_ms"]["mean"],
        "caveat": "Sparse frequency samples do not prove constant frequency. TPOT uses first/last client receipt; later merged updates cannot measure individual inter-token latency. No profiler or logits export is enabled.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = analyze(args.directory)
    (args.directory / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
