"""Run the existing stream benchmark with source identity and device telemetry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = a.benchmark_args[1:] if a.benchmark_args[:1] == ["--"] else a.benchmark_args
    if "--npu" not in args or int(args[args.index("--npu") + 1]) not in range(1, 8):
        p.error("Pass an explicit --npu 1..7")
    npu = args[args.index("--npu") + 1]
    a.directory.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, "-B", str(ROOT / "benchmark.py"),
           *args, "--result", str(a.directory / "result.json")]
    files = [ROOT / name for name in (
        "pyproject.toml", "utils/register_spark.py",
        "benchmark.py", "utils/packing.py", "utils/bf16_codec.py",
        "utils/spark_fused_decode.py",
        "utils/spark_w4a16_fused.py",
        "utils/spark_w4a16.py",
        "utils/w4a16_packing.py")]
    snapshot = a.directory / "sources"
    for f in files:
        target = snapshot / f.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f.read_bytes())
    record = {"command": cmd, "cwd": str(ROOT), "start_utc": time.time(),
              "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    stop = threading.Event()
    def monitor():
        with (a.directory / "health.jsonl").open("w") as stream:
            while not stop.is_set():
                start = time.time()
                r = subprocess.run(["npu-smi", "info", "-i", npu, "-t", "common"],
                                   capture_output=True, text=True, timeout=20)
                stream.write(json.dumps({"utc": start, "sample_end_utc": time.time(),
                                         "returncode": r.returncode, "common": r.stdout,
                                         "stderr": r.stderr}) + "\n")
                stream.flush()
                stop.wait(2)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with (a.directory / "benchmark.log").open("w") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = proc.pid
            (a.directory / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
            record["returncode"] = proc.wait()
    finally:
        stop.set()
        thread.join()
        record["end_utc"] = time.time()
        (a.directory / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    result = a.directory / "result.json"
    if result.exists():
        print(json.dumps(json.loads(result.read_text())["summary"], indent=2), flush=True)
    print(f"exit={record.get('returncode')} evidence={a.directory}", flush=True)
    return record.get("returncode", 1)


if __name__ == "__main__":
    raise SystemExit(main())
