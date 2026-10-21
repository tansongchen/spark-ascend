"""INT4 tile round-trip, device unpack and independent full-layer CPU reference."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/data1/t00995472/models/Spark-X2.5-4B-W4A16-G128"))
    p.add_argument("--npu", type=int, choices=range(1, 8), default=2)
    p.add_argument("--layer", type=int, choices=range(36), default=0)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--slot32", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unpack-only", action="store_true")
    p.add_argument("--debug", help="Save full workspace for diagnostic stage stamps")
    a = p.parse_args()
    info = subprocess.check_output(["npu-smi", "info"], text=True)
    if f"No running processes found in NPU {a.npu}" not in info:
        raise RuntimeError(f"NPU {a.npu} occupied")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(a.npu)
    import torch
    import torch_npu
    from utils.w4a16_packing import pack_tile, unpack_tile
    from utils.packing import tile_weights
    from utils import validate_kernel as v

    torch.set_num_threads(8)
    torch.manual_seed(a.seed)
    torch.npu.set_device(0)
    lib = ctypes.CDLL(str(a.build / "libdecode_layer_w4a16.so"))
    probe = lib.launch_w4a16_unpack_probe
    probe.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
    probe.restype = None
    checks = []
    for kind, (k, m) in enumerate(((2560, 8), (4096, 8), (10240, 4))):
        # All 16 codes, nonuniform and signed scales, several magnitudes.
        q = torch.randint(-8, 8, (m, k), dtype=torch.int8)
        scales = (torch.randn(m, k // 128) * torch.logspace(-4, 2, k // 128)).bfloat16()
        packed = pack_tile(q, scales, m)
        uq, us = unpack_tile(packed, m, k, m)
        assert torch.equal(q, uq) and torch.equal(scales.float(), us), "CPU tile round-trip"
        ref = tile_weights((q.float() * scales.float().repeat_interleave(128, 1)).bfloat16(), m)
        src = packed.npu()
        dst = torch.full((m * k + 128,), float("nan"), dtype=torch.bfloat16, device="npu")
        for iteration in range(3):
            dst.fill_(float("nan"))
            torch.npu.synchronize()
            probe(torch.npu.current_stream().npu_stream, src.data_ptr(), dst.data_ptr(), kind)
            torch.npu.synchronize()
            actual = dst.cpu()
            equal = torch.equal(actual[:m*k].view(torch.int16), ref.flatten().view(torch.int16))
            if not equal:
                count = int((actual[:m*k] != ref.flatten()).sum())
                raise AssertionError(f"Unpack kind={kind}: {count}/{m*k} differ")
            assert actual[m*k:].isnan().all(), "Unpack output guard overwritten"
        checks.append({"k": k, "m": m, "values": m*k, "bit_exact": True, "repeats": 3})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.with_suffix(".unpack.json").write_text(json.dumps(checks, indent=2) + "\n")
    if a.unpack_only:
        print(json.dumps(checks), flush=True)
        return 0
    from utils.w4a16_reference import make_reference
    encoding = json.loads((a.build / "execution.json").read_text())["weight_encoding"]
    load, pack = make_reference(a.model, encoding)
    a.variant, a.stop = "w4a16", "full"
    return v.run(a, weights_loader=load, input_packer=pack)


if __name__ == "__main__":
    raise SystemExit(main())
