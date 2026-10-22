#!/usr/bin/env python3
"""Verify the upstream mixed INT4/INT8 checkpoint, native arithmetic and generation."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.model_config import DEFAULT_MODEL, validate_config, validate_audit, quantization_for


def packing_checks():
    import torch
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from utils.int4_packing import repack_int4, int4_scales
    torch.set_num_threads(4)
    q = (torch.arange(64 * 256).reshape(64, 256) % 16 - 8).to(torch.int8)
    zp = (torch.arange(64 * 8).reshape(64, 8) % 16 - 8).to(torch.int8)
    scales = torch.linspace(0.001, 0.1, 64 * 8).reshape(64, 8).bfloat16()
    for scale in (scales, scales.float()):
        packed, s, offset = repack_int4(pack_to_int32(q, 4), scale, pack_to_int32(zp, 4, packed_dim=0))
        values = torch.stack([(packed >> (4*i)) & 15 for i in range(8)], -1).flatten(1)
        values = torch.where(values >= 8, values - 16, values).t()
        assert torch.equal(q, values)
        assert torch.equal(s.t(), scales) and torch.equal(-offset.t(), zp)
    for bad in (scales.float() + 1e-7, torch.zeros_like(scales), torch.full_like(scales, float('nan'))):
        try:
            int4_scales(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid or lossy scale conversion accepted')
    return dict(all_16_weight_codes=True, all_16_zero_points=True, fp32_scales_checked_losslessly=True)


def kernel_checks(model_path):
    import torch
    import torch_npu
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationScheme
    from utils.int4_packing import Checkpoint, repack_int4

    torch.set_num_threads(8)
    torch.manual_seed(20261022)
    config = validate_config(json.loads((model_path / "config.json").read_text()))
    checkpoint = Checkpoint(model_path)
    results = []
    names = [f"model.layers.{i}.{suffix}" for i in (0, 35)
             for suffix in ("self_attn.q_k_v_proj", "self_attn.out_proj", "mlp.gate_proj",
                            "mlp.up_proj", "mlp.down_proj")] + ["lm_head"]
    for name in names:
        group = quantization_for(config, name)
        int4 = group["weights"]["num_bits"] == 4
        scale = checkpoint.read(name + ".weight_scale")
        if int4:
            state = {key: checkpoint.read(name + "." + key) for key in
                     ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")}
            # Independent upstream decompressor is the numerical oracle.
            scheme = QuantizationScheme.model_validate({k: v for k, v in group.items() if k != "format"})
            reference = PackedQuantizationCompressor.decompress(state, scheme)["weight"].bfloat16()
            packed, native_scale, offset = repack_int4(state["weight_packed"], scale, state["weight_zero_point"])
            # Independently decode Ascend's signed nibble layout and verify exact BF16 weights.
            codes = torch.stack([(packed >> (4*i)) & 15 for i in range(8)], -1).flatten(1)
            codes = torch.where(codes >= 8, codes - 16, codes)
            restored = ((codes.float() + offset.float().repeat_interleave(32, 0))
                        * native_scale.float().repeat_interleave(32, 0)).t().bfloat16()
            assert torch.equal(reference, restored), name
            device_weight = packed.npu()
            native = torch_npu.npu_convert_weight_to_int4pack(codes.int().contiguous().npu())
            assert torch.equal(native.cpu(), packed), name
            device_scale, device_offset = native_scale.npu(), offset.npu()
        else:
            weight = checkpoint.read(name + ".weight")
            assert weight.dtype == torch.int8 and scale.dtype == torch.float32
            device_weight, device_scale = weight.t().contiguous().npu(), scale.flatten().npu()
        k = reference.shape[1] if int4 else weight.shape[1]
        for m in (1, 17):
            x = torch.randn(m, k).bfloat16()
            if int4:
                y = torch_npu.npu_weight_quant_batchmatmul(
                    x.npu(), device_weight, device_scale, antiquant_offset=device_offset,
                    antiquant_group_size=32).cpu().float()
                ref = x.float() @ reference.float().t()
            else:
                qx, sx = torch_npu.npu_dynamic_quant(x.npu())
                y = torch_npu.npu_quant_matmul(qx, device_weight, device_scale,
                                              pertoken_scale=sx, output_dtype=torch.bfloat16).cpu().float()
                # Check the dynamic-activation/INT8-weight matmul including FP32 scales.
                ref = (qx.cpu().float() @ weight.float().t()) * sx.cpu().float()[:, None] * scale.float().t()
            rel_l2 = float((y-ref).norm() / ref.norm().clamp_min(1e-12))
            assert torch.isfinite(y).all() and rel_l2 < 0.01, (name, m, rel_l2)
            row = dict(name=name, bits=4 if int4 else 8, m=m, relative_l2=rel_l2,
                       reference="upstream dequantized BF16" if int4 else "dynamic INT8 matmul with FP32 scales")
            results.append(row)
            print(json.dumps(row), flush=True)
    torch.npu.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--npu", type=int, default=2, choices=range(1, 8))
    p.add_argument("--result", type=Path, default=Path("/data1/t00995472/spark-int4/validation.json"))
    p.add_argument("--kernels-only", action="store_true")
    p.add_argument("--packing-only", action="store_true", help="CPU 位级布局与 scale 防损失检查，不使用 NPU")
    p.add_argument("--check-kernels", action="store_true")
    args = p.parse_args()
    checks = packing_checks()
    if args.packing_only:
        print(json.dumps(checks), flush=True)
        return
    from utils.device import select_idle_npu
    select_idle_npu(args.npu)
    os.environ.update({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "HF_HUB_OFFLINE": "1",
        "SPARK_INT4": "1",
        "SPARK_FUSED_DECODE": "0", "SPARK_FUSED_GRAPH": "0",
    })
    os.environ.pop("SPARK_LOGITS_DIR", None)
    os.environ.pop("SPARK_FORCE_TOKENS", None)
    report = {
        "model": str(args.model.resolve()), "physical_npu": args.npu,
        "mode": "upstream mixed INT4 G32 / dynamic INT8",
        "packing_checks": checks,
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "torch-npu", "vllm", "vllm-ascend", "spark-ascend")},
    }
    validate_config(json.loads((args.model / "config.json").read_text()))
    if args.check_kernels or args.kernels_only:
        # Separate process releases its NPU context before the vLLM worker starts.
        kernel_result = args.result.with_suffix(".kernels.json")
        command = (
            "import json; from pathlib import Path; "
            "from utils.validate_spark_int4 import kernel_checks; "
            f"Path({str(kernel_result)!r}).write_text(json.dumps(kernel_checks(Path({str(args.model)!r})), indent=2))"
        )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([os.sys.executable, "-B", "-c", command], check=True, cwd=Path(__file__).resolve().parents[1])
        report["kernel_checks"] = json.loads(kernel_result.read_text())
        if args.kernels_only:
            return
    from transformers import AutoTokenizer
    from vllm import LLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    extra = {"worker_extension_cls": "utils.spark_int4.Int4WorkerExtension"}
    llm = LLM(
        model=str(args.model), trust_remote_code=True, dtype="bfloat16",
        tensor_parallel_size=1, max_num_seqs=1, max_model_len=2048,
        max_num_batched_tokens=256, block_size=128, gpu_memory_utilization=0.2,
        enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=True,
        seed=17, **extra,
    )
    try:
        run_requests(llm, tokenizer, args, report)
    finally:
        llm.llm_engine.engine_core.shutdown()


def run_requests(llm, tokenizer, args, report):
    from vllm import SamplingParams

    prompts = [
        ("请用一句中文介绍你自己。", None),
        ("2加3等于几？只输出一个阿拉伯数字。", "5"),
        ("中国的首都是哪里？请只回答城市名称。", "北京"),
        ("请用Python写一个函数，返回两个数的和。只输出函数代码，不要解释、注释或示例。", "return"),
    ]
    rows = []
    for prompt, expected in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False,
        )
        start = time.monotonic()
        result = llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0, max_tokens=256, logprobs=1),
            use_tqdm=False,
        )[0].outputs[0]
        assert result.token_ids and result.text.strip(), "Empty generation"
        assert result.finish_reason == "stop", f"Generation did not finish normally: {result.text}"
        assert all(math.isfinite(row[token].logprob) for token, row in zip(result.token_ids, result.logprobs))
        if expected is not None:
            assert expected in result.text, f"Smoke answer mismatch: {prompt}: {result.text}"
        row = {
            "prompt": prompt, "input_tokens": len(ids), "text": result.text,
            "token_ids": list(result.token_ids), "output_tokens": len(result.token_ids),
            "finish_reason": result.finish_reason, "elapsed_seconds": time.monotonic() - start,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    # Exercise chunked prefill beyond 256 tokens, then 64 decode steps.
    seed = tokenizer.encode("这是一段用于验证长输入处理的中文文本。", add_special_tokens=False)
    long_ids = (seed * (1024 // len(seed) + 1))[:1024]
    output = llm.generate(
        [{"prompt_token_ids": long_ids}],
        SamplingParams(temperature=0, min_tokens=64, max_tokens=64, ignore_eos=True, logprobs=1),
        use_tqdm=False,
    )[0].outputs[0]
    assert len(output.token_ids) == 64
    assert all(math.isfinite(row[token].logprob) for token, row in zip(output.token_ids, output.logprobs))
    report["chat_requests"] = rows
    report["chunked_prefill"] = {
        "input_tokens": 1024, "output_tokens": 64, "all_selected_logprobs_finite": True,
        "token_ids": list(output.token_ids), "finish_reason": output.finish_reason,
    }
    audit = llm.collective_rpc("spark_quant_audit")
    validate_audit(audit, minimum_calls=64)
    assert all(1 in row["token_counts"] for row in audit[0]["quantized_linear_layers"])
    assert all(256 in row["token_counts"] for row in audit[0]["quantized_linear_layers"] if row["name"] != "lm_head")
    report["runtime_audit"] = audit
    report["passed"] = True
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"PASS: {args.result}", flush=True)


if __name__ == "__main__":
    main()
