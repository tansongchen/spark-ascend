#!/usr/bin/env python3
"""Verify native INT4 arithmetic, vLLM weight residency and complete generation."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time


def kernel_checks(model_path):
    import torch
    import torch_npu
    from safetensors import safe_open

    torch.manual_seed(20261020)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    config = json.loads((model_path / "config.json").read_text())
    group = config["quantization_config"]["group_size"]
    results = []
    for suffix in ("self_attn.q_k_v_proj", "self_attn.out_proj", "mlp.gate_proj", "mlp.down_proj"):
        name = f"model.layers.0.{suffix}"
        with safe_open(model_path / index["weight_map"][name + ".weight"], framework="pt") as f:
            words = f.get_tensor(name + ".weight")
            scale = f.get_tensor(name + ".weight_scale")
        q = torch.stack([((words >> (4 * i)) & 15) for i in range(8)], dim=1).flatten(0, 1)
        q = torch.where(q >= 8, q - 16, q)
        packed = words.t().contiguous().npu()
        native = torch_npu.npu_convert_weight_to_int4pack(q.t().contiguous().npu())
        assert torch.equal(native.cpu(), words.t().contiguous()), "Native INT4 packing differs"
        reconstructed = (q.float() * scale.float().repeat_interleave(group, 1)).bfloat16()
        for m in (1, 17):
            x = torch.randn(m, q.shape[1]).bfloat16()
            y = torch_npu.npu_weight_quant_batchmatmul(
                x.npu(), packed, scale.t().contiguous().npu(), antiquant_group_size=group,
            ).cpu().float()
            ref = x.float() @ reconstructed.float().t()
            rel_l2 = ((y - ref).norm() / ref.norm()).item()
            assert torch.isfinite(y).all() and rel_l2 < 0.01, (name, m, rel_l2)
            results.append({"name": name, "m": m, "relative_l2_vs_dequantized_fp32_mm": rel_l2})
    torch.npu.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("/data1/t00995472/models/Spark-X2.5-4B-W4A16-G128"))
    p.add_argument("--npu", type=int, default=2, choices=range(1, 8))
    p.add_argument("--result", type=Path, default=Path("/data1/t00995472/spark-w4a16/validation.json"))
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--check-kernels", action="store_true")
    args = p.parse_args()
    query_env = dict(os.environ)
    query_env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    status = subprocess.check_output(["npu-smi", "info"], env=query_env, text=True)
    if f"No running processes found in NPU {args.npu}" not in status:
        raise RuntimeError(f"Physical NPU {args.npu} is occupied")
    os.environ.update({
        "ASCEND_RT_VISIBLE_DEVICES": str(args.npu),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "HF_HUB_OFFLINE": "1",
        "SPARK_W4A16": "0" if args.bf16 else "1",
        "SPARK_FUSED_DECODE": "0", "SPARK_FUSED_GRAPH": "0",
    })
    os.environ.pop("SPARK_LOGITS_DIR", None)
    os.environ.pop("SPARK_FORCE_TOKENS", None)
    report = {
        "model": str(args.model.resolve()), "physical_npu": args.npu,
        "mode": "BF16" if args.bf16 else "W4A16",
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "torch-npu", "vllm", "vllm-ascend", "spark-w4a16-ascend")},
    }
    if args.check_kernels:
        # Separate process releases its NPU context before the vLLM worker starts.
        kernel_result = args.result.with_suffix(".kernels.json")
        command = (
            "import json; from pathlib import Path; "
            "from utils.validate_spark_w4a16 import kernel_checks; "
            f"Path({str(kernel_result)!r}).write_text(json.dumps(kernel_checks(Path({str(args.model)!r})), indent=2))"
        )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([os.sys.executable, "-B", "-c", command], check=True, cwd=Path(__file__).resolve().parents[1])
        report["kernel_checks"] = json.loads(kernel_result.read_text())
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    extra = {} if args.bf16 else {"worker_extension_cls": "utils.spark_w4a16.W4A16WorkerExtension"}
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
        ("请用Python写一个函数，返回两个数的和。", "return"),
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
    if not args.bf16:
        audit = llm.collective_rpc("spark_quant_audit")[0]
        layers = audit["quantized_linear_layers"]
        assert len(layers) == 144, len(layers)
        assert all(r["weight_dtype"] == "torch.int32" and r["scale_dtype"] == "torch.bfloat16" for r in layers)
        assert all(r["calls"] > 64 and r["input_dtypes"] == ["torch.bfloat16"] for r in layers)
        assert all(1 in r["token_counts"] and 256 in r["token_counts"] for r in layers)
        assert audit["tied_lm_head"] and audit["embedding_dtype"] == "torch.bfloat16"
        report["runtime_audit"] = audit
    report["passed"] = True
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"PASS: {args.result}", flush=True)


if __name__ == "__main__":
    main()
