"""Same-engine native/fused W4A16 teacher-forced logits; not a timing run."""
import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ValidationWorker:
    def spark_readability_mode(self,mode):
        import torch
        from utils.spark_w4a16_fused import W4A16FusedState
        model=self.get_model();state=model.model._spark_fused_state
        torch.npu.synchronize()
        if mode not in ('native','fused'):
            raise ValueError(mode)
        state.history_ready=False;state.last_length=None
        state.forward=state.original if mode=='native' else W4A16FusedState.forward.__get__(state)
        model._logit_dir=None;model._forced=None;model._validation_active=False
        return {'mode':mode}

    def spark_validation_mode(self, mode, directory, forced, debug_step=None, debug_cache_layers=None):
        import torch
        from utils.spark_w4a16_fused import W4A16FusedState
        model = self.get_model()
        state = model.model._spark_fused_state
        torch.npu.synchronize()
        if mode not in ("native", "fused"):
            raise ValueError(mode)
        state.history_ready = False
        state.last_length = None
        # The outer validation wrapper remains active in both modes.
        state.forward = state.original if mode == "native" else W4A16FusedState.forward.__get__(state)
        model._logit_dir = Path(directory)
        model._logit_dir.mkdir(parents=True, exist_ok=False)
        model._forced = forced
        model._validation_step = 0
        model._debug_step = debug_step
        model._debug_cache_layers = set(debug_cache_layers or [])
        if debug_step is not None and not hasattr(model, "_debug_hooks"):
            model._debug_hooks = []
            def hook(layer_index, name):
                def capture(module, inputs, output):
                    if model._validation_step != model._debug_step or not model._validation_active:
                        return
                    value = output[0] if isinstance(output, tuple) else output
                    torch.save(value.detach().cpu().clone(), model._logit_dir/f"layer-{layer_index:02d}-{name}.pt")
                    if name in ("norm", "postnorm") and isinstance(output, tuple) and len(output) == 2:
                        torch.save(output[1].detach().cpu().clone(),
                                   model._logit_dir/f"layer-{layer_index:02d}-{name}-residual.pt")
                    if name == "attention" and layer_index in model._debug_cache_layers:
                        md = state.metadata()[module.layer_name]
                        length = md.seq_lens_list[0]
                        first = max(0, length-1-512)//128 if (layer_index+1)%4 else 0
                        count = (length+127)//128
                        ids = md.block_tables[0, first:count].long()
                        cache = module.kv_cache
                        if isinstance(cache, (list, tuple)) and len(cache) == 1:
                            cache = cache[0]
                        torch.save({"length": length, "first_page": first,
                                    "key": cache[0].index_select(0, ids).cpu(),
                                    "value": cache[1].index_select(0, ids).cpu()},
                                   model._logit_dir/f"layer-{layer_index:02d}-cache.pt")
                return capture
            def attention_input_hook(layer_index):
                def capture(module, inputs):
                    if model._validation_step != model._debug_step or not model._validation_active:
                        return
                    for name, value in zip(("q", "k", "v"), inputs):
                        torch.save(value.detach().cpu().clone(),
                                   model._logit_dir/f"layer-{layer_index:02d}-{name}.pt")
                return capture
            def output_projection_input_hook(layer_index):
                def capture(module, inputs):
                    if model._validation_step == model._debug_step and model._validation_active:
                        torch.save(inputs[0].detach().cpu().clone(),
                                   model._logit_dir/f"layer-{layer_index:02d}-merged.pt")
                return capture
            for i, layer in enumerate(model.model.layers):
                model._debug_hooks.append(layer.self_attn.core_attention.register_forward_pre_hook(
                    attention_input_hook(i)))
                model._debug_hooks.append(layer.self_attn.out_proj.register_forward_pre_hook(
                    output_projection_input_hook(i)))
                for name, module in (
                    ("norm",layer.input_layernorm), ("postnorm",layer.post_attention_layernorm),
                    ("qkv",layer.self_attn.q_k_v_proj), ("g",layer.self_attn.g_proj),
                    ("attention",layer.self_attn.core_attention), ("oproj",layer.self_attn.out_proj),
                    ("gateup",layer.mlp.gate_up_proj), ("down",layer.mlp.down_proj)):
                    model._debug_hooks.append(module.register_forward_hook(hook(i,name)))
            original_launch = state.launch_layers
            def debug_launch(stream, hidden, residual, positions):
                if model._validation_step != model._debug_step or not model._validation_active:
                    return original_launch(stream,hidden,residual,positions)
                for i, entry in enumerate(state.entries):
                    torch.save(hidden.cpu(),model._logit_dir/f"layer-{i:02d}-norm.pt")
                    torch.save(residual.cpu(),model._logit_dir/f"layer-{i:02d}-norm-residual.pt")
                    ptrs=entry["ptrs"]
                    ptrs[0],ptrs[1],ptrs[10]=hidden.data_ptr(),residual.data_ptr(),positions.data_ptr()
                    out,resout=state.output[i%2],state.residual[i%2]
                    torch.npu.current_stream().synchronize()
                    state.launch(stream,ptrs,*[c.data_ptr() for c in entry["caches"]],state.ws.data_ptr(),
                                 out.data_ptr(),resout.data_ptr(),entry["rd"],0,1e-6,1e-6)
                    torch.npu.current_stream().synchronize()
                    ws=state.ws.cpu()
                    for name,offset,count in (("q",0,4096),("k",8192,1024),("v",10240,1024),
                                              ("g",12288,16),("merged",80128,4096),
                                              ("oproj",88320,2560),("mid",93440,10240),
                                              ("down",113920,2560),("resmid",119040,2560)):
                        torch.save(ws[offset:offset+count*2].view(torch.bfloat16).clone(),
                                   model._logit_dir/f"layer-{i:02d}-{name}.pt")
                    if i in model._debug_cache_layers:
                        length = int(positions.item())+1
                        first = max(0,length-1-512)//128 if entry["rd"] == 256 else 0
                        count = (length+127)//128
                        torch.save({"length": length, "first_page": first,
                                    "key": entry["caches"][0][first:count].cpu(),
                                    "value": entry["caches"][1][first:count].cpu()},
                                   model._logit_dir/f"layer-{i:02d}-cache.pt")
                    hidden,residual=out,resout
                return hidden
            state.launch_layers = debug_launch
        return {"mode": mode, "library": state.library_path, "decode_steps": state.steps}

    def spark_validation_status(self):
        from utils.spark_fused_decode import fused_status
        return fused_status(self.get_model())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--npu", type=int, choices=range(1, 8), default=2)
    p.add_argument("--contexts", type=int, nargs="+", default=[128, 1024, 32768, 36800])
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--debug-step", type=int, help="Capture layer intermediates at one failing decode step")
    p.add_argument("--debug-cache-layers", type=int, nargs="+", choices=range(36),
                   help="Capture logical KV pages for selected layers at --debug-step")
    p.add_argument("--model", default="/data1/t00995472/models/Spark-X2.5-4B-W4A16-G128")
    p.add_argument('--readability-only',action='store_true',help='同一引擎原生/融合自由生成三类自然问题；不导出 logits，不作性能计时')
    p.add_argument('--thinking',action='store_true',help='可读性检查保留模型思考模式，默认关闭以观察完整回答')
    a = p.parse_args()
    if not 2 <= a.steps <= (512 if a.readability_only else 128) or min(a.contexts) < 1 or max(a.contexts) + a.steps > 40960:
        p.error("Supported contexts+steps<=40960, 2<=steps<=128")
    if a.debug_cache_layers and a.debug_step is None:
        p.error("--debug-cache-layers requires --debug-step")
    info = subprocess.check_output(["npu-smi", "info"], text=True)
    if f"No running processes found in NPU {a.npu}" not in info:
        raise RuntimeError(f"NPU {a.npu} occupied")
    os.environ.update(ASCEND_RT_VISIBLE_DEVICES=str(a.npu), SPARK_W4A16="1",
                      SPARK_FUSED_DECODE="1", SPARK_FUSED_GRAPH="0",
                      SPARK_FUSED_LIBRARY=str(a.library.resolve()),
                      SPARK_FUSED_CAPACITY=str(max(a.contexts) + a.steps),
                      HF_HUB_OFFLINE="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    os.environ.pop("SPARK_LOGITS_DIR", None)
    os.environ.pop("SPARK_FORCE_TOKENS", None)
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    seed = tokenizer.encode("这是用于测量长上下文推理性能的文本。请继续生成内容。", add_special_tokens=False)
    seed = [t for t in seed if t not in tokenizer.all_special_ids]
    llm = LLM(model=a.model, trust_remote_code=True, dtype="bfloat16",
              tensor_parallel_size=1, max_num_seqs=1, block_size=128,
              max_model_len=max(a.contexts)+a.steps, max_num_batched_tokens=32768,
              gpu_memory_utilization=0.7, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=True,
              worker_extension_cls="utils.validate_w4a16_model.ValidationWorker")
    results = []
    try:
        if a.readability_only:
            prompts=[
                '请用三句话解释为什么晴天的天空通常是蓝色的，适合中学生理解。',
                '请计算17乘23，写出计算过程。然后给出一行Python代码，将列表[3, 1, 2]从小到大排序。',
                '请给同事写一封约100字的中文邮件，说明模型性能优化已完成初步测试，但仍需核对输出质量，邀请他明天下午一起复核。',
            ]
            rows=[]
            for mode in ('native','fused'):
                llm.collective_rpc('spark_readability_mode',args=(mode,))
                for number,prompt in enumerate(prompts):
                    ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                                                      add_generation_prompt=True,enable_thinking=a.thinking)
                    if isinstance(ids,Mapping):
                        ids=ids['input_ids']
                    if len(ids)>max(a.contexts):
                        raise ValueError('Readability prompt exceeds configured context capacity')
                    params=SamplingParams(temperature=0,max_tokens=a.steps,ignore_eos=False,detokenize=True)
                    output=llm.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)[0].outputs[0]
                    rows.append({'mode':mode,'prompt':prompt,'prompt_tokens':len(ids),
                                 'token_ids':list(output.token_ids),'text':output.text,
                                 'finish_reason':output.finish_reason,'stop_reason':output.stop_reason})
                    print(json.dumps({'mode':mode,'prompt_number':number,'text':output.text},ensure_ascii=False),flush=True)
            status=llm.collective_rpc('spark_validation_status')
            # These requests configure no explicit stop strings/tokens. On EOS,
            # vLLM removes the terminating token from the returned token list,
            # but sampling that EOS still consumed one decode invocation.
            expected=sum(max(0,len(r['token_ids'])-1+int(r['finish_reason']=='stop'))
                         for r in rows if r['mode']=='fused')
            execution_ok=all(s['ready'] and s['decode_steps']==expected for s in status)
            report={'scope':'free-generation readability inspection; no forced tokens, logits export or timing claim',
                    'rows':rows,'fused_status':status,'execution_ok':execution_ok,'expected_decode_steps':expected,
                    'library':str(a.library),'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),
                    'thinking':a.thinking,'human_review_required':True,'quality_benchmark':False}
            (a.output/'readability.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            return 0 if execution_ok else 1
        for length in a.contexts:
            directory = a.output / str(length)
            directory.mkdir()
            ids = (seed * ((length+len(seed)-1)//len(seed)))[:length]
            params = SamplingParams(temperature=0, max_tokens=a.steps, min_tokens=a.steps,
                                    ignore_eos=True, detokenize=False)
            llm.collective_rpc("spark_validation_mode", args=("native", str(directory/"logits-base"), None, a.debug_step, a.debug_cache_layers))
            baseline = list(llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0].outputs[0].token_ids)
            llm.collective_rpc("spark_validation_mode", args=("fused", str(directory/"logits-fused"), baseline, a.debug_step, a.debug_cache_layers))
            candidate = list(llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0].outputs[0].token_ids)
            assert candidate == baseline and len(baseline) == a.steps
            (directory/"tokens.json").write_text(json.dumps(baseline)+"\n")
            subprocess.run([sys.executable, "-B", str(ROOT/"utils/compare_spark_fused.py"), str(directory)], check=True)
            comparison = json.loads((directory/"logit-comparison.json").read_text())
            # Predeclared implementation screen, not a task-quality benchmark.
            passed = (comparison["decode_steps"] == a.steps-1
                      and comparison["decode_max_nrmse"] <= 0.02
                      and comparison["decode_min_cosine"] >= 0.999
                      and comparison["decode_top1_agreement"] >= 0.95)
            results.append({"context": length, "passed": passed,
                            **{k:v for k,v in comparison.items() if k != "steps"}})
            print(json.dumps(results[-1]), flush=True)
        status = llm.collective_rpc("spark_validation_status")
        expected = len(a.contexts) * (a.steps-1)
        passed = all(r["passed"] for r in results) and all(s["decode_steps"] == expected for s in status)
        report = {"passed": passed, "contexts": results, "fused_status": status,
                  "library": str(a.library), "library_sha256": hashlib.sha256(a.library.read_bytes()).hexdigest(),
                  "forced_outputs_are_not_a_free_generation_equality_claim": True,
                  "screen": {"nrmse_max":0.02,"cosine_min":0.999,"top1_agreement_min":0.95},
                  "quality_benchmark": False}
        (a.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        return 0 if passed else 1
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
