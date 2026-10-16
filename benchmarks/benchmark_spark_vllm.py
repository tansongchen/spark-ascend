#!/usr/bin/env python3
"""单请求、精确 token 长度的 vLLM Ascend TTFT/TPOT 基准。"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if os.environ.get('SPARK_BENCH_DEBUG') == '1':
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='/data1/t00995472/models/Spark-X2.5-4B')
    p.add_argument('--npu', type=int, default=2)
    p.add_argument('--input-tokens', type=int, default=32768)
    p.add_argument('--output-tokens', type=int, default=4096)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.7)
    p.add_argument('--max-num-batched-tokens', type=int, default=32768)
    p.add_argument('--enforce-eager', action='store_true')
    p.add_argument('--fused-decode',action='store_true',help='使用本地 stage_prefetch 单 token decode；当前要求 --enforce-eager')
    p.add_argument('--fused-graph',action='store_true',help='捕获融合 decode 的 36 层调用链；要求 --fused-decode --enforce-eager')
    p.add_argument('--fused-library',help='选择 ABI 和权重布局兼容的 BF16 融合候选共享库')
    p.add_argument('--validation-dir',help='导出前 128 步 logits；此模式不得当作性能结果')
    p.add_argument('--force-tokens-from',help='校验时强制使用 JSON token 列表，保持相同前缀')
    p.add_argument('--profile-dir', help='启用 Ascend worker profiler；采集运行不作为性能基线')
    p.add_argument('--profile-output-tokens', type=int, default=128, help='采集请求的输出长度；引擎上下文容量仍按 --output-tokens 配置')
    p.add_argument('--profile-delay', type=int, default=16, help='跳过的 worker 步数，避开 prefill')
    p.add_argument('--profile-steps', type=int, default=16, help='采集的 worker decode 步数')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--result', default='/data1/t00995472/spark-benchmark/result.json')
    a = p.parse_args()
    if a.fused_graph and not a.fused_decode:
        p.error('--fused-graph 要求 --fused-decode')
    if a.fused_library:
        if not a.fused_decode:
            p.error('--fused-library 要求 --fused-decode')
        if not Path(a.fused_library).is_file():
            p.error('--fused-library 指定的共享库不存在')
        a.fused_library=str(Path(a.fused_library).resolve())
    if a.fused_decode and (not a.enforce_eager or a.input_tokens+a.output_tokens>40960):
        p.error('当前融合适配要求 --enforce-eager，且总上下文不超过 40960')
    if (a.validation_dir or a.force_tokens_from) and not a.enforce_eager:
        p.error('logits 校验要求 --enforce-eager')
    if a.force_tokens_from and not a.validation_dir:
        p.error('--force-tokens-from 要求同时指定 --validation-dir')
    if a.npu <= 0 or a.npu > 7:
        p.error('本机仅允许使用物理 NPU 1～7，禁止 NPU 0')
    if min(a.input_tokens, a.runs, a.max_num_batched_tokens) < 1 or a.output_tokens < 2 or a.warmup < 0:
        p.error('输入长度、运行次数和 token 预算须为正；输出至少 2 token；预热次数非负')
    if a.profile_delay < 1 or a.profile_steps < 1:
        p.error('采集延迟和步数须为正')
    if a.profile_dir and min(a.output_tokens, a.profile_output_tokens) <= a.profile_delay + a.profile_steps:
        p.error('采集请求输出长度须大于跳过步数与采集步数之和')
    return a


def preflight(a):
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['SPARK_FUSED_DECODE']='1' if a.fused_decode else '0'
    os.environ['SPARK_FUSED_GRAPH']='1' if a.fused_graph else '0'
    os.environ['SPARK_FUSED_CAPACITY']=str(a.input_tokens+a.output_tokens)
    if a.fused_library:
        os.environ['SPARK_FUSED_LIBRARY']=a.fused_library
    for name,value in [('SPARK_LOGITS_DIR',a.validation_dir),('SPARK_FORCE_TOKENS',a.force_tokens_from)]:
        if value: os.environ[name]=str(Path(value).resolve())
        else: os.environ.pop(name,None)
    os.environ['PYTHONPATH']=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
    # 查询全部物理设备，避免可见设备变量影响 npu-smi 的展示。
    query_env = os.environ.copy()
    query_env.pop('ASCEND_RT_VISIBLE_DEVICES', None)
    info = subprocess.check_output(['npu-smi', 'info'], text=True, env=query_env)
    if f'No running processes found in NPU {a.npu}' not in info:
        raise RuntimeError(f'NPU {a.npu} 未确认空闲，请先运行 npu-smi info 检查，或用 --npu 指定空闲卡')
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(a.npu)
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.model_executor.models import ModelRegistry
    config = json.loads((Path(a.model) / 'config.json').read_text())
    archs = config.get('architectures', [])
    if not set(archs).intersection(ModelRegistry.get_supported_archs()):
        raise RuntimeError(
            f'当前 vLLM 未注册架构 {archs}。请先在当前 venv 安装模型 README 指定的 '
            'XHToken/Spark-plugin，并检查插件加载日志和 VLLM_PLUGINS 白名单。'
            '本脚本不会更改模型架构或用其他模型代替测试。')
    if a.fused_decode and getattr(ModelRegistry.models.get('Spark2_5ForCausalLM'),'module_name',None)!='spark_fused_decode':
        raise RuntimeError('融合模型未注册；请先应用 spark-fused-register.patch')
    return config


async def benchmark(a):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    # 直接传 token IDs，避免聊天模板、BOS 和重新分词改变输入长度。
    seed = tokenizer.encode('这是用于测量长上下文推理性能的文本。请继续生成内容。', add_special_tokens=False)
    seed = [t for t in seed if t not in tokenizer.all_special_ids]
    if not seed:
        raise RuntimeError('无法构造非特殊 token 输入')
    prompt = (seed * ((a.input_tokens + len(seed) - 1) // len(seed)))[:a.input_tokens]
    profiler_kwargs = {}
    if a.fused_decode:
        profiler_kwargs['worker_extension_cls']='spark_fused_decode.SparkFusedWorkerExtension'
    if a.profile_dir:
        from vllm.config import ProfilerConfig
        profiler_kwargs['profiler_config'] = ProfilerConfig(
            profiler='torch', torch_profiler_dir=str(Path(a.profile_dir).resolve()),
            torch_profiler_with_stack=False, ignore_frontend=True,
            delay_iterations=a.profile_delay, max_iterations=a.profile_steps)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=a.model, trust_remote_code=True, dtype='bfloat16',
        tensor_parallel_size=1, max_num_seqs=1,
        block_size=128,
        max_model_len=a.input_tokens + a.output_tokens,
        max_num_batched_tokens=a.max_num_batched_tokens,
        gpu_memory_utilization=a.gpu_memory_utilization,
        enable_prefix_caching=False, enable_chunked_prefill=True,
        stream_interval=1, enforce_eager=a.enforce_eager,
        speculative_config=None, **profiler_kwargs,
    ))
    async def once(request_id):
        output_tokens = (min(a.output_tokens, a.profile_output_tokens)
                         if a.profile_dir and request_id.startswith('measure-') else a.output_tokens)
        params = SamplingParams(temperature=0, max_tokens=output_tokens,
                                min_tokens=output_tokens, ignore_eos=True,
                                detokenize=False, output_kind=RequestOutputKind.DELTA)
        count = 0
        token_ids = []
        arrivals = []
        first = last = None
        start = time.perf_counter()
        async for output in engine.generate({'prompt_token_ids': prompt}, params, request_id):
            now = time.perf_counter()
            if output.prompt_token_ids is not None and len(output.prompt_token_ids) != a.input_tokens:
                raise RuntimeError('实际输入 token 数不符')
            ids = output.outputs[0].token_ids if output.outputs else []
            if ids:
                # 合并流式更新会使首 token 时间不可辨认，拒绝报告误导结果。
                if len(ids) != 1:
                    raise RuntimeError('收到合并 token 更新，无法精确测量 TTFT；请检查流式输出设置')
                count += len(ids)
                token_ids.extend(ids)
                arrivals.append(now - start)
                if first is None:
                    first = now
                last = now
        if count != output_tokens or first is None:
            raise RuntimeError(f'实际输出 {count} token，预期 {output_tokens}')
        return {'token_ids_prefix128_sha256': hashlib.sha256(json.dumps(token_ids[:128]).encode()).hexdigest(),
                'token_ids':token_ids,
                'token_ids_sha256': hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                'token_arrivals_s': arrivals,
                'input_tokens': a.input_tokens, 'output_tokens': count,
                'ttft_ms': (first - start) * 1000,
                'tpot_ms': (last - first) * 1000 / (count - 1),
                'e2e_ms': (last - start) * 1000}

    rows = []
    try:
        for i in range(a.warmup):
            print(f'预热 {i + 1}/{a.warmup}（完整长度，不计入统计）', flush=True)
            await once(f'warmup-{i}')
        for i in range(a.runs):
            if a.profile_dir:
                await engine.start_profile(f'decode-{i}')
            try:
                row = await once(f'measure-{i}')
            finally:
                if a.profile_dir:
                    await engine.stop_profile()
            rows.append(row)
            print(f'第 {i + 1} 次: TTFT={row["ttft_ms"]:.3f} ms, TPOT={row["tpot_ms"]:.3f} ms/token', flush=True)
        from importlib.metadata import version
        result = {'settings': vars(a), 'batch_size': 1,
                  'versions': {name: version(name) for name in ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'vllm-spark2_5-plugin')},
                  'profiled': bool(a.profile_dir),
                  'validation_overhead':bool(a.validation_dir or a.force_tokens_from),
                  'measurement': '引擎客户端 perf_counter 流式接收时间；不含加载、分词、预热、HTTP；含引擎调度和进程通信',
                  'runs': rows,
                  'summary': {key: {'mean': statistics.mean(r[key] for r in rows),
                                    'median': statistics.median(r[key] for r in rows)}
                              for key in ('ttft_ms', 'tpot_ms', 'e2e_ms')}}
        if a.fused_decode:
            status=await engine.collective_rpc('spark_fused_status')
            expected=a.warmup*(a.output_tokens-1)+sum(row['output_tokens']-1 for row in rows)
            if not status or any(not s['ready'] or s['layers']!=36 or s['decode_steps']!=expected for s in status):
                raise RuntimeError(f'融合 decode 执行计数不符：预期 {expected}，实际 {status}')
            result['fused_worker_status']=status
            if a.fused_graph and any(not s['graph_enabled'] or s['graph_captures']!=1 or s['graph_replays']!=expected for s in status):
                raise RuntimeError(f'融合图捕获/重放计数不符：{status}')
            library=Path(os.environ.get('SPARK_FUSED_LIBRARY','/data1/t00995472/spark-kernel-validation/build/libdecode_layer_stage_prefetch.so'))
            result['fused_kernel']={'library':str(library),'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
                                    'source_sha256':hashlib.sha256((ROOT/'kernels/decode_layer_stage_prefetch.asc').read_bytes()).hexdigest(),
                                    'capacity':a.input_tokens+a.output_tokens,'historical_kv_conversion_included':True}
            if any(s['library']!=str(library) for s in status):
                raise RuntimeError(f'融合 worker 加载库与结果不符：{status}')
            generated=library.parent/'decode_layer_stage_prefetch.asc'
            manifest=library.parent/'execution.json'
            result['fused_kernel']['baseline_source_sha256']=result['fused_kernel']['source_sha256']
            if generated.is_file():
                result['fused_kernel']['generated_source_sha256']=hashlib.sha256(generated.read_bytes()).hexdigest()
                result['fused_kernel']['source_sha256']=result['fused_kernel']['generated_source_sha256']
            result['fused_kernel']['source_path']=str(generated if generated.is_file() else ROOT/'kernels/decode_layer_stage_prefetch.asc')
            if manifest.is_file():
                result['fused_kernel']['build_manifest']=json.loads(manifest.read_text())
        target = Path(a.result)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
        print(f'结果已保存：{target}')
    finally:
        engine.shutdown()


def main():
    a = arguments()
    preflight(a)
    if a.check_only:
        print('架构已注册、设备空闲；尚未验证权重加载和完整推理。')
        return
    asyncio.run(benchmark(a))


if __name__ == '__main__':
    main()
