"""AB/BA complete requests in one engine; compatible fused libraries only."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def identity(path):
    path = Path(path).resolve(strict=True)
    result = {'library': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    names = {'execution.json', 'identity.json'}
    manifest = path.parent / 'execution.json'
    if manifest.exists():
        names.update(json.loads(manifest.read_text()).get('source_sha256', {}))
    for name in sorted(names):
        sibling = path.parent / name
        if sibling.exists():
            result[name] = (json.loads(sibling.read_text()) if name.endswith('.json')
                            else hashlib.sha256(sibling.read_bytes()).hexdigest())
    return result


async def run(a):
    from benchmark import preflight
    envargs = SimpleNamespace(model=a.model, npu=a.npu, fused_decode=True,
                              fused_graph=False, enforce_eager=True, input_tokens=a.input_tokens,
                              output_tokens=a.output_tokens, fused_library=a.baseline,
                              validation_dir=None, force_tokens_from=None)
    preflight(envargs)
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    seed = tokenizer.encode('这是用于测量长上下文推理性能的文本。请继续生成内容。', add_special_tokens=False)
    seed = [t for t in seed if t not in tokenizer.all_special_ids]
    prompt = (seed * ((a.input_tokens + len(seed) - 1)//len(seed)))[:a.input_tokens]
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=a.model, trust_remote_code=True, dtype='bfloat16', tensor_parallel_size=1,
        max_num_seqs=1, block_size=128, max_model_len=a.input_tokens+a.output_tokens,
        max_num_batched_tokens=32768, gpu_memory_utilization=.7,
        enable_prefix_caching=False, enable_chunked_prefill=True, stream_interval=1,
        enforce_eager=True, speculative_config=None,
        worker_extension_cls='utils.spark_fused_decode.SparkFusedWorkerExtension'))

    async def request(label, number):
        selected = await engine.collective_rpc('spark_fused_select_library',
                                               args=(getattr(a, label),))
        if len(selected) != 1 or selected[0]['sha256'] != artifacts[label]['sha256']:
            raise RuntimeError(f'Worker selected unexpected library: {selected}')
        cooldown_start = time.monotonic()
        start_temperature = None
        if a.max_start_temp is not None:
            while True:
                query = await asyncio.to_thread(subprocess.run,
                    ['npu-smi','info','-t','common','-i',str(a.npu)], capture_output=True,text=True,check=True)
                fields = dict(line.split(':',1) for line in query.stdout.splitlines() if ':' in line)
                values = {key.strip():value.strip() for key,value in fields.items()}
                start_temperature = float(values['Temperature(C)'])
                if start_temperature <= a.max_start_temp:
                    break
                if time.monotonic()-cooldown_start > 600:
                    raise RuntimeError('Device did not cool to the configured request-start temperature')
                print(json.dumps({'waiting_for_cooldown':label,'request':number,
                                  'temperature_C':start_temperature,'limit_C':a.max_start_temp}),flush=True)
                await asyncio.sleep(5)
        cooldown_seconds = time.monotonic()-cooldown_start
        tokens, arrivals = [], []
        params = SamplingParams(temperature=0, max_tokens=a.output_tokens,
                                min_tokens=a.output_tokens, ignore_eos=True,
                                detokenize=False, output_kind=RequestOutputKind.DELTA)
        utc_start = time.time()
        start = time.perf_counter()
        async for output in engine.generate({'prompt_token_ids':prompt}, params, f'{label}-{number}'):
            now = time.perf_counter()
            ids = output.outputs[0].token_ids if output.outputs else []
            if ids:
                if len(ids) != 1:
                    raise RuntimeError('Merged token arrival is not valid latency evidence')
                tokens.extend(ids)
                arrivals.append(now-start)
        if len(tokens) != a.output_tokens:
            raise RuntimeError('Incomplete request')
        utc_end = time.time()
        finished = await engine.collective_rpc('spark_fused_status')
        if (len(finished) != 1 or finished[0]['decode_steps'] - selected[0]['decode_steps'] != a.output_tokens-1
                or finished[0]['library_sha256'] != artifacts[label]['sha256']):
            raise RuntimeError(f'Unexpected per-request execution: {selected} -> {finished}')
        row = {'mode':label, 'request':number, 'utc_start':utc_start, 'utc_end':utc_end,
               'input_tokens':a.input_tokens, 'output_tokens':len(tokens),
               'tpot_ms':1000*(arrivals[-1]-arrivals[0])/(len(tokens)-1),
               'ttft_ms':1000*arrivals[0], 'e2e_ms':1000*arrivals[-1],
               'token_ids':tokens, 'token_arrivals_s':arrivals,
               'token_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
               'worker_selection':selected, 'worker_finished':finished}
        row.update(cooldown_seconds=cooldown_seconds, start_temperature_C=start_temperature)
        print(json.dumps({k:v for k,v in row.items() if k not in ('token_ids','token_arrivals_s')}), flush=True)
        return row

    artifacts = {key:identity(getattr(a,key)) for key in ('baseline','candidate')}
    rows, warmups = [], []
    result = {'settings':vars(a), 'libraries':artifacts, 'runs':rows, 'warmups':warmups,
              'profiled':False, 'validation_overhead':False,
              'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path(__file__).resolve(), ROOT/'utils/spark_fused_decode.py',
                            ROOT/'utils/bf16_codec.py')},
              'measurement':'Client streamed arrival times; complete requests interleaved in one engine; excludes library switch RPC'}
    target = Path(a.result)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        for i in range(a.warmup):
            for label in ('baseline','candidate'):
                row = await request(label, f'warmup-{i}')
                warmups.append({k:v for k,v in row.items() if k not in ('token_ids','token_arrivals_s')})
        for i in range(a.rounds):
            for label in (('baseline','candidate') if i%2 == 0 else ('candidate','baseline')):
                rows.append(await request(label, i))
                target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        status = await engine.collective_rpc('spark_fused_status')
        expected = 2*(a.warmup+a.rounds)*(a.output_tokens-1)
        if len(status)!=1 or not status[0]['ready'] or status[0]['layers']!=36 or status[0]['decode_steps']!=expected:
            raise RuntimeError(f'Unexpected execution count: {status}; expected {expected}')
        result['worker_status'] = status
        result['all_output_tokens_equal'] = all(r['token_ids']==rows[0]['token_ids'] for r in rows)
        result['summary'] = {label:{key:statistics.mean(r[key] for r in rows if r['mode']==label)
                                   for key in ('tpot_ms','ttft_ms','e2e_ms')}
                             for label in ('baseline','candidate')}
        result['tpot_reduction_percent'] = 100*(1-result['summary']['candidate']['tpot_ms']/result['summary']['baseline']['tpot_ms'])
        result['complete'] = True
        result['validation_passed'] = result['all_output_tokens_equal']
        result['timing_target_met'] = (result['validation_passed']
                                     and a.input_tokens == 32768 and a.output_tokens == 4096
                                     and a.rounds >= 3
                                     and result['summary']['candidate']['tpot_ms'] <= 7.0)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        print('SUMMARY', json.dumps({k:v for k,v in result.items() if k in ('summary','all_output_tokens_equal','tpot_reduction_percent','worker_status')}), flush=True)
        if not result['validation_passed']:
            raise RuntimeError('Measurement finished but output-token validation failed')
    finally:
        engine.shutdown()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='/data1/t00995472/spark-kernel-validation/build/libdecode_layer_stage_prefetch.so')
    p.add_argument('--candidate', required=True)
    p.add_argument('--model', required=True, help='显式 BF16 checkpoint')
    p.add_argument('--npu', type=int, default=2)
    p.add_argument('--input-tokens', type=int, default=32768)
    p.add_argument('--output-tokens', type=int, default=4096)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--max-start-temp', type=float, help='Optional identical request-start cooling condition; outside timing')
    p.add_argument('--result', required=True)
    a=p.parse_args()
    if a.npu not in range(1,8) or a.input_tokens<1 or a.input_tokens+a.output_tokens>40960 or a.output_tokens<2 or a.rounds<1 or a.warmup<1:
        p.error('Require NPU1..7, context<=40960, output>=2, complete warmup and positive rounds')
    if a.max_start_temp is not None and not 40 <= a.max_start_temp <= 85:
        p.error('Request-start temperature must be between 40 and 85 C')
    if Path(a.result).exists():
        p.error('Use a new output path; existing evidence is immutable')
    a.baseline=str(Path(a.baseline).resolve(strict=True))
    a.candidate=str(Path(a.candidate).resolve(strict=True))
    os.environ.pop('SPARK_LOGITS_DIR',None)
    os.environ.pop('SPARK_FORCE_TOKENS',None)
    os.environ.pop('ASCEND_LAUNCH_BLOCKING',None)
    stop=threading.Event()
    target=Path(a.result)
    target.parent.mkdir(parents=True, exist_ok=True)
    def sample():
        with target.with_suffix('.health.jsonl').open('w') as f:
            while not stop.is_set():
                q=subprocess.run(['npu-smi','info','-t','common','-i',str(a.npu)], capture_output=True,text=True)
                f.write(json.dumps({'utc':time.time(),'returncode':q.returncode,'common':q.stdout,'stderr':q.stderr})+'\n')
                f.flush()
                stop.wait(2)
    thread=threading.Thread(target=sample,daemon=True)
    thread.start()
    try:
        asyncio.run(run(a))
    finally:
        stop.set()
        thread.join(timeout=10)


if __name__=='__main__':
    main()
