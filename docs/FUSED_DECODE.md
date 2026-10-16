# Spark 融合 decode 实测

2026-10-16，将未修改的 `decode_layer_stage_prefetch.asc` 接入 Spark-X2.5-4B。两轮实验均使用物理 NPU 2、BF16、TP=1、batch=1，32768 输入 / 4096 输出；每种模式完整预热一次，再测三次。下表均为三次均值，保留两轮各自的融合 eager 对照。

| 实验 | 模式 | TPOT（ms/token） | TTFT（ms） | E2E（秒） |
| --- | --- | ---: | ---: | ---: |
| 第一轮：融合加速 | 原 eager | 12.2099 | 1485.4 | 51.485 |
| 第一轮：融合加速 | 原 vLLM 默认图 | 11.0508 | 1613.4 | 46.866 |
| 第一轮：融合加速 | 融合 eager | **7.5557** | 1529.8 | 32.470 |
| 第二轮：融合图对照 | 融合 eager（复测） | 7.5566 | 1480.7 | 32.425 |
| 第二轮：融合图对照 | 融合内部图 | **7.5431** | 1543.8 | 32.433 |

第一轮融合 eager 相对原 vLLM 图执行，生成速度提升 **1.463 倍**，TPOT 降低 **31.63%**，E2E 缩短 **30.72%**；相对原 eager，TPOT 降低 **38.12%**。第二轮加入内部图后，TPOT 仅下降 **0.18%**，E2E 基本持平，因此默认保留融合 eager，内部图作为可选开关。

后续保持 BF16 的优化中，MLP tile/DMA 和 LM head 重写均未确认有值得接入的收益。七分片 attention 在 32K/4K 下未加速；在 NPU 4 的同卡 36K/4K 对照中，TPOT 从 7.7714 降至 7.6646 ms/token，观测下降 **1.37%**，E2E 下降 **1.11%**，各三轮输出 token 均一致。该库通过 `--fused-library` 可选启用，详细结果及使用方法见 [BF16 优化记录](BF16_OPTIMIZATION.md)。

计时包含引擎调度、进程通信和首个 decode 步的历史 KV 转换，排除加载、分词、预热及 HTTP；profiler 和 logits 校验不参与性能统计。两轮均为同卡顺序测量，未随机交错，部分设备采样存在低频，结果是本轮观测值。输入为重复中文文本，不代表所有业务负载。

**接入与适用范围**

`integration/spark_fused_decode.py` 通过已应用的 `patches/spark-fused-register.patch` 按开关注册，未启用时仍使用原模型。初始化时重排真实权重；prefill 沿用原实现，每个请求首次 decode 将历史 KV 转入私有分页 cache，随后每 token 执行 36 个融合层。LM head 和采样仍由 vLLM 完成。

`--fused-graph` 将这 36 层捕获为一个 NPU 图，固定缓冲区地址，每步更新输入与 position/slot 后重放。**外层 vLLM 仍需 `--enforce-eager`**；prefill、embedding、首层输入 RMSNorm、LM head 和采样位于内部图外。

- 支持 BF16、TP=PP=1、单请求顺序执行、block size=128、总上下文≤40960；关闭前缀缓存和推测解码，暂不支持并发多请求、量化或 KV 传输。
- 保留原权重，另持有重排权重及私有 cache，额外占用约 **13.38 GiB**。显存利用率参数为 0.7 时，框架可用 KV 池由 62.87 GiB 降至 49.49 GiB。
- 算子与本机 vLLM Ascend/FIA 的滑窗均包含 `[pos-512,pos]`，最多 513 个位置；与本地 Transformers 的 512 位置窗口有差异，见 [算子验证说明](../kernel_validation/README.md)。

**正确性与执行核验**

两轮所有模式、每轮请求的 4096 个输出 token 均一致。另对 128 输入 / 64 输出请求固定相同前缀，63 个 decode 步相对原模型的 logits top-1 一致率为 100%，最大 NRMSE 为 0.00515122；融合图与融合 eager 导出的 64 组 logits 逐元素完全一致。这些验证不替代广泛的模型质量评测。

融合图长请求共一次捕获、16,380 次重放，对应一次预热和三次测量。第一轮独立 profiler 轨迹核验了 16 步、每步 36 个融合 kernel；融合层占设备 kernel 时间之和约 92%，与图执行额外收益很小的现象相符。完整轨迹及采集后 RPC 报错记录保留在第一轮证据目录，不作为性能计时结果。

**运行方法**

先确认物理 NPU 2 空闲，禁止使用物理 NPU 0。本机已应用注册补丁并构建共享库；需要重建时运行 `bash kernel_validation/build.sh`，产物位于数据盘。

```bash
source /data1/t00995472/spark/activate_ascend.sh
npu-smi info

# 融合 eager；加入 --fused-graph 即启用融合内部图
python -B benchmark_spark_vllm.py --npu 2 --enforce-eager --fused-decode \
  --result /data1/t00995472/spark-fused-benchmark/new-fused.json

# 复现第一轮：原 eager、融合 eager、原 vLLM 图
python -B benchmarks/run_spark_fused_benchmarks.py \
  --directory /data1/t00995472/spark-fused-benchmark/new-performance
python -B benchmarks/analyze_spark_fused.py /data1/t00995472/spark-fused-benchmark/new-performance

# 复现第二轮：融合 eager、融合内部图
python -B benchmarks/run_spark_fused_benchmarks.py --fused-graph-comparison \
  --directory /data1/t00995472/spark-fused-graph/new-performance
python -B benchmarks/analyze_spark_fused.py --fused-graph-comparison \
  /data1/t00995472/spark-fused-graph/new-performance
```

每次复现使用新结果目录。原始逐次结果、token、日志、频率/温度采样及数值校验均保留在数据盘：

- 第一轮：`/data1/t00995472/spark-fused-benchmark/`，性能汇总为 `performance/comparison.json`。
- 第二轮：`/data1/t00995472/spark-fused-graph/`，性能汇总为 `performance/comparison.json`，总体验证为 `result-summary.json`。
