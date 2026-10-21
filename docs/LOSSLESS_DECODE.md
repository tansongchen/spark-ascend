# BF16 无损压缩：减少 MLP 权重读取

本方案在 BF16 融合基线 `stage_prefetch` 上压缩 MLP 权重的存储表示，计算前逐位恢复全部 16 位，再执行原 GEMV。计算 dtype、累加顺序和 BF16 舍入边界保持不变。交付实现为 `lossless_prefetch`，完整模型历史 TPOT 约 **6.82 ms/token**。

构建与模型运行见 [README](../README.md)。

## 为什么选择无损编码

阶段分析表明 MLP 权重读取活动占主要时间。调整 tile、访问顺序、L2 策略或局部 up→down 数据流没有稳定收益，因此转向减少搬运字节。普通量化会改变权重，本方案利用权重指数集中这一特征，缩短常见值的编码，对异常值保留完整原始位。

QKV/O、attention、prefill、LM head 和采样沿用 BF16 融合基线。优化范围仅为 MLP gate/up 合并矩阵 W1 与 down 矩阵 W2 的存储和解码。

## 编码与解码

BF16 包含 1 位符号、8 位指数和 7 位尾数。先按 GEMV tile 重排权重，每个值保存：

1. 完整的符号和尾数，共 8 位。
2. 相对 base=109 的 4 位指数差，覆盖指数 109..124。
3. 指数超出窗口时，在修补表中保存元素索引及原始 16 位值。

每个 tile 的布局为 `32 字节头 + E 字节符号/尾数 + E/2 字节指数差 + 256×4 字节修补表`，其中 E 是元素数。全模型最大每 tile 异常数为 234，故固定预留 256 项；超出容量直接拒绝编码。NaN、无穷、正负零等位模式也按原始位恢复。

| MLP tile | 元素数 | 原始字节 | 编码字节 | 减少 |
| --- | ---: | ---: | ---: | ---: |
| W1，K=2560、M=8 | 20480 | 40960 | 31776 | 22.42% |
| W2，K=10240、M=4 | 40960 | 81920 | 62496 | 23.71% |

全部 72 个 MLP 矩阵的编码数据从 5,662,310,400 降至 4,368,384,000 字节，减少 **22.85%**。这只衡量编码数据量。

设备端在 UB 中还原权重。最终解码器用 128 个 uint16 lane 并行恢复符号/尾数，合并指数运算，减少初版的整数指令开销，同时保留原来的跨阶段预取。独立单层 profile 中，精简解码将 Vector 活动约 105.7→79.3 μs，MTE2 约保持 116 μs；这些流水活动可重叠，不能直接相加解释总耗时。

## 实现对应关系

| 文件 | 职责 |
| --- | --- |
| [decode_layer_lossless_prefetch.asc](../kernels/decode_layer_lossless_prefetch.asc) | 融合层及独立 `launch_decode_tile` 校验入口 |
| [decode_bf16_lossless_vf.h](../kernels/decode_bf16_lossless_vf.h) | 精简向量解码 |
| [bf16_codec.py](../utils/bf16_codec.py) | CPU 编解码、头部与修补容量检查 |
| [packing.py](../utils/packing.py) | 原 BF16 tile 重排 |
| [spark_fused_decode.py](../utils/spark_fused_decode.py) | 按 manifest 打包和加载编码权重 |

ABI 仍是 `launch_decode_layer_stage_prefetch`，但参数数组 `a[4]`、`a[5]` 必须指向编码权重。`execution.json` 使用 `weight_encoding=bf16_lossless12`、`codec_base=109`、`codec_repairs=256` 标记布局；不能只替换共享库而继续传原 BF16 packing。workspace 仍为 128 KiB。

## 结果与验收

2026-10-16，Ascend 950PR、32768 输入 / 4096 输出、TP=PP=1、单请求、融合 eager，完整预热一次后测三次：

| 流程 | 原融合 BF16 | 无损版 |
| --- | ---: | ---: |
| NPU 2，同引擎 AB/BA，请求前温度≤70°C | 7.5551 ms/token | 6.8227 ms/token |
| NPU 2，连续流程，无轮间冷却 | — | 6.8202 ms/token |
| NPU 4，连续流程，无轮间冷却 | — | 6.8212 ms/token |

同卡配对观测 TPOT 下降 9.69%，但候选采到一条低频，不作严格同频结论。连续复验的 decode 频率样本均为 1650 MHz，稀疏采样不证明全程恒频，也不代表无限时长服务的热稳定性。

已完成的数值证据：

- 全部 72 个真实 MLP 矩阵在 CPU 编解码后逐位一致。
- NPU 解码覆盖全部 65536 个 BF16 码点、异常数 0..256、边界索引、保护区及重复复用。
- 滑窗、全局和含最大异常数权重的单层检查通过，最终输出与残差相对原融合库逐位一致。
- 128、32768、35839、40896 输入各 64 步同前缀 logits 与原融合库逐位一致；独立长请求运行的每轮 4096 个自由生成 token 也一致。

同前缀 logits 校验使用强制 token 固定上下文，自由生成一致性由独立运行验证。当前适配器保留原始重排权重，额外持有约 **4.07 GiB** 编码权重；CPU 打包是初始化开销。本版只支持融合 eager，其他约束见 README。

## 验证工具与证据

激活环境后，从仓库根目录运行；以下 `SPARK_BUILD` 和 `SPARK_RESULTS` 沿用 README 示例：

```bash
# 全码点 CPU/NPU 解码检查；省略 --library 时只检查 CPU
python -B utils/validate_bf16_codec.py --npu 2 \
  --library "$SPARK_BUILD/lossless/libdecode_layer_stage_prefetch.so" \
  --output "$SPARK_RESULTS/codec.json"

# 全部真实 MLP 权重；--layers 0 可做单层抽查
python -B utils/validate_bf16_model_codec.py \
  --output "$SPARK_RESULTS/model-codec.json"

# 完整模型 AB/BA 对照
python -B utils/benchmark_paired_model.py --npu 2 \
  --baseline "$SPARK_BUILD/stage/libdecode_layer_stage_prefetch.so" \
  --candidate "$SPARK_BUILD/lossless/libdecode_layer_stage_prefetch.so" \
  --result "$SPARK_RESULTS/paired.json"

python -B utils/audit_paired_model.py "$SPARK_RESULTS/paired.json" \
  --output "$SPARK_RESULTS/paired-audit.json"
```

配对脚本默认不冷却；复现表中温控配对流程需加 `--max-start-temp 70`。修补数量与权重分布分析分别用 `utils/analyze_bf16_repairs.py`、`utils/analyze_bf16_weight_stats.py`，参数见 `--help`。

原始证据根目录为 `/data1/t00995472/spark-tpot7/`：`result-summary.json` 汇总验收，`lossless-fast/` 保存长请求与配对结果，`validation-lossless-fast/` 保存 logits，`mlp_pipeline/` 保存编解码验收。
