# Spark 7.0 ms/token 优化实验

2026-10-16，本轮目标是在现有融合 decode 基线上达到完整模型 TPOT≤7.0 ms/token。固定条件为 Ascend 950PR、BF16、不量化、TP=PP=1、单请求、32768 输入 / 4096 输出、关闭前缀缓存与推测解码。保留 `[pos-512,pos]` 的 513 位置滑窗语义，原始 `kernels/decode_layer_stage_prefetch.asc` 不变。

MLP 权重无损编码与精简向量解码已达到目标：按原连续基准流程，默认物理 **NPU 2 为 6.8202 ms/token**，NPU 4 独立复验为 **6.8212 ms/token**。保持 BF16，计算前逐位还原权重。实验记录为 `kernel_optimization/tpot7/campaign.json`，统一验收汇总为 `/data1/t00995472/spark-tpot7/result-summary.json`，原始证据均在同一数据盘目录。

## 完整模型结果

每种模式均完整预热一次，正式测量三次；输入为原基准的重复中文文本，不含 HTTP、profiler 或 logits 导出。

| 测量流程 | 物理 NPU | 模式 | TPOT 均值 / ms | TTFT 均值 / ms | E2E 均值 / s |
| --- | ---: | --- | ---: | ---: | ---: |
| 同进程 AB/BA，请求前温度≤70°C | 2 | 原融合 BF16 | 7.5551 | 1453.349 | 32.3914 |
| 同进程 AB/BA，请求前温度≤70°C | 2 | 精简无损编码 | **6.8227** | 1456.911 | 29.3958 |
| 原流程连续预热与三轮，无轮间冷却 | 2 | 精简无损编码 | **6.8202** | 1508.275 | 29.4370 |
| 原流程连续预热与三轮，无轮间冷却 | 4 | 精简无损编码 | **6.8212** | 1487.379 | 29.4203 |

NPU 2 三轮候选为 6.8191、6.8243、6.8246 ms/token，同卡观测 TPOT 下降 **9.69%**，E2E 下降 **9.25%**。全部六次测量请求的 4096 个自由生成 token 一致，每请求 4095 次 decode、每步 36 层，累计 32760 次，实际库 SHA256 校验通过。基线 decode 窗口 37 条采样全部为 1650 MHz；候选 34 条中 33 条为 1650 MHz，第一轮有一条 1350 MHz。因此不把这组宣布为严格同频性能结论，审计中的含频率门槛 `timing_target_met` 为 false；原始数据与异常采样均保留。

NPU 2 原连续流程三轮为 **6.8200、6.8225、6.8181 ms/token**，全部低于 7.0。三轮 4096-token 自由生成均与原融合基线一致，累计 16380 次 decode。三个 decode 窗口各有 14 条采样，均为 1650 MHz，温度依次为 75–82、83–88、89–93°C。第三轮 prefill 时有一条 1200 MHz 采样。该复验期间其他实验卡已释放，没有请求间冷却。

NPU 4 原连续流程三轮为 6.8177、6.8222、6.8238 ms/token，也全部低于 7.0。三轮 token、16380 次 decode 和库身份均通过；三个 decode 窗口各 14 条采样，均为 1650 MHz，温度依次为 69–75、77–82、83–87°C。第二轮 prefill 时有一条 1250 MHz 采样。该卡只用于独立绝对 TPOT 复验，不跨卡计算加速比。

AB/BA 的时间窗口由请求内直接记录的 UTC 和流式到达时间划定；两组连续基准的窗口由完成日志接收时间减去请求耗时估计，未消除日志传输延迟。NPU 2 交错对照与 NPU 4 连续测量部分时间重叠，不能排除共享主机资源影响。连续四个完整请求的验证不代表无限时长服务的热稳定性。

对应原始性能与审计：`lossless-fast/paired-32k-cooled.json`、`paired-32k-cooled-audit.json`、`continuous-npu2.json`、`continuous-npu2-audit.json`、`continuous-npu4.json`、`continuous-npu4-audit.json`。默认融合库未替换，使用下文的 `--fused-library` 显式启用候选。

## 方法及适用边界

已有 profile 显示 MLP 权重读取活动占主要时间。单纯改 tile、访问次序或同步结构，未减少主要读取量。本轮将原 BF16 权重按 tile 可逆编码：每值保存 8 位符号/尾数和 4 位指数差，指数超出 109..124 的值使用完整 16 位原值修补。计算前在 UB 逐位恢复 BF16，再执行原 GEMV。不存在舍入或截断，计算 dtype、累加顺序和激活边界保持不变。

全部 36 层的 72 个 MLP 矩阵最大每 tile 异常数为 234，因此统一预留 256 个修补项。超过容量会拒绝编码。32/64 修补槽的早期探针不能用于本模型。MLP 存储字节从 5,662,310,400 降至 4,368,384,000，减少 **22.85%**；这只是编码数据量，不是模型总显存减少量。

初版解码增加了较多整数指令。精简版用 128 个 uint16 lane 并行恢复符号/尾数，合并指数运算。单层深 profile 中，Vector 活动由初版约 105.7 降至 79.3 μs，MTE2 约 116 μs；流水线活动有重叠，不能相加解释耗时，也不作为硬件带宽上限。

候选仍为融合 eager：QKV/O、attention、prefill、LM head 和采样沿用原实现；workspace 仍为 128 KiB。适配器根据同目录 `execution.json` 的 `bf16_lossless12` 标识打包 MLP 参数 4/5，普通 BF16 库保持原布局。库切换要求请求已完成，拒绝已加载路径的库字节变化，保留所有已加载库和权重的引用。

当前实现为支持对照切换保留原始重排权重，额外持有约 **4.07 GiB** 编码权重；首次 CPU 打包也有初始化开销，均不计入稳态 TPOT。此版本未开放 `--fused-graph`，不应直接将编码库交给只支持原 packing 的独立调用器。其余限制沿用 [融合接入说明](FUSED_DECODE.md)。

## 精度证据

- 全部 72 个实际 MLP 矩阵在 CPU 编码、解码后按原始 int16 逐位相等，SHA256 一致。
- NPU 独立解码覆盖全部 65536 个 BF16 原始码点，包含 NaN、无穷、正负零；两种 tile 各覆盖异常数 0..256、首尾和向量边界、输入不变、双侧 guard、10 轮复用。
- 精简版通过 layer 0/32769、layer 3/32769、layer 11/40960 的单层 CPU 分阶段参考和重复执行检查。最后一组包含全模型最大 234 个修补项的权重 tile；输出及残差与原库逐位相等。
- 完整模型使用 128、32768、35839、40896 输入各生成 64 步同前缀 logits，共 256 步（252 次 decode）与原融合库逐位一致，最大绝对误差与 NRMSE 均为 0。强制 token 仅用于固定数值比较前缀，不能当作自由生成一致性的证据；自由生成在独立 32K/4K 性能运行检查。

独立数值证据见 `mlp_pipeline/codec-all-model.json`、`codec256-fast-npu.json`，单层结果见 `weight_path/summary.json`。长上下文结果在 `validation-lossless-fast/`。

## 其余方向与负结果

| 方向 | 观测结果 | 决策 |
| --- | --- | --- |
| 七分片按 384 对齐 | 完整 32K/4K：7.5521→7.5396 ms，下降 0.165% | 保留独立候选，不与无损编码混合 |
| 两遍 attention、Q 复用 | 九套 KV 轮换：263.183→262.749 μs | 收益过小，不接入 |
| QKV 期间预取首块历史 K | 九套 KV 轮换：263.291→262.876 μs | 收益过小，不接入 |
| MLP 局部生产并立即 down | 164.210→164.228 μs | 持平，不接入 |
| 权重 L2 策略 | 固定单层约 0.8% 表观收益，12 套权重轮换后慢 0.26% | 不接入 |
| 循环权重分工 | 完整融合层慢约 0.6%–0.8% | 不接入 |

固定单层重复命中缓存会高估收益：全局层单套 KV 约 229 μs，九套轮换约 263 μs，后者接近已有完整模型 profile。单层微测用于筛选，不能直接外推 TPOT。

初版无损编码未经冷却的完整三轮为 7.0877、7.0847、8.3868 ms/token，实际均值 **7.5198**，对应基线 7.5482。第三轮温度 97–99°C，采到 650–1550 MHz 的低频，前两轮采样均为 1650 MHz；该完整结果保留，不能只取前两轮宣布收益。精简版的配对复测对两种实现都采用请求开始前温度≤70°C 的相同条件，每次 4096-token 请求内部连续执行。冷却时间不计入请求 TPOT，结果与持续无间隔服务吞吐区分。

各分支源码、复现命令及负结果索引：

- [attention 实验](../kernel_optimization/tpot7/attention_pipeline/README.md)
- [MLP 数据流与独立编码验证](../kernel_optimization/tpot7/mlp_pipeline/README.md)
- [权重布局、无损编码与构建](../kernel_optimization/tpot7/weight_path/README.md)

## 复现与审计

先确认 NPU 2 空闲。以下使用已构建且不可变的精简库，SHA256 为 `d33b8dc1a428a71afa4cd33afb492bd720db6dbaf7c4404c7f161e7c97d40243`。必须保留同目录的 `execution.json`，不能只复制 `.so` 后丢弃编码标识。重新构建应使用新目录，不能覆盖运行中或已加载的 `.so`。

```bash
source /data1/t00995472/spark/activate_ascend.sh
npu-smi info

# 同进程完整请求：双方各预热一次，AB / BA / AB 各三轮
python -B kernel_optimization/tpot7/paired_model.py --npu 2 \
  --candidate /data1/t00995472/spark-tpot7/weight_path/lossless_fast/libdecode_layer_stage_prefetch.so \
  --max-start-temp 70 \
  --result /data1/t00995472/spark-tpot7/new-paired.json

python -B kernel_optimization/tpot7/audit_pair.py \
  /data1/t00995472/spark-tpot7/new-paired.json \
  --output /data1/t00995472/spark-tpot7/new-paired-audit.json

# 普通基准入口按 manifest 自动选择编码，不需要新安装包
python -B benchmark_spark_vllm.py --npu 2 --enforce-eager --fused-decode \
  --fused-library /data1/t00995472/spark-tpot7/weight_path/lossless_fast/libdecode_layer_stage_prefetch.so \
  --result /data1/t00995472/spark-tpot7/new-fast.json
```

配对脚本保存每 token 到达时间、全部自由生成 token、实际库 SHA256、每请求 decode 计数和设备采样；打包、库切换、冷却、logits 导出与 profiler 都不进入请求计时。每个请求必须执行 4095 次 decode、每次 36 层，总计 32760 步。`complete` 仅代表采集完成，`validation_passed` 与 `timing_target_met` 分别表示 token 验证和本基准计时门槛；最终验收还需要上述独立数值证据。

`audit_pair.py` 从原始到达时间复算 TPOT，验证 token 与库哈希、执行计数，并只在首个到最后一个流式 token 的时间窗口统计频率。稀疏采样不能证明每个时刻都同频；本实验不作严格 roofline 或机器峰值结论。
