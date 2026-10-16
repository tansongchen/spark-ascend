# Spark BF16 融合 decode 优化记录

## 约束与基线

- 用户授权保持 BF16 优化，禁止权重量化；允许在不同空闲 NPU 并行实验，禁止 NPU 0。
- 原始 `decode_layer_stage_prefetch.asc` 保持不变，候选位于 `kernel_optimization/`；构建、日志、缓存和结果放 `/data1/t00995472/spark-bf16-opt/`。
- 基线：物理 NPU 2，BF16、TP=1、batch=1、32768 输入 / 4096 输出，完整预热一次、测三次。融合 eager 7.5566 ms/token，融合内部图 7.5431 ms/token。
- 现有完整融合采集：每步 36 个融合 kernel 合计 6.7933 ms，LM head 0.4906 ms，设备 kernel 合计 7.3821 ms。滑窗融合 AIV MTE2/AIV 约 90.4%；这不是 HBM 带宽饱和的证明。
- 单层准确性沿用独立 CPU FP32/BF16 边界参考：NRMSE≤0.03、cosine≥0.999，并记录中间输出和 cache/guard。规约变化需完整模型同前缀 logits 复验。
- 微测用于筛选，最终收益以 NPU 2 同卡完整模型结果为准；不将重复单层缓存命中结果直接作为模型加速。

## 第一轮分工

| 负责人 | 物理 NPU | 方向 | 状态 |
| --- | ---: | --- | --- |
| 主线程 | 2、4 | 阶段计时、attention 独立集成与模型复验 | 完成；32K 无加速，36K TPOT 观测下降 1.37% |
| gemv | 4 | GEMV/MLP tile、权重预取与访问 | 已完成，未确证收益，设备已释放 |
| attention | 5，随后 2 | 全局 attention 分片与 KV 流水线 | NPU5 原始与候选均等待；NPU2 完整融合验证通过 |
| lm_head | 6 | BF16 单 token LM head | 已完成，无值得接入收益，设备已释放 |

所有参与者使用独立子目录和设备，不改安装包或全局运行环境。设备最初均确认空闲。以 `cake3`、`bisheng` 和 `msprof` 的本机接口与原始采集作为分析依据，不把技能中未验证的能力或其他工具链的结果作为本机性能证据。

## 验证流程

1. 用阶段计时定位开销，记录插桩扰动；微测对照 baseline/candidate/null，保留负结果。
2. 候选通过真实权重单层参考及重复执行检查；包含滑窗/全局层和分页边界。
3. 幸存候选由主线程独立集成，先校验短请求 token/logits，再测 32K/4K 完整请求。
4. 同卡比较，保留设备频率、温度、执行计数和来源哈希；正式计时排除 profiler/logits 导出。
5. 没有独立验证收益的候选不设为默认。阶段测试与最终模型结果分开记录。

## 阶段诊断与候选筛选

`kernel_optimization/diagnostics/stages/` 生成原始源码的插桩副本，在未使用的 workspace 区域记录每个 AIV 的 8 个 `GetSystemCycle` 时间戳。保持原有数学和 barrier；以下是 NPU 2、长度 32769、固定输入下 core0 的相邻时间戳占比，包含阶段同步等待，用于定位方向，不作为生产 TPOT 或独立阶段基准。末阶段时间戳也不能替代完整设备完成事件。

| 阶段 | 滑窗融合层 | 全局融合层 |
| --- | ---: | ---: |
| QKV / RoPE / cache / 同步 | 15.1% | 10.8% |
| attention / 同步 | 2.3% | 30.3% |
| O projection / 同步 | 10.1% | 7.0% |
| up / GELU / 同步 | 49.3% | 35.5% |
| down / 同步 | 23.1% | 16.0% |

插桩版真实权重参考与 guard 均通过。另做同进程交错 baseline/null/插桩对照：滑窗基线 164.311 μs，插桩版 164.657 μs，观测增加 0.21%；null 为 164.306 μs。原始时间戳及验证在数据盘 `profile/`。

- **GEMV**：M16 up 和 M2 down 旧微测为负；旧计时未显式刷新起始事件，保留但不作严格加速证据。修正后测试 `up_pair`，保持 M8 计算与每核原始任务，仅合并相邻 tile DMA，滑窗 163.841→163.790 μs（0.031%）、全局 229.823→229.564 μs（0.113%），未确证有实际收益，不集成。中间/最终输出逐位一致，分页边界、CPU 参考、guard 通过。见 `kernel_optimization/rejected/gemv/README.md`。
- **LM head**：持续负载交错复验，torch 490.529 μs、自定义最佳 489.216 μs（0.27%），对全模型的简单估算约 0.02%，不集成。GM→UB 探针相同 671 MB 数据最快 488.804 μs、1.3729 TB/s；仅代表此访问路径的实测速率，不是硬件峰值或最优证明。见 `kernel_optimization/rejected/lm_head/README.md`。
- **Attention**：`balanced7_all` 将全局 attention 的 width 改为 `ceil(length/(7×128))×128`，对应 7×4 个 KV-head 任务；保留 8 个 partial slot 和原 ABI/权重布局，第 8 个分片无效。SWA 不变；候选只改变分片与归约边界，保持 BF16。

NPU 2 同进程 baseline/null/候选交错微测，真实权重验证、重复 3 次及 guard 均通过：

| 全局层上下文 | 原融合 / μs | 七分片 / μs | 观测耗时下降 |
| ---: | ---: | ---: | ---: |
| 129 | 166.161 | 166.278 | -0.071% |
| 32769 | 229.664 | 229.459 | 0.089% |
| 36864 | 255.324 | 250.799 | 1.772% |
| 40960 | 292.084 | 270.949 | 7.236% |

以上重复单层使用同一分配，不代表完整模型收益；null 差异绝对值≤0.052%。完整模型 128 输入 / 64 输出同前缀校验已通过：63 个 decode 步 logits top-1 一致率 100%、最大 NRMSE 0.00648655、最小 cosine 0.99998325。七分片改变规约，logits 不要求与原路径逐位相同。

完整模型基准使用显式 `--fused-library`，记录 worker 实际加载路径、库/源码哈希和构建 manifest。32K/4K 在 NPU 2、36K/4K 在 NPU 4，各组内部同卡原融合对照，均完整预热一次测三次；两个设备并行运行，结果不跨卡相减。

## 完整模型最终结果

| 输入 / 输出 | 物理 NPU | 原融合 TPOT | 七分片 TPOT | TPOT 下降 | 原融合 E2E | 七分片 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 / 4096 | 2 | 7.5524 ms | 7.5605 ms | -0.108% | 32.406 s | 32.491 s |
| 36864 / 4096 | 4 | 7.7714 ms | 7.6646 ms | **1.374%** | 33.542 s | 33.170 s |

36K 输入下 E2E 观测下降 **1.110%**。32K 原融合/候选 TTFT 均值为 1479.373 / 1530.771 ms，36K 为 1718.365 / 1783.400 ms；prefill 使用相同原路径，本轮没有 TTFT 收益。

两组各三轮、每轮 4096 个输出 token 均逐个一致；各进程均正常退出，每个 worker 校验 ready=true、layers=36、decode_steps=16380。源码与共享库哈希、实际加载路径、逐 token 时间及执行命令均保存。当前候选作为可选库保留，不替换默认融合实现；当前 32K/4K 工作负载未取得额外加速，较长上下文的收益也仅为本轮观测值。

频率/温度采样覆盖整个进程，包含加载与预热。按利用率≥50%筛选，32K 原融合/候选分别有 53/53 个活跃样本，其中 3/2 个低于 1650 MHz；36K 为 55/53 个，其中 1/2 个低于 1650 MHz。对应全进程温度范围分别为 54–87 / 75–94°C、54–84 / 74–91°C。本轮非随机交错、非严格同频校准，两张卡并行测量也可能有共享主机资源影响。

原始证据均在 `/data1/t00995472/spark-bf16-opt/`：

- `profile/`：阶段插桩、CPU 参考与扰动对照。
- `gemv/`、`lm_head/`：各分支候选、正负结果、原始采集和测量修正记录。
- `attention/balanced7_all/`：生成源码、库、四个上下文边界验证及 paired/null 微测。
- `model/performance-32k/`、`model/performance-36k/`：完整请求各三次结果、`comparison.json`、执行日志、频率/温度采样。
- `model/logit-comparison.json`：同前缀数值校验；`result-summary.json` 为总体索引。

### 使用候选

下面示例使用 36K 输入，运行前检查 NPU 2 空闲。默认融合库仍适用于原 32K/4K 基准；本轮无需安装或替换全局包。

```bash
source /data1/t00995472/spark/activate_ascend.sh
npu-smi info
python -B benchmark_spark_vllm.py --npu 2 --enforce-eager --fused-decode \
  --input-tokens 36864 --output-tokens 4096 \
  --fused-library /data1/t00995472/spark-bf16-opt/attention/balanced7_all/libdecode_layer_stage_prefetch.so \
  --result /data1/t00995472/spark-bf16-opt/new-balanced7-model.json
```

候选构建和独立检查见 [attention/README.md](../kernel_optimization/attention/README.md)。
