# BF16 LM head 独立实验

结论：保留独立候选，暂不接入完整模型。满载交错复验中，最佳自定义实现仅比 torch 快 **1.31 µs（0.27%）**；按每 token 调用一次估算，对约 7.56 ms 的融合 decode 仅相当于 0.02%，尚未做完整模型收益验证。未量化权重或 KV，未修改全局适配器、安装包或原始融合源码。

## 条件与结果

物理 NPU 6，Ascend950PR_957b；BF16，输入 `[1,2560]`，真实 embedding/LM head 权重 `[131072,2560]`，完整输出 131072 个 logits。每种方法轮换 3 份独立的 671,088,640 字节权重分配，减少重复单矩阵缓存命中的影响。自定义权重按 M 行与 64 列分块重排，输入与权重仍为 BF16、乘加使用 FP32。

| 实现 | 初筛均值 / µs | 持续负载复验 / µs |
| --- | ---: | ---: |
| torch linear | 492.132 | 490.529 |
| M4 / 56 logical blocks | 491.993 | — |
| M8 / 56 logical blocks | 499.060 | — |
| M16 / 56 logical blocks | 494.056 | — |
| M8 / 56 logical blocks，允许 L2 | 492.326 | — |
| M8 / 28 logical blocks | 485.837 | 492.476 |
| M8 / 112 logical blocks | 482.626 | 489.216 |

初筛为 10 轮、每点 30 次调用，未记录满载频率；不能把其中的约 2% 差异视作成立的收益。复验为 4 轮交替正反顺序、每点 3000 次，预先持续负载 3000 次；12 个测量窗口均有 2 个忙时频率样本，全部 1650 MHz。112 是逻辑 block 数，不代表有 112 个物理 Vector 核。

准确性：初筛 8 个随机 BF16 输入 × 6 个变体全部通过；相对独立 CPU FP32 累加、一次 BF16 舍入的参考，最大 NRMSE `4.02e-5`，全部 top-1 一致。持续负载复验两变体重新检查，相对 torch NRMSE `1.19e-5`、top-1 一致。

## 搬运诊断

复用了 round2 的 Level1 PipeUtilization 采集，核对到 21 个任务：6 个变体各 3 次，torch MatMulV3 3 次。自定义 M4 的 MTE2/AIV 约 93%，M8/M16 约 97%；torch MatMulV3 的 MTE2/AIC 约 99.7%。各流水线重叠，比例不能相加，也不能据此断言 HBM 已饱和。

另测 GM→UB 流式读取探针：同样读取 671 MB、轮换 3 份分配，tile 为 16/32/64/128 KiB，逻辑 block 为 28/56/112；每点预热 60 次，测量 3 × 3000 次。最佳为 **488.804 µs / 1.3729 TB/s**（128 KiB、112 blocks）；56 blocks、64 KiB 为 489.377 µs。62 个忙时样本均 1650 MHz。

这是该访问路径的实测速率，不是硬件峰值、性能下界或最优证明。探针仅逐 tile 验证前 32 字节；保留的设备 IR 确认请求仍为完整 tile 搬运，未直接验证未采样字节。现有数据表明，单独减少 Vector 乘加或 logits 写回难以带来明显收益；更值得投入融合层内调度、attention 或完整权重路径的其他改进。

## 证据与复现

全部产物在 `/data1/t00995472/spark-bf16-opt/lm_head/`：`round2/result.json`、`round2/profile/`、`read/result.json`、`read/health.json`、`sustained/result.json`、`sustained/health.json`、`analysis.json` 和 `manifest.json`。频率采样不等同于完整硬件校准 attestation。

失败候选的构建、初筛和持续负载脚本已从工作区移除，保留本页及 `summary.json`。整理前完整源码快照位于 `/data1/t00995472/spark-workspace-archive/20261016-143530/before-cleanup.tar.gz`。

仍有用途的流式读取探针移至 [diagnostics/bandwidth](../../diagnostics/bandwidth/)，构建与运行方法见 [诊断工具说明](../../diagnostics/README.md)。原始采集、脚本来源哈希和全部结果继续保留在数据盘。
