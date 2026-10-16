# Spark-X2.5-4B 昇腾推理

在本机 Ascend 950PR 上运行 Spark 模型，提供 BF16 融合 decode、基准、算子验证和性能诊断。环境与设备使用约定见 [AGENTS.md](AGENTS.md)。

| 目录 | 内容 |
| --- | --- |
| [docs/](docs/) | [融合推理结果与运行方法](docs/FUSED_DECODE.md)、[BF16 优化记录](docs/BF16_OPTIMIZATION.md)、[基准说明](docs/BENCHMARK.md) |
| [kernels/](kernels/) | 已验证的 Ascend C 融合算子及公共头文件；`legacy/` 留存原始 mega 版本 |
| [integration/](integration/) | Spark/vLLM 融合 decode 接入实现 |
| [benchmarks/](benchmarks/) | 完整模型基准、同卡对照、logits 比较和 profiler 分析 |
| [kernel_validation/](kernel_validation/) | 单层真实权重校验、兼容检查与必要的同步诊断 |
| [kernel_optimization/attention/](kernel_optimization/attention/) | 保留的七分片 BF16 attention 候选 |
| [kernel_optimization/diagnostics/](kernel_optimization/diagnostics/) | 阶段计时和 GM→UB 带宽探针 |
| [kernel_optimization/rejected/](kernel_optimization/rejected/) | 未取得收益的实验结论与结果索引；冗余脚本已清理 |
| [patches/](patches/) | 已应用的 Spark/vLLM 兼容与注册补丁 |

根目录 `benchmark_spark_vllm.py` 是命令兼容入口，`spark_fused_decode.py` 是已安装插件使用的导入兼容入口，实现位于上表对应目录。

```bash
source /data1/t00995472/spark/activate_ascend.sh
npu-smi info
# 确认物理 NPU 2 空闲；禁止使用物理 NPU 0
python -B benchmark_spark_vllm.py --npu 2 --enforce-eager --fused-decode \
  --result /data1/t00995472/spark-benchmark/new-fused.json
```

模型、构建、缓存和历史实验结果均位于 `/data1/t00995472/`。本次整理的快照、原文件哈希及移动/删除清单在 `/data1/t00995472/spark-workspace-archive/20261016-143530/`；历史结果中的旧源码路径为当时位置，不重写原始证据。
