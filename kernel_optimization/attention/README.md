# BF16 七分片 attention 候选

`prepare.py --candidate balanced7_all` 从原始 `decode_layer_stage_prefetch.asc` 生成副本，将全局 attention 的分片宽度改为 `ceil(length/896)×128`。7 个分片 × 4 个 KV head 对应 28 个 AIC 任务，保留第 8 个无效 partial slot、8 路合并和原工作区布局；滑窗路径、BF16 权重及打包方式不变。

原实现为 8×4 个逻辑任务、每分片 5120 token，映射到 28 个 AIC；长于 35840 的上下文会使前 4 个核承担第二个有效任务。七分片将剩余 KV 工作分摊到 7 个分片。它改变 softmax 分片及归约边界，结果不要求逐位一致。

NPU 2 同进程 baseline/null/候选交错微测中，完整融合全局层在长度 32769 时基本持平，36864 时观测下降 1.77%，40960 时下降 7.24%；这些重复单层结果用于筛选，不能代替完整模型 TPOT。129、32769、36864、40960 的真实权重 CPU 参考、重复执行和 guard 均通过。完整模型结果统一记录在 [BF16_OPTIMIZATION.md](../../docs/BF16_OPTIMIZATION.md)。

早期 `balanced7` 仅在长度≤35840 时使用七分片，其余回退原宽度；本轮完整模型测试使用 `balanced7_all`。NPU 5 上原始 full kernel 与候选均等待，故不使用该卡的 full 结果判断候选；同卡 split 诊断通过，后续 full 复验由 NPU 2 完成。

```bash
source /data1/t00995472/spark/activate_ascend.sh
npu-smi info
# 使用全新目录；构建不执行设备计算
bash kernel_optimization/attention/build.sh balanced7_all \
  /data1/t00995472/spark-bf16-opt/new-balanced7

# 单层 CPU 参考与 baseline/null 微测；先确认物理 NPU 2 空闲
ASCEND_RT_VISIBLE_DEVICES=2 python -B kernel_optimization/bench_pair.py \
  --build /data1/t00995472/spark-bf16-opt/new-balanced7 \
  --layer 3 --length 40960 --repeat 3 \
  --output /data1/t00995472/spark-bf16-opt/new-balanced7/check.json

# 完整模型；候选保持原 ABI 与权重布局
python -B benchmark_spark_vllm.py --npu 2 --enforce-eager --fused-decode \
  --fused-library /data1/t00995472/spark-bf16-opt/new-balanced7/libdecode_layer_stage_prefetch.so \
  --result /data1/t00995472/spark-bf16-opt/new-balanced7/model.json
```

同卡完整请求对照使用 `run_spark_fused_benchmarks.py --candidate-library <so> --directory <新目录>`，并用 `analyze_spark_fused.py --bf16-optimization-comparison <目录>` 汇总。若输入改为 36864，两条命令均加 `--input-tokens 36864`。所有性能统计排除 profiler 与 logits 导出。
