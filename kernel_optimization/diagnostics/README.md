# 性能诊断工具

- `stages/build_profile.py`：生成保留原数学和屏障的融合阶段计时副本；`stages/analyze_profile.py` 读取 `validate.py --debug` 保存的 workspace 时间戳。
- `bandwidth/`：独立 GM→UB 流式读取探针，逐 tile 核对采样值。结果只代表该访问路径，不代表硬件带宽上限。

所有生成源码、共享库、日志和结果写入数据盘。插桩结果不作为生产 TPOT。

```bash
source /data1/t00995472/spark/activate_ascend.sh
python -B kernel_optimization/diagnostics/stages/build_profile.py \
  --output /data1/t00995472/spark-bf16-opt/new-stages

bash kernel_optimization/diagnostics/bandwidth/build.sh \
  /data1/t00995472/spark-bf16-opt/new-bandwidth-build
# 先确认物理 NPU 2 空闲；结果目录必须尚不存在
ASCEND_RT_VISIBLE_DEVICES=2 python -B kernel_optimization/diagnostics/bandwidth/bench_read.py \
  --build /data1/t00995472/spark-bf16-opt/new-bandwidth-build \
  --output /data1/t00995472/spark-bf16-opt/new-bandwidth-result
```

历史阶段计时在 `/data1/t00995472/spark-bf16-opt/profile/`；历史带宽数据在 `/data1/t00995472/spark-bf16-opt/lm_head/read/`。
