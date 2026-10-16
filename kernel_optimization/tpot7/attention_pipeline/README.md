# 全局 attention 两遍计算与预取实验

2026-10-16，物理 NPU 4，Ascend950PR_957b，CANN 9.1.0。保持 BF16、原 ABI、原权重/KV 打包及滑窗实现；仅替换全局 attention。原始 `kernels/decode_layer_stage_prefetch.asc` 未修改。

本轮没有得到值得接入完整模型的新候选。两遍计算通过真实权重 CPU 参考，但相对已对齐到 384 token 的七分片，单层收益约 0.2%，不足以支持 7.0 ms/token 目标。新增的九套 KV 轮换基准揭示了单套 KV 微基准的缓存影响，适合后续筛选；两者均不能代替完整模型 TPOT。

## 产物和结果

生成源码、共享库、构建日志、原始 JSON、诊断 workspace 均位于：

`/data1/t00995472/spark-tpot7/attention_pipeline/`

`summary.json` 汇总验证、每轮计时和源码/库 SHA256。所有表中计时来自 layer 3、length=32769、真实 checkpoint 权重、构造 hidden/history；同进程按正反顺序交错 baseline/null/candidate，共 8 轮，每组 250 次。null 调用与 baseline 相同的库，用于观察漂移。数值验证采用既有工程筛选 NRMSE ≤ 3%、cosine ≥ 0.999，未修改阈值，不代表厂商精度标准。

| 候选 | KV 工作集 | baseline / candidate（μs） | 单层下降 | 结论 |
| --- | --- | ---: | ---: | --- |
| v1：完整 QK 存 UB，分片统一 softmax，PV 跨 tile 在 L0C 累加 | 单套 | 231.086 / 229.308 | 0.769% | 基线为原始固定分片，收益含七分片边界变化 |
| v2：概率整块传入 L1，PV 用 128-token 双缓冲 | 单套 | 229.927 / 229.594 | 0.145% | 基线为 balanced7_tile384，收益很小 |
| v3：v2 加 Q 在整个 QK 阶段复用 L0A | 单套 | 229.600 / 229.204 | 0.172% | 收益很小 |
| v3：同上 | 九套轮换 | 263.183 / 262.749 | 0.165% | 流式 KV 工作集仍无显著改善 |
| v4：原 tile384 流水线，仅禁用 K/V L2 hint | 单套 | 229.318 / 311.382 | -35.786% | 明显退化，y/resout 与基线逐元素一致 |
| v5：QKV 计算时，Cube 先读首块历史 K | 九套轮换 | 263.291 / 262.876 | 0.158% | 收益很小，y/resout 与基线逐元素一致 |

除 v1 外，表中基线均为 `/data1/t00995472/spark-tpot7/tile384/build/libdecode_layer_stage_prefetch.so`。表中 null 相对 baseline 变化绝对值均小于 0.04%。不同表行不是同频跨运行对照，不作硬件带宽上限或完整模型加速结论。

`bench_rotating.py` 克隆九套 ND/NZ K/V，共约 2.26 GiB，保持相同非 identity 页表，并对四个 cache 指针一同轮换。首次 CPU 参考验证仍使用 poisoned 当前 slot，候选必须正确执行 cache 写入；随后轮换基准复用已验证数据。输出和 workspace 复用，权重仍只有一层，不能把它叫作完整模型或严格冷缓存实验。

## 结构及数值

- v1/v2/v3 的七分片宽度为 `ceil(length / 2688) * 384`，最大 6144。每个 AIV 的两个 query head 共占 48 KiB FP32 QK；概率仍舍入为 BF16，softmax 使用真实分片最大值，PV 用 FP32 L0C 累加后归一化。
- v2/v3 的完整概率 L1 占 192 KiB。QK 阶段第二个 K buffer 临时使用尚未写概率的 L1 区域；进入 PV 前 K 已退役，两个 128-token V buffer 位于其他 L1 区域。
- 两遍方案改变了 softmax 和 PV 累加顺序。v1 输出相对原始 baseline 的 y NRMSE 为 0.001031，非逐元素相等；所有记录的 CPU 参考/guard/3 次重复调用通过。未运行全模型 logits/token 验证，因此不具备接入资格。
- v3 额外通过 length=40960，覆盖最大分片存储容量。
- v5 性能证据使用 `length>=32768` 的历史首块条件。当前 `prepare_prefetch.py` 生成的是 v6：按每个 task 实际拷贝页的终点判断 `start + ceil(min(N, valid)/128)*128 <= floor(pos/128)*128`，仅在不读取当前 token 页时提前预取。v6 在 length=129 和 35841、repeat=3 下通过 CPU 参考和 guards，尚无单独性能结论。历史 v5 生成源码保留在数据盘，未覆盖。

关键生成源码 SHA256：

- v3：`768b32966e3cbfb8554c8808218f1d8ff9b086bc09e92a00c5d8e1f8be72c81b`
- v6：`af1e17842af1831aed3b9fdffe233f35867931facd332943869ad9d41123c495`

## 阶段诊断

`v2_profile` 在 QK、等待概率、PV 和 AIV 对应阶段加入 `PIPE_ALL` 屏障与 `GetSystemCycle()`。`v2-profile-timing.json` 的中位原始 ticks 为：QK 34016、Cube 等待概率 1832、PV 29318；AIV softmax 763、概率拷贝 1043。诊断库相对未插桩 v2 慢 0.516%，这些数字不转换为 μs，也不代表无屏障生产流水线。它支持继续检查 K/V 搬运及 Cube 路径的优先级，不足以区分 HBM、L1→L0 和 MMAD 各自贡献。

## 复现

先检查物理 NPU 4 空闲；禁止使用 NPU 0。以下命令均使用目标 venv，不修改安装环境。

```bash
source /data1/t00995472/spark/activate_ascend.sh
export ASCEND_RT_VISIBLE_DEVICES=4

bash kernel_optimization/tpot7/attention_pipeline/build_reuse.sh \
  /data1/t00995472/spark-tpot7/attention_pipeline/new-v3

python -B kernel_optimization/tpot7/attention_pipeline/bench_rotating.py \
  --build /data1/t00995472/spark-tpot7/attention_pipeline/new-v3 \
  --baseline-library /data1/t00995472/spark-tpot7/tile384/build/libdecode_layer_stage_prefetch.so \
  --layer 3 --length 32769 --repeat 3 \
  --output /data1/t00995472/spark-tpot7/attention_pipeline/new-v3-rotating.json

bash kernel_optimization/tpot7/attention_pipeline/build_prefetch.sh \
  /data1/t00995472/spark-tpot7/attention_pipeline/new-v6

python -B kernel_optimization/tpot7/attention_pipeline/verify.py \
  --build /data1/t00995472/spark-tpot7/attention_pipeline/new-v6 \
  --layer 3 --length 129 --repeat 3 \
  --output /data1/t00995472/spark-tpot7/attention_pipeline/new-v6-129.json

python -B kernel_optimization/tpot7/attention_pipeline/summarize.py
```

`build.sh`、`build_bulk.sh`、`build_reuse.sh`、`build_cache.sh`、`build_prefetch.sh` 分别生成 v1、v2、v3、v4、v6；原来的 v5 应直接使用其保留源码重编译，不能用当前 v6 生成器冒充历史 v5。
