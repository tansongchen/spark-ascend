# 权重访问与无损 BF16 存储实验

本目录只生成候选，未修改默认 kernel、模型权重或安装环境。实验使用物理 NPU 7；完整模型验收由主线程执行。产物位于 `/data1/t00995472/spark-tpot7/weight_path/`。

## 访问路径对照

先复用既有阶段插桩和 MLP/LM head 否证记录，再对真实权重形状检查连续分工、循环分工和 L2 策略。纯搬运探针每个策略先进行三次完整字节比较；计时模式只写每 tile 的 32 字节样本，使用三份轮换权重，属于访问路径诊断，不是硬件带宽上限。

| 真实形状 | tile | 连续分工、禁用 L2 | 循环分工、禁用 L2 |
| --- | ---: | ---: | ---: |
| up，104.86 MB | 40 KiB | 84.120 µs | 73.029 µs |
| down，52.43 MB | 80 KiB | 48.142 µs | 34.353 µs |
| O，20.97 MB | 64 KiB | 21.415 µs | 12.836 µs |

但循环分工接入完整融合层没有收益；纯搬运数字不能代替实际算子。允许 L2 的单层固定分配有少量表观收益，12 份完整层权重轮换后消失。

| 单层 layer 0 / 32769 | baseline | candidate | 结论 |
| --- | ---: | ---: | --- |
| up/down 循环分工 | 164.065 µs | 165.336 µs | 慢 0.77% |
| O/up/down 循环分工 | 164.264 µs | 165.267 µs | 慢 0.61% |
| 权重允许 L2，固定分配 | 160.291 µs | 159.005 µs | 不可外推到全模型 |
| 权重允许 L2，12 份完整权重轮换 | 167.693 µs | 168.132 µs | 慢 0.26% |

这些候选均通过 CPU 分阶段参考和 guards，最终输出与基线逐位相同。数据在各目录的 `swa*.json`、`.benchmark.json`、`.health.json`。

## 无损编码

仅 MLP up/down 使用压缩的存储布局，计算前逐位还原完整 BF16，包括所有异常指数、负零、非有限值；没有舍入、截断或量化。QKV/O 的布局与计算不变。CPU 编解码器是 `../lossless/codec.py`；输入须先经过原来的 `tile_weights`。

每 tile 保存 32 字节头、每值 8 位 sign/mantissa、每值 4 位指数差，以及 256 个异常修补项。全模型真实 tile 的最大异常数为 234，超容量会拒绝编码。最初 32 项设计在 layer 0 down 的 38 项 tile 正确拒绝，未执行错误布局；32/64 产物仅为历史验证，不可用作全模型。

- up：20480 个 BF16 值，40960 → 31776 字节，减少 22.42%。
- down：40960 个 BF16 值，81920 → 62496 字节，减少 23.71%。
- ABI 保持相同，但参数 4/5 必须指向对应编码权重，不能直接替换 `.so` 后仍传原 packing。
- `execution.json` 的 `weight_encoding=bf16_lossless12`、`codec_repairs=256`、`codec_base=109` 标识此布局。

O/up/down 的首个 UB 分配统一为两个 64 KiB 槽，以保留跨阶段预取。另设 80 KiB 解码输出区。down 从 BF16 输入逐块精确转 FP32，避免保留一份额外完整 FP32 输入。实际分配 O/up/down 分别 239808/250304/234880 字节，低于本机头文件的 248 KiB 可用上限。原有每个 BF16 舍入点和 GEMV 累加顺序保留。

`lossless_vf.h` 是先验证版本；`lossless_fast_vf.h` 利用 128 个 uint16 lanes 同时恢复 sign/mantissa，并对两个指数域一次加 base，减少整数指令。两者均提供 `launch_decode_tile` 独立验证入口。新候选的全部 65536 BF16 位模式、异常数 0..256、边界位置、动态 base、输入不变和双侧 guards 已在 NPU 上通过；CPU 全部 72 个真实 MLP 矩阵逐位还原通过。独立证据也保存在 `../mlp_pipeline/` 的数据盘目录。

| 单层筛选 | 原库 | 无损原版 | 无损精简 VF |
| --- | ---: | ---: | ---: |
| layer 0 / 32769 | 164.037 / 164.228 µs | 149.892 µs | 142.713 µs |
| layer 3 / 32769 | 230.698 / 228.631 µs | 216.535 µs | 207.416 µs |
| layer 11 / 40960，含 234 项异常 tile | 291.628 µs | — | 270.488 µs |

各列来自独立的同卡 AB/BA 运行，不能把不同列当作同一计时窗口。每次 3000 预热、6 轮交替次序、每组 3000 次，含 baseline/null 对照；`start.record()` 后提交并同步起始事件。所有对应的 CPU 分阶段参考与 guards 通过，候选最终输出和 residual 与原库逐位相同。固定层微测只是筛选，最终以完整模型 32K/4K 结果为准。

深 profile（由 MLP 分支采集，均核对到 6/6 launches）显示原版无损解码使 AIV Vector 活动从约 35.5 增至 105.7 µs，MTE2 从约 148.6 降至 116.0 µs。精简 VF 后 Vector 活动降至 79.3 µs，MTE2 为 116.8 µs，整层 AIV 为 143.2 µs，与完整层微测相符。流水线活动可重叠，不能相加解释总时长。

## 复现

```bash
source /data1/t00995472/spark/activate_ascend.sh
# 先查询占用，再选择空闲非 0 物理卡。本分支测量脚本限定卡 7。
export ASCEND_RT_VISIBLE_DEVICES=7
bash kernel_optimization/tpot7/weight_path/build_lossless.sh \
  /data1/t00995472/spark-tpot7/weight_path/rebuild_fast --fast
python kernel_optimization/tpot7/weight_path/bench_lossless.py \
  --build /data1/t00995472/spark-tpot7/weight_path/rebuild_fast \
  --layer 0 --length 32769 --repeat 3 \
  --output /data1/t00995472/spark-tpot7/weight_path/rebuild_fast/swa.json
```

构建生成独立 `identity.json`，分别记录基线源码、生成源码、解码头和共享库 SHA256。历史测量文件未被覆盖。稳定原版库是 `lossless256/libdecode_layer_stage_prefetch.so`；精简版是 `lossless_fast/libdecode_layer_stage_prefetch.so`。完整模型接入及长序列验收另见主线程记录。

本轮已验证精简库 SHA256：`d33b8dc1a428a71afa4cd33afb492bd720db6dbaf7c4404c7f161e7c97d40243`。数据盘顶层 `summary.json` 汇总所有单层运行，并链接原始结果；它不包含完整模型 TPOT。
