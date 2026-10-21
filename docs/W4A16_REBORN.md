# W4A16 新 decode 设计与实验记录

目标是完整模型 32768 输入 / 4096 输出、单请求达到 300 token/s，即 TPOT ≤3.333 ms。输出可读性是本轮质量底线；数值误差仍记录，不据此声称通过旧版严格 logits 门限。

## 基线诊断

2026-10-21 在物理 NPU 4 采集 v10b，16 个 decode 步含 576 次融合层调用。融合层合计 3.434 ms/token，BF16 LM head 0.490 ms/token。融合层平均 AIV vector ratio 0.652、MTE2 ratio 0.519；计数器存在重叠，不能相加推导空闲时间。profile 含采集开销，不作为端到端性能验收。

原始采集：`/data1/t00995472/spark-w4a16-reborn/profile/`。运行记录：同级 `baseline-profile/`。目前没有验证 340 token/s 是本机可达带宽上限；它仅作为用户给出的分析参照。

按实际矩阵尺寸分解搬运量，比把 decode 当成一个大 GEMV 更能说明优化方向：

| 每 token 的唯一数据量，32K 上下文 | v10b | 新版 |
| --- | ---: | ---: |
| 36 层 INT4 线性权重与 scale | 2,005,401,600 B | 1,946,419,200 B |
| LM head | 671,088,640 B（BF16） | 173,015,040 B（INT4 + BF16 scale） |
| 9 层完整 attention 的 K+V | 1,207,959,552 B（BF16） | 603,979,776 B（FP8） |
| 27 层滑窗 attention 的 K+V | 56,733,696 B | 56,733,696 B |

第一行的差异来自 scale 从 FP32 存储改为原始 BF16，不改已有 INT4 代码。表中未列小型 g_proj、norm、激活及工作区，也未计执行中的重读和 padding；它是流量预算，不是 HBM 计数器实测。新版在这些主要项上少读约 1.16 GB/token，改变了原先“340 token/s 极限”所依赖的数据量，因此不能将 310 token/s 解释成达到同一理论上限的某个百分比。

## 新线性层算法

从 `y[n] = Σg scale[n,g] × Σk∈g q[n,k]x[k]` 出发，将 scale 移至 G128 点积之后。采用输出通道方向 SIMD：每个寄存器覆盖 128 个输出，每次读取 64 字节 INT4，按需转 FP16，广播一个输入。组内分四条 FP16 累加链，每条 32 项；四链合并后转 FP32、乘原始 BF16 scale，跨组 FP32 累加。

这会改变原生 `BF16(q×scale)` 的舍入边界，并引入组内 FP16 累加误差。输入与最终输出仍是 BF16，不把激活量化成 INT8。需要记录极值与生成输出，不能仅凭有限值检查宣布成功。

布局以 128 输出通道为 tile，K 分片使一次 DMA 为约 66–83 KiB；使用双缓冲读取下一任务。INT4 的低/高半字节分别对应 tile 内前/后 64 通道。输出是各 K 分片的 FP32 部分和，下一阶段将其归约并融合后处理。

## 进展与证据

实验根目录：`/data1/t00995472/spark-w4a16-reborn/`。构建目录不可覆盖，目录内保存静态源码、库及 SHA256 manifest。

| 构建/记录 | 结论 |
| --- | --- |
| r1 | 原型上半通道转换错误，已否决；常数和通道编码输入发现上半输出为零 |
| r2 | 显式提取高半字后，QKV/O/gate-up/down 的真实权重 NRMSE 为 0.158%/0.169%/0.165%/0.162% |
| r4 | 新调度器已集成；短滑窗层通过，32K 滑窗层轮换权重约 62.04 μs；完整 attention 部分 PV 结果异常 |
| r5 | 修正 AIV→AIC 启动信号的排序，32K 完整 attention 层全部阶段通过，约 162.50 μs；同条件 v10b 约 167.23 μs |
| `model-r5-screen/` | 完整模型 32K/512，预热一次、测一次，TPOT 3.903 ms（约 256.2 token/s），尚未达标 |
| r6 / `head-r6.json` | 加入 decode 专用 W4A16 LM head，完整 head 算子约 142.59 μs；实现相对反量化参考通过，量化相对 BF16 的误差单独记录 |
| `model-r6-head-screen/` | 32K/512 完整模型约 3.561 ms/token（281 token/s） |
| `model-r6-graph-screen/` | 内部图重放约 4.390 ms/token，反而退化；512 个自由生成 token 与 eager 一致，未采用 |
| r7 | 四 K 值合并装载的布局验证失败，且无性能收益，已从当前源码撤回 |
| r8 / `kv8-codec.json` | 范围内 FP8 KV 编解码 65,536 值逐位匹配 CPU，编码/解码保护区通过 |
| r9 / `layer-r9-full-32k.json` | 全新 KV 搬运与 Cube/AIV 流水，32K 完整 attention 层全部阶段通过，轮换权重约 120.82 μs |
| `build-final-clamped/` | 显式截断修复 FP8 溢出边界；最终三轮 TPOT 3.217846 ms/token，310.77 token/s |

首次完整层集成复用 v10b 的 Cube attention 原语。r9 重新实现了完整 attention 的搬运、分块及跨核流水；滑窗 attention 和 softmax 的数学原语仍有复用。新的线性层、权重布局、跨 K 合并、RoPE 后处理、MLP 激活、归一化和阶段调度独立实现。

### 屏障缺陷与修复

MTE3 发出的启动 Cube 信号，不能由仅在 MTE2 等待的全核屏障排序。r4 的 Q/K/V 本身及 cache 写入检查正常，但 Cube 的读取可能早于其他核完成写入。r5 在启动 Cube 前、attention 完成后使用完整 `SyncAll<true>()`，同一 32K 输入验证通过。该缺陷不是可接受的量化误差。

### LM head 的新量化边界

原始 embedding / tied LM head 为 BF16。r6 在初始化时复用现有对称 RTN G128 规则，额外准备 decode 专用 INT4 head；embedding 查表及 prefill head 保持 BF16。压缩 head 为 173,015,040 字节，不常驻 BF16 反量化副本。

权重相对 L2 误差 11.646%，构造随机 BF16 激活的 logits NRMSE 14.615%、cosine 0.989495，**未通过旧版数值门限**。这只记录量化代价，不代表自然语言质量；仍须检查自由生成。独立计时含跨 K 分片最终合并，四份权重轮换。

### FP8 私有 KV 存储与 BF16 attention

r9 的完整 attention 层使用 FP8 E4M3FN 存储私有 NZ KV，带有限值饱和。vLLM 的 BF16 KV 和 embedding/prefill 路径保留；首次 decode 前将历史 KV 编码，TTFT 包含该转换。当前实现为复用多请求，仍保留一份 BF16 NZ 历史缓冲，所以字节流量减半不等于总显存减半。

新 attention 每块处理 512 个位置。每个 Cube 的两个 AIV 各取一半 token，读取 FP8、在寄存器中还原为 BF16，然后写入共享 L1。Cube 完成对应 L0 装载后释放 L1，AIV 可提前准备 V 或下一块 K；QK/PV 仍是 BF16 MMAD、FP32 累加，softmax 保持 FP32。不删上下文，不改变滑窗长度。

独立编解码构造数据 NRMSE 2.284%；真实权重单层所用 KV 历史的 K/V 量化误差约 2.66%。单层实现验证的 attention 参考明确使用 **CPU FP8 舍入后的 K/V**，检查的是编码与计算实现，不代表与原 BF16 KV 模型通过相同数值门限。完整模型自由生成质量另行检查。

末轮加入超过 ±448 的有限值后发现：本机编译器仅设置 `SatMode::SAT`，FP8 转换仍会在溢出时输出 NaN。最终编码显式截断到 `[-448,448]`。`kv8-codec-clamped.json` 包含这些溢出样本，全部编码/解码逐位匹配截断后的 CPU 参考且保护区通过；它对原始超范围值的截断误差当然很大，不应将该误差视为范围内 KV 的量化精度。

## 完整模型结果

2026-10-21，物理 NPU 4、Ascend950PR、eager、TP=PP=1、单请求，32768 输入 / 4096 输出，完整预热一次后测三次。最终库记录在 `model-clamped-32k/`，计时审计为该目录的 `analysis.json`：

| 指标 | 第 1 次 | 第 2 次 | 第 3 次 | 均值 |
| --- | ---: | ---: | ---: | ---: |
| TPOT，ms/token | 3.217258 | 3.219741 | 3.216540 | **3.217846** |
| TTFT，ms | 1530.467 | 1547.183 | 1568.039 | 1548.563 |

均值对应 **310.77 token/s**，三轮均低于 300 token/s 目标的 3.333 ms/token。每轮输出 4096 token，三轮 token 列表一致；运行时审计为 144 个原生 W4A16 线性层、16,380 个融合 decode 步和 16,380 个 INT4 head 调用，私有完整 attention KV 为 `torch.uint8`、原生 KV 为 BF16。无 profiler、logits 导出或强制 token。15 个完全落在 decode 窗口内的频率样本均为 1650 MHz，稀疏采样不证明全程恒频。

同卡同负载顺序重测 v10b 为 4.168127 ms/token、239.92 token/s，记录为 `baseline-final-32k/`。新版吞吐提高 **29.53%**、TPOT 降低 **22.80%**。这是顺序测量，未做交错 A/B。该重复文本负载下，最终库三轮生成的完整 4096 token 列表均与基线相同，不能外推为任意问题都相同。

该结果对应 **额外量化 head 与 FP8 私有 KV**，并非只替换线性 GEMV 后仍维持 BF16 head/KV 的结果。仅新线性与调度器的 r5 为约 256 token/s；再加 INT4 head 的 r6 为约 281 token/s。三者的数值预算不同。

## 复现与验证

先按 README 激活环境并确认非 0 设备空闲；使用新的空构建目录：

```bash
bash utils/build.sh /data1/t00995472/spark-builds/reborn-example --kernel w4a16_reborn

python -B utils/benchmark_kernel.py --kernel w4a16_reborn \
  --build /data1/t00995472/spark-builds/reborn-example --npu 4 \
  --layer 3 --length 32768 --weight-rotations 16 --cache-banks 2 \
  --output /data1/t00995472/spark-benchmark/reborn-layer.json

python -B utils/run_model_recorded.py \
  --directory /data1/t00995472/spark-benchmark/reborn-model \
  -- --npu 4 --enforce-eager --w4a16 --fused-decode \
  --model /data1/t00995472/models/Spark-X2.5-4B-W4A16-G128 \
  --fused-library /data1/t00995472/spark-builds/reborn-example/libdecode_layer_w4a16.so

python -B utils/validate_w4a16_model.py --npu 4 --readability-only \
  --contexts 1024 --steps 256 \
  --library /data1/t00995472/spark-builds/reborn-example/libdecode_layer_w4a16.so \
  --output /data1/t00995472/spark-benchmark/reborn-readability
```

独立格式检查可在 `utils/benchmark_kernel.py` 使用 `--kv-codec-only`；独立完整 head 检查用 `--linear-only lm_head --weight-rotations 4`。自然问题工具默认关闭思考模式、允许 EOS 正常结束；可加 `--thinking`。它使用原生与融合路径各自由生成一次，不强制相同 token，不作为性能测试或系统质量评测。

可直接使用的库为 `/data1/t00995472/spark-w4a16-reborn/build-final-clamped/libdecode_layer_w4a16.so`，需配套该目录的 `execution.json`。库 SHA256 为 `546a951e07763f6c1753a559ed43f330a610f6b2c939cede3bdffcd66efa58a7`。源码已移除未调用的旧完整 attention 实现。

最终边界回归为 `edge-clamped-*.json`：完整 attention 长度 1、5121、40960，滑窗长度 514，均使用 32 位 slot、重复三次执行；各阶段、KV 写入和输出/工作区保护区通过。

自然问题记录为 `readability-clamped/readability.json`，原生与融合各自由生成三题，均正常 EOS 结束。中文连贯、无乱码，乘法结果为 391、排序代码有效、邮件完整。科学解释仍有不准确表述，原生对照也有类似问题；这项检查只满足本轮可读性底线，不是系统准确率评测。可读性工具将输出中省略的终止 EOS 计入 decode 调用，431 次执行审计通过。新增量化边界不代表通过旧版严格 logits 门限。
