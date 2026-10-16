# Spark 单层算子验证记录

本页保留第一阶段历史结果；完整模型结果见 [融合推理说明](../docs/FUSED_DECODE.md)。当前默认构建仅包含已验证的 stage_prefetch 版本。

2026-10-16，在 Ascend950PR_957b（28 Cube / 56 Vector）完成独立验证。开始时默认 NPU 2 被占用，因此先在空闲 NPU 3 做分阶段诊断；NPU 2 空闲后，未修改的融合算子在物理 NPU 2 通过完整检查。ACL 查询确认可见设备设为 2 时，用户设备 0 映射到物理 NPU 2。未使用物理 NPU 0，未修改模型、插件、vLLM 或用户提供的四个算子源码文件。

**结论：新增 `decode_layer_stage_prefetch.asc` 的原始单次 launch 融合算子已通过第一阶段数值检查；最初的 `decode_layer_mega.asc` 尚不能直接使用。尚未测量加速比。**

| 验证对象 | 结果 |
| --- | --- |
| 原始 `decode_layer_mega.asc` 和新增 `decode_layer_stage_prefetch.asc` | 两者均编译、链接成功 |
| 原始 mega 的 QKV + RoPE + cache 写入，去掉末尾屏障后提前返回 | 第 0 / 3 层、长度 129 均通过 |
| 原始 mega 的未修改完整入口，NPU 2 | 第 0 层、长度 128 在 40 秒外部超时前未返回，退出码 124 |
| stage_prefetch 的七次 launch 诊断链，NPU 3 | 17 / 17 个完整单层用例通过当前工程精度检查 |
| 用户提供的 stage_prefetch 原始整层融合 kernel，NPU 2 | **17 / 17 个边界用例通过**；不修改设备代码、不拆 launch |
| stage_prefetch 追加用例，NPU 2 | 第 0 层长度 514、第 35 层长度 129，另外两个随机种子；每例复用分配连续调用 3 次，最终输出检查通过 |
| NPU 3 的融合/同步复核 | stage_prefetch 第 0 层长度 128 在 100 秒内未返回；最小 C 同步探针也超时 |
| NPU 2 的最小 C 同步复核 | 初次 15 秒停在设备初始化；延长至 60 秒后完整执行成功，退出码 0 |

同步探针保存在本目录的 `diagnostics/sync_probe.asc` 和 `diagnostics/sync_host.c`，包含 `InitSocState()`，不依赖 PyTorch、权重和 KV cache。两张卡结果不同，尚未确定 NPU 3 阻塞的底层原因，不能据此断言某一方存在缺陷。主结论来自 NPU 2 的未修改融合 kernel，不依赖诊断链来推断屏障和跨阶段 UB 预取正确。

旧版 mega 固定按 32 Cube / 64 Vector 划分工作；新版按 28 / 56 核执行，并循环承担 32 / 64 个逻辑 attention 任务。这是本机上的重要接入差别，但本次没有进一步定位旧版完整入口的确切阻塞指令，不能把核数差异当作已证明的唯一原因。

**数值结果与检查范围**

使用本地真实 checkpoint 的第 0 层（滑窗）、第 3 层（全局）和追加的第 35 层（全局、最终 model.norm）权重。输入 residual 和历史 KV 是固定随机种子构造的数据，并非真实 prompt 的完整模型激活。参考使用 CPU PyTorch FP32 运算，并显式处理 BF16 阶段边界。

| 层 | 上下文长度，包含当前 token |
| --- | --- |
| 0，滑窗 | 1、127、128、129、512、513、514、32768 |
| 3，全局 | 1、127、128、129、5120、5121、32768、36864、40960 |

用例交替使用 int32/int64 slot，使用打乱的物理页表，并先用 NaN 覆盖当前 cache slot，检查 kernel 是否确实写入。比较 Q/K/V、gate、ND/NZ 当前 KV、各非空分区的 attention 输出和 LSE、门控归并、输出投影、FFN 中间激活、down 投影、两次残差和最终归一化。另用实际上游输出计算阶段局部参考，以区分本阶段错误与前序误差传播。

| 中间结果 | 17 例中的最大 NRMSE |
| --- | --- |
| Q + RoPE | 0.0000036411 |
| K + RoPE | 0 |
| V | 0.0003060 |
| attention + gate | 0.0035538 |
| 输出投影 | 0.0049309 |
| FFN 中间激活 | 0.0035543 |
| FFN down 投影 | 0.0040800 |
| 最终 y | **0.00184114（约 0.1841%）** |
| 最终 residual | 0.00182942 |

最终 y 的最小余弦相似度为 **0.999998629**。所有用例的 workspace 末尾、y 和 resout 尾部保护区均未被改写。这只是有限保护区检查，没有进行完整内存/竞争 sanitizer 验证，也没有证明其他 cache 页均未变化。

上述表格对应 17 例边界矩阵，融合执行和诊断链的误差一致。计入两例追加检查后，共 **19 / 19 例通过**；最终 y 的最大 NRMSE 为 **0.00398511（约 0.3985%）**，最小余弦相似度为 **0.999992430**，出现在第 35 层追加用例。重复调用只对最后一次结果做上述逐阶段检查。

工程筛查条件预先设为每个比较项 NRMSE ≤ 0.03 且余弦相似度 ≥ 0.999，并要求数值有限、保护区保持。NRMSE 定义为 `||actual-reference||₂ / ||reference||₂`。此条件不是厂商算子精度标准或模型质量验收标准；报告保留全部原始误差，不能仅凭通过标记认定模型等价。

**滑窗语义仍有差异。** 本次 17 例的数学参考遵循源码，使用 `[max(0,pos-512),pos]`，最多保留 513 个 token。本地 Transformers 5.5.4 的 `masking_utils.py:sliding_window_overlay` 使用 `kv_idx > q_idx - sliding_window`，窗口参数 512 最多保留 512 个 token；本地 Spark 模型调用该 mask。当前 vLLM Ascend 后端将 `pre_tokens=self.sliding_window` 传给 FIA，但不能只凭参数名称判断它包含多少位置。

额外提供 `window_semantics.py`：Q/K=0，历史长度 513，仅 V[0]=1；512 窗口应输出 0，513 窗口应输出约 1/513。在 NPU 2 按当前 vLLM decode 的分页 FIA 形状执行，实际得到 BF16 **0.001953125**，与 513 窗口一致，HF 512 窗口则应为 0。因此源码窗口与本机当前 vLLM Ascend/FIA 行为相符，仍与本地 HF 路径存在差别。证据为 `window-semantics.json` 和 `window-semantics-npu2.log`。NPU 3 上的非分页额外探针超时已单独保留，不能用它作语义判断。

**两版权重布局不同**

两版均需要把 Q/K 每个 head 的旋转部分，按 `[前半 4 行，配对后半 4 行]` 分组重排；QKV 后追加 16 行 attention gate。FFN gate/up 按每组 4 行 gate、4 行 up 交错。

原始 mega 随后使用行主序权重。新增 stage_prefetch 还要求以 64 个 K 元素为单位交错：

```text
[N/M, M, K/64, 64] → [N/M, K/64, M, 64]
```

QKV 主体、O、FFN up 使用 M=8，FFN down 使用 M=4；追加的 16 个 gate 为 M=1，不再交错。最初测试封装漏做新版的这一步，产生明显数值错误；补齐后通过对照。不能把该测试封装错误解释成 kernel 算术错误。

**复现命令**

所有命令在项目根目录执行。先检查设备占用，再将可见设备设为选定的非 0 空闲物理卡；以下使用本次通过融合检查的默认物理 NPU 2。

```bash
npu-smi info
source /data1/t00995472/spark/activate_ascend.sh
export ASCEND_RT_VISIBLE_DEVICES=2
unset DEBUG

# 编译已验证的融合版本，产物全部写入数据盘
bash kernel_validation/build.sh

# 原始融合算子的全部 17 例
python kernel_validation/run_suite.py \
  --build /data1/t00995472/spark-kernel-validation/build \
  --output /data1/t00995472/spark-kernel-validation/fused-suite

# 单例，使用未修改的融合算子
python kernel_validation/validate.py --layer 3 --length 32768 --slot32 \
  --output /data1/t00995472/spark-kernel-validation/manual-fused-l3-32k.json

# 末层 + 三次连续调用，输入和分配复用
python kernel_validation/validate.py --layer 35 --length 129 --slot32 \
  --seed 20261017 --repeat 3 \
  --output /data1/t00995472/spark-kernel-validation/manual-fused-last.json

# 验证当前 FIA 与 HF 的窗口差异
python kernel_validation/window_semantics.py --paged

# 可选：生成七次 launch 的诊断链，原文件保持不变
python kernel_validation/diagnostics/build_split.py \
  --output /data1/t00995472/spark-kernel-validation/split

# 全部 17 例；每例独立进程、顺序运行，超时会停止后续设备测试
python kernel_validation/run_suite.py

# 单例
python kernel_validation/validate.py \
  --build /data1/t00995472/spark-kernel-validation/split \
  --layer 3 --length 32768 --slot32 \
  --output /data1/t00995472/spark-kernel-validation/manual-l3-32k.json

```

复现最小同步问题：

```bash
export ASCEND_SET_ENV=/data1/t00995472/Ascend/cann/set_env.sh
bash .agents/skills/bisheng/build.sh --arch dav-3510 \
  --kernel "$PWD/kernel_validation/diagnostics/sync_probe.asc" \
  --host "$PWD/kernel_validation/diagnostics/sync_host.c" \
  --out /data1/t00995472/spark-kernel-validation/sync_reproducer
timeout -k 3s 60s /data1/t00995472/spark-kernel-validation/sync_reproducer
# NPU 3 曾在 launched 后超时；NPU 2 延长初始化等待后同步成功。
# 设备冷初始化可能超过 15 秒，不能把初始化超时当成同步失败。
```

结果均在 `/data1/t00995472/spark-kernel-validation/`：

- `validation-summary.json`：最终摘要。
- `fused-suite/summary.json`：NPU 2 原始融合算子的 17 例；同目录每例有完整 JSON 和日志。
- `fused-l35-repeat3.json`、`fused-l0-repeat3.json`：追加的末层、不同随机种子和连续调用检查。
- `suite/summary.json`：NPU 3 七次 launch 诊断链的 17 例。
- `mega-qkv-l0-n129.json`、`mega-qkv-l3-n129.json`：原始 mega QKV 对照。
- `mega-fused-npu2-l0-n128.log`：原始 mega 完整入口在 NPU 2 上的超时日志。
- `failures.json`、`sync-reproducer.log`：NPU 3 超时证据；`sync-reproducer-npu2-long.log` 为 NPU 2 成功的最小同步复核。
- `environment.json`、`device-before-suite.txt`、`device-after.txt`：环境和设备记录。
- `split/`、`mega-qkv/`：生成的诊断源码、共享库和执行方式清单。

每例 JSON 记录用户源码、头文件、实际加载共享库及生成诊断源码的 SHA-256。使用 CANN 9.1.0（编译器包时间戳 `20260730_231653901`）、Python 3.12.3、torch 2.10.0+cpu、torch-npu 2.10.0.post4。

本阶段没有修改 vLLM 推理路径、评估完整模型输出质量或测量加速比。日志中的单次 launch 主机等待时间不是性能基准。后续应优先接入已通过的 stage_prefetch 版本，先对接 prefill 历史 cache、层间归一化和只支持单 token / BF16 / TP=1 的分支，再验证完整模型输出和端到端 TPOT。长度超过 40960 必须回退或扩展分区，不能按模型的百万上下文配置直接调用。

整理说明：原始 mega 源码留存于 `kernels/legacy/`；其一次性提前返回生成脚本已清理，原始诊断产物与日志仍在上述数据盘目录。
