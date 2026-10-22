# Spark-X2.6 官方 INT4 接入

当前模型为 [XHToken/Spark-X2.6-4B-INT4](https://huggingface.co/XHToken/Spark-X2.6-4B-INT4)，核对 revision `4cd05c9e3bd23b08331fcc70fea6dbd5139ef29f`。下载、环境和运行入口见 [README](../README.md)。

## 上游格式

模型仍使用 `Spark2_5ForCausalLM` / `spark2_5` 架构标识：36 层，hidden=2560，MLP=10240，16 个 Q head、4 个 KV head、head_dim=256，vocab=131072。版本号更新不意味着应将架构字符串改为 `spark2_6`。

`model.safetensors` 是单文件 checkpoint，不要求分片索引。`config.json` 的 `quantization_config` 为 `compressed-tensors`、`mixed-precision`：

| 模块 | 权重 | scale / 零点 | 激活 |
| --- | --- | --- | --- |
| QKV、gate/up/down | G32 非对称 INT4 | BF16 数值 scale、INT4 零点 | BF16 |
| out_proj、LM head | 逐输出通道 INT8 | FP32 scale，对称 | 动态逐 token INT8 |
| embedding、gate、norm | BF16 | 无 | BF16 |

`tie_word_embeddings=false`，LM head 有独立 INT8 权重，不能复用 embedding，也不能重新量化成旧版 INT4 head。模型没有要求 FP8 KV，本接入保留 BF16 KV。

实际权重中 36 个 down_proj 的 scale 容器是 FP32，尽管量化配置声明 BF16。加载时逐值检查它们可由 BF16 **精确表示**才转换；若上游后续版本出现不可精确表示的 scale，则报错，不能默默截断。out_proj 与 LM head 的 scale 始终以 FP32 保留。

## Ascend 适配

本机 vLLM Ascend 0.23 已有动态 W8A8 算子，但缺少 Dense compressed-tensors W4A16 路径，且其通用 compressed-tensors 配置未为 `ParallelLMHead` 返回量化实现。

[模型适配器](../utils/spark_int4.py) 在本模型实例中补足这两处：

1. QKV/MLP：加载 `.weight_packed`、`.weight_scale`、`.weight_zero_point` 并校验 `.weight_shape`。gate/up 使用官方 Spark 插件的分片 loader 合并；已经合并的 QKV 不再按 q/k/v 名称匹配。
2. [布局转换](../utils/int4_packing.py)：官方 `INT32 [N,K/8]` 沿 K 存储 offset-binary nibble（码值减 8）；零点沿 N 打包为 `[N/8,K/32]`。逐码转换为 Ascend 的 signed-nibble `INT32 [K,N/8]`，scale 转置为 `[K/32,N]`，offset 为零点取负。反量化仍为 `(q-zp)*scale`，没有新量化误差。
3. W4A16 调用 `npu_weight_quant_batchmatmul`，`antiquant_group_size=32`；out_proj 和 LM head 使用原生 `npu_dynamic_quant` 与 `npu_quant_matmul`。W8A8 scale 参数覆写为 FP32，避免被通用 BF16 参数分配截断。
4. 插件只在 `SPARK_INT4=1` 时注册；完整模型 benchmark 依据 checkpoint 自动启用。当前接入限定 TP=PP=1、eager，不替换其他模型的全局量化实现。

加载检查要求所有权重、scale、零点、shape metadata 完整。运行时审计要求 108 个 INT4 模块（36×QKV/gate_up/down）与 37 个 INT8 模块（36×out_proj + head），embedding 保持 BF16，head 不与 embedding 共享。

## 清理与适用范围

旧自制 RTN G128 导出器、专用格式 loader/packer、`v10b`/`reborn` kernel、运行时 head 再量化、FP8 KV 及对应专属验证/分析脚本已移除。它们的分组、零点、head 和激活语义均与官方 checkpoint 不一致。

BF16 `stage_prefetch` / `lossless_prefetch` 保留为独立参考。BF16 工具必须显式指定 `--model`，官方 INT4 使用旧融合库会提前报错。之前的 X2.5 性能和生成结果不作为本模型的验收依据。

## 验证方法

`utils/validate_spark_int4.py --packing-only` 在 CPU 检查全部 16 个权重码及零点码、两种 scale 容器，并确认有损转换/无效 scale 会被拒绝。

`--check-kernels` 用官方 `PackedQuantizationCompressor.decompress` 作为独立权重参考，检查第 0/35 层 QKV、gate、up、down、out_proj 以及 LM head 的真实权重，覆盖 M=1 和 M=17。INT4 重排后反量化权重须逐位相同；NPU 输出相对独立 FP32 matmul 的 NRMSE 须小于 1%。INT8 参考使用动态量化后的激活、原始 INT8 权重和 FP32 scale。

完整生成检查覆盖中文自我介绍、2+3、北京、Python 加法函数，以及 1024 token 分块 prefill 后的 64 token decode，核对停止状态、logprob 有限性和运行时 dtype/调用次数。它是接入冒烟检查，不替代任务质量评测。

## 本次验证记录（2026-10-22）

使用 README 中列出的本机环境和上述固定 revision，原始记录保存在 `/data1/t00995472/spark-int4/migration-20261022/`。

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| Hub 文件校验 | 本地 14 个上游文件校验通过；未下载展示图片 | `checkpoint-verify.log` |
| CPU 布局转换 | 全部 INT4 码、零点及 scale 防损失检查通过 | `packing.json` |
| 真实权重量化算子，物理 NPU5 | 22 项全部通过，最大相对 L2 误差 0.00168321（约 0.168%） | `validation.kernels.json` |
| 默认异步模式生成，物理 NPU7 | 四项请求正常停止；算术输出 `5`、首都输出 `北京`、代码输出 `return a + b` | `validation.json` |
| 分块 prefill 与 decode，物理 NPU7 | 1024 输入、64 输出，所选 token 的 logprob 全部有限 | `validation.json` |
| 运行时量化审计 | 108 个 INT4 和 37 个 INT8 模块均实际调用，BF16 输入、scale dtype 和独立 head 检查通过 | `validation.json` |
| 默认异步 benchmark，物理 NPU7 | 128 输入、8 输出，预热一次、测量一次及量化调用审计通过 | `smoke-npu7.json` |

物理 NPU5 的完整模型首次请求出现过两次停滞，其中一次运行时日志报告设备队列超时；设备健康查询仍为 OK，原因未确定。相同代码在 NPU7 的默认异步模式通过完整生成检查及相同参数的短请求 benchmark。异常进程已清理，记录见 `runtime-observation.json`；不能据此断言所有物理卡均已验证。开启 `ASCEND_LAUNCH_BLOCKING` 的诊断运行不计入性能结果，本次也未运行 32768/4096 标准长序列性能基准。
