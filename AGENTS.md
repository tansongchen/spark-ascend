# Spark-X2.5-4B 昇腾推理

## 工作约定

- 始终使用中文回复。
- home 分区剩余空间较少，大文件、下载缓存、模型和构建产物均放在 `/data1/t00995472/` 下。
- `~/.bashrc` 配置了网络代理。下载缓慢时，先比较代理与直连速度；如直连更快，仅对相应下载或安装进程禁用代理，不修改全局代理配置。
- 默认使用 **NPU 2**。使用前检查占用情况；若被占用，可选择其他空闲 NPU，但**禁止使用 NPU 0**。仅查询设备信息不受此限制。
- 使用本机 Python venv 安装和运行，不使用 Docker。

## 项目目录

- `kernels/` 保存可用算子和公共头文件，原始 mega 版本留存在 `kernels/legacy/`；默认构建只包含 stage_prefetch。
- `integration/` 保存模型接入，`benchmarks/` 保存基准与分析，`docs/` 保存实验说明，`patches/` 保存补丁。
- `kernel_optimization/attention/` 是保留候选，`diagnostics/` 是有效诊断工具，`rejected/` 只保留未获收益实验的结论索引。
- 根目录两个 Python 文件是命令/插件兼容入口。目录导航、整理快照和历史证据位置见 `README.md`。

## 本机环境

以下信息于 2026-10-15 安装验证后记录，后续变更时应同步更新。

| 项目 | 路径或版本 |
| --- | --- |
| 硬件 | Ascend 950PR，芯片型号 `Ascend950PR_957b` |
| Python venv | `/data1/t00995472/spark/`，Python 3.12 |
| CANN / NNAL | 9.1.0，安装于 `/data1/t00995472/Ascend/` |
| vLLM | `0.23.0+empty` |
| vLLM Ascend | `0.23.0.post1` |
| Spark vLLM 插件 | `vllm-spark2_5-plugin==0.1.0`，提交 `3e1040e63a5907e4c748a485d6795d739dcd5bd6`，带本地兼容补丁 |
| torch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post4` |
| Triton / Triton Ascend | `3.2.0` / `3.2.2` |
| Transformers | `5.5.4` |
| NumPy | `2.2.6` |
| 本地模型 | `/data1/t00995472/models/Spark-X2.5-4B/` |

安装参考：[vLLM Ascend v0.23.0 官方教程](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html)。

## 激活与运行

```bash
source /data1/t00995472/spark/activate_ascend.sh
```

此脚本加载 CANN、ATB 和 venv，设置 `ASCEND_RT_VISIBLE_DEVICES=2`，并将模型缓存、编译缓存及临时文件放在 `/data1/t00995472/` 下。脚本不会修改代理配置。

- 如需改用其他空闲 NPU，应在激活后设置 `ASCEND_RT_VISIBLE_DEVICES`；不要设置为 `0`。
- 设置可见设备后，进程内的逻辑设备编号会重新从 0 开始；例如仅暴露物理 NPU 2 时，进程内的 `npu:0` 对应物理 NPU 2。
- 运行、安装和验证均使用目标 venv 的 Python，避免误用系统 Python。

## 安装维护注意事项

- 本机需要 **950 系列算子包**；仅安装 `910b` 算子包不足以支持 950PR。当前已补装 CANN 9.1.0 的 950 算子包。
- vLLM 与 vLLM Ascend 均以可编辑方式从源码安装。源码位于 `/data1/t00995472/vllm-install/vllm/` 和 `/data1/t00995472/vllm-install/vllm-ascend/`，**不可作为临时文件删除或移动**。
- `vllm` 版本中的 `+empty` 是 Ascend 插件安装方式的一部分，不代表只能使用 CPU。
- 编译前加载 CANN 和 ATB 环境，并使用目标 venv 已安装的 torch。安装源码包时优先使用 `--no-build-isolation`，避免隔离构建额外下载 CUDA 版 torch。
- 本机继承的 `DEBUG=release` 会使部分源码依赖构建失败；构建时仅在当前进程中 `unset DEBUG`，无需修改全局配置。
- 普通 Triton 与 Triton Ascend 共享部分文件。重装时先卸载两者，再依次安装 `triton==3.2.0` 和 `triton-ascend==3.2.2`，避免文件覆盖顺序错误。
- 安装脚本、构建日志及安装前后的依赖清单位于 `/data1/t00995472/vllm-install/`。其中 `tmp/` 是激活脚本配置的临时目录，清理后应在运行前重新创建。

## 验证结果与已知限制

- 已在物理 **NPU 2** 上通过 torch-npu 矩阵运算验证。
- 已确认 vLLM 激活 Ascend 插件，并通过本地随机权重小模型的端到端生成验证。该测试用于检查推理链路，不用于评估模型质量或性能。
- 验证脚本：`/data1/t00995472/vllm-install/checks/smoke_test.py`；日志：`/data1/t00995472/vllm-install/smoke-test.log`。
- `pip check` 仍会报告 NumPy 依赖声明冲突：Triton Ascend 3.2.2 要求 `numpy==1.26.4`，而该版本 vLLM 的 OpenCV 依赖要求 NumPy 2。当前采用 NumPy 2.2.6，已通过上述推理验证，但不能据此认定所有功能均兼容。
- Spark-X2.5-4B 已安装模型 README 指定的 `XHToken/Spark-plugin`，位于 `/data1/t00995472/vllm-install/Spark-plugin/`，以可编辑方式安装，**不可删除或移动**。vLLM 核心未直接注册该架构，需加载通用插件后查询注册表。
- 本机兼容修改见工作目录的 `patches/spark-plugin-vllm023.patch`（旧版权重加载 API、FIA 连续 V 张量）及 `patches/ascend-rope-head-tiling.patch`（950PR RoPE head 分块 UB 溢出），已应用到对应源码。
- 已通过 MLP 分片加载、完整/部分 RoPE 数值验证，以及物理 NPU 4 上完整模型的 128 输入 / 8 输出短请求生成；短请求日志 `/data1/t00995472/vllm-install/spark-model-smoke.log`。长序列性能结果应另行查看，不能用短请求推断。
- 已在物理 NPU 4 完成完整模型 32768 输入 / 4096 输出、batch=1、BF16、TP=1、eager 模式测试，完整预热一次后测量三次。平均 TTFT 1482.533 ms、平均 TPOT 12.285 ms/token。结果 `/data1/t00995472/spark-benchmark/result.json`，日志 `/data1/t00995472/vllm-install/spark-benchmark-32k-4k.log`，使用说明见 `docs/BENCHMARK.md`。此结果为合成重复文本，不含 HTTP 开销。
- Spark 插件工具调用解析器与当前 vLLM 有独立兼容问题：插件自带测试因 `find_tool_name` 缺失而失败；TTFT/TPOT 基准未使用该解析器。
- 2026-10-16 在物理 NPU 4 完成同配置 eager／ACL 图执行对照及独立 16 步 decode profiler 采集：平均 TPOT 12.325／11.020 ms/token，观测下降 10.6%；两组各有一次低频采样，不作严格同频 roofline 结论。图中 kernel 时间主要为 MLP 矩阵乘与全局注意力。结果目录 `/data1/t00995472/spark-benchmark/eager-graph-20261016/`，完整解读和复现见 `docs/BENCHMARK.md`。
- 2026-10-16 已通过 `integration/spark_fused_decode.py` 将未修改的 `decode_layer_stage_prefetch.asc` 接入完整模型 decode；Spark 插件注册函数已应用 `patches/spark-fused-register.patch`，默认路径不变，基准用 `--fused-decode --enforce-eager` 开启。当前仅支持 BF16、TP=PP=1、单请求、block=128、总上下文≤40960，禁用前缀缓存和推测解码。物理 NPU 2 的 32K/4K 完整预热一次、测三次：原 eager／原图执行／融合 eager 平均 TPOT 为 12.210／11.051／7.556 ms/token；相对图执行生成速度提升 1.463 倍。三组各三轮的 4096 输出 token 全部一致。同前缀 63 步 logits top-1 一致率 100%。融合适配额外占用约 13.38 GiB 非框架 KV 显存；设备采样存在频率差异，结果为本轮观测值。证据 `/data1/t00995472/spark-fused-benchmark/`，说明与复现见 `docs/FUSED_DECODE.md`。
- 2026-10-16 融合路径新增 `--fused-graph`（`SPARK_FUSED_GRAPH=1`），捕获 36 层融合 kernel 调用链；外层引擎仍需 `--enforce-eager`，不是 vLLM 整模型图。物理 NPU 2 同配置 32K/4K、完整预热一次测三次：融合 eager／内部图 TPOT 7.5566／7.5431 ms/token，差异仅 0.18%，E2E 基本持平，默认保持融合 eager。每轮 4096 输出 token 全部一致，短请求同前缀 logits 与融合 eager 逐元素一致；一次捕获、16380 次重放计数通过。证据 `/data1/t00995472/spark-fused-graph/`，说明见 `docs/FUSED_DECODE.md`。
- 2026-10-16 完成保持 BF16、禁止量化的并行优化。MLP tile/DMA 与 LM head 候选无值得接入收益；`kernel_optimization/attention/` 七分片候选通过单层 CPU 参考和完整模型校验，保持原 ABI/打包，使用 `--fused-library <so>` 选择。NPU 2 的 32K/4K 同卡三次对照 TPOT 7.5524→7.5605 ms，无加速；NPU 4 的 36K/4K 对照 7.7714→7.6646 ms，观测下降 1.37%、E2E 下降 1.11%，每轮 4096 输出 token 全部一致。候选不替换默认融合实现。两卡并行、每组顺序测量且有低频采样，不作同频校准结论。证据 `/data1/t00995472/spark-bf16-opt/`，说明与候选用法见 `docs/BF16_OPTIMIZATION.md`。
