# Spark vLLM Ascend 延迟测试

```bash
cd /home/t00995472/spark
source /data1/t00995472/spark/activate_ascend.sh
python -B benchmark_spark_vllm.py --check-only
python -B benchmark_spark_vllm.py --enforce-eager
```

默认物理 NPU 2、BF16、TP=1，输入 32768 token、输出 4096 token，
最大上下文 36864 token，单个请求串行执行（batch=1）。
先执行一次相同长度的预热，再测量三次。输入是重复中文文本的 token ID，
用于合成长上下文性能测试，不代表真实业务文本的质量或速度分布。
忽略 EOS 并校验输出长度；不使用聊天模板。

- TTFT：提交请求到收到第一个输出 token 的时间。
- TPOT：`(最后 token 接收时间 - 首 token 接收时间) / (4096 - 1)`。
- 单位分别为 ms 和 ms/token；报告每次结果、均值与中位数。
- 使用引擎客户端流式时间，包含调度和进程通信，不包含模型加载、分词、预热及 HTTP。
- 关闭前缀缓存和推测解码；启用分块 prefill，默认预算 32768 token。
  默认使用框架图执行配置，可通过 `--enforce-eager` 切换，比较结果时应保持一致。
- 结果写入 `/data1/t00995472/spark-benchmark/result.json`，可用 `--result` 修改。
- `--npu 3` 可指定其他空闲卡，禁止 NPU 0。脚本发现目标卡有进程会退出。
- `--warmup`、`--runs` 可调整次数。重复执行默认覆盖结果文件。

## 本机模型适配

已在现有 venv 中以 `--no-deps --no-build-isolation -e` 安装模型 README 指定的
官方插件 https://github.com/XHToken/Spark-plugin ，未升级 vLLM、Ascend 或 torch。
源码位于 `/data1/t00995472/vllm-install/Spark-plugin/`，不可删除或移动。
插件版本 0.1.0，源码提交 `3e1040e63a5907e4c748a485d6795d739dcd5bd6`。
基准脚本在架构检查前显式调用 `load_general_plugins()`。

本机 vLLM 0.23.0 / Ascend 0.23.0.post1 需要两份兼容补丁：

- `spark-plugin-vllm023.patch`：按当前 vLLM 参数的 weight_loader 加载 gate/up
  分片；为 Ascend FIA 的预填充传入连续的 V 张量。
- `ascend-rope-head-tiling.patch`：RoPE 的 head 分块不超过实际 head 数，
  修复 950PR、head_dim=256 时默认 64-head 分块造成的 UB 溢出。

两份补丁已应用到本机对应源码。重建环境时，在上述精确插件提交及当前 Ascend
源码上分别执行 `git apply`（不要在已应用补丁的目录重复执行），然后安装插件：

```bash
source /data1/t00995472/spark/activate_ascend.sh
unset DEBUG
python -m pip install --no-deps --no-build-isolation -e /data1/t00995472/vllm-install/Spark-plugin
python -B check_spark_compat.py --npu 4
```

已通过 MLP 分片加载与 RoPE 数值检查，并在物理 NPU 4 上通过完整权重
128 输入 / 8 输出的短请求验证。正式长序列测试日志：
`/data1/t00995472/vllm-install/spark-benchmark-32k-4k.log`。
实际测试使用 `--npu 4 --enforce-eager`；默认设备仍为 2。

2026-10-15 的完整长度实测（BF16、TP=1、batch=1，完整预热一次后测三次）：

| 指标 | 第 1 次 | 第 2 次 | 第 3 次 | 均值 | 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTFT (ms) | 1449.926 | 1487.494 | 1510.179 | 1482.533 | 1487.494 |
| TPOT (ms/token) | 12.258 | 12.304 | 12.293 | 12.285 | 12.293 |

三次输入均为 32768 token、输出均为 4096 token，校验通过。
平均首 token 至末 token 总时延加 TTFT（E2E）为 51.788 秒。
结果文件 `/data1/t00995472/spark-benchmark/result.json` 包含逐轮数值和环境版本。
以上为合成重复文本及 eager 模式的引擎客户端测量，不包含 HTTP 服务开销。

`--check-only` 只检查设备空闲及架构注册，不保证权重加载和算子兼容。
通用插件白名单 `VLLM_PLUGINS` 如有设置，必须同时保留 Ascend 和 `spark2_5`。
插件自带工具调用解析器测试目前因当前 vLLM 缺少 `find_tool_name` 而失败；
本基准不启用该解析器，这不影响已通过的模型生成检查。
设置 `SPARK_BENCH_DEBUG=1` 可让各进程每 60 秒输出 Python 栈用于诊断；
性能测试不应设置 `ASCEND_LAUNCH_BLOCKING=1`。

## 2026-10-16 eager / ACL 图执行与 profiler 实测

本轮使用物理 NPU 4（NPU 2 被其他进程占用），BF16、TP=1、batch=1，
32768 输入 / 4096 输出，完整预热一次后测三次。前缀缓存、推测解码关闭。
原始结果、日志、设备状态和 profiler 文件保存在：
`/data1/t00995472/spark-benchmark/eager-graph-20261016/`。
未覆盖此前的 `result.json`。

| 指标 | eager | 默认图执行 |
| --- | ---: | ---: |
| TPOT 第 1 次 (ms/token) | 12.125 | 11.023 |
| TPOT 第 2 次 (ms/token) | 12.013 | 11.019 |
| TPOT 第 3 次 (ms/token) | 12.836 | 11.019 |
| 平均 TPOT (ms/token) | 12.325 | 11.020 |
| TPOT 中位数 (ms/token) | 12.125 | 11.019 |
| 平均 TTFT (ms) | 1483.805 | 1542.049 |
| 平均 E2E (s) | 51.954 | 46.670 |

按均值 TPOT 下降 10.584%，生成速度提升 1.118 倍；按中位数下降约 9.1%。
图执行日志明确记录 `FULL_AND_PIECEWISE`、capture size `[1]`、decode FULL 捕获，
以及 `Replaying aclgraph`。图捕获约 1 秒、额外约 0.24 GiB，属于初始化，
不计入 TPOT。不能从该模式名称推断 prefill 也全图执行。

设备采样中，两组各有一次活跃低频读数：eager 1300 MHz，图执行 1200 MHz，
其余活跃采样为 1650 MHz。温度范围分别为 77–88°C、72–94°C。
因此这些是本轮运行的观测收益，不是严格同频校准的硬件性能结论，
也不能据此建立 roofline。eager 的持续采样覆盖后半段，未覆盖启动和完整预热。
两组顺序运行，没有随机交错；图模式在 eager 之后测量。

### 独立 profiler 窗口

采用当前 vLLM Ascend 自带 `TorchNPUProfilerWrapper`，其底层是 torch-npu profiler，
CPU+NPU、Level1、PipeUtilization，关闭 stack/memory/frontend 采集。
本机 CANN runtime 9.1.0，包时间戳 `20260730_231653901`；msprof 26.1.0。
这些实测不代表 msprof skill 在其他芯片/构建上的验证覆盖可自动迁移到 957b。

两组均使用与完整对照相同的引擎上下文容量 36864，并完成一次 32K/4K 完整预热。
采集请求为 32768 输入 / 128 输出；`delay_iterations=16`、`max_iterations=16`。
worker 在执行模型前调用 profiler.step()：跳过第 1 步 prefill 和随后 14 步 decode，
记录 worker 第 16–31 步，共 16 步纯 decode，约为 32K 上下文。
不把 profiler 开启/关闭与同步解析造成的停顿算作生产 TPOT。

两份 `kernel_details.csv` 均有 9664 行，即每步 604 个 kernel。
逐类核验均通过：embedding 16、attention 576、matmul 2896、RoPE 576、argmax 16。
原始日志有 `Stop profiler while current state is RECORD` 提示；因此额外核对了上述计数，
确认这些关键类别完整覆盖 16 步，不以 profiler 成功退出代替完整性检查。
两组采集请求的 128 个输出 token SHA256 相同；这只是本输入的生成一致性，
不是广泛的数值精度验证。完整 eager 基准启动在时间戳/摘要功能加入之前，
仅有原先的汇总时间；不能与图执行提供对称的逐 token 分位数或完整输出摘要对比。

| 16 步采集窗口的量 | eager | 图执行 |
| --- | ---: | ---: |
| kernel 耗时之和 / 步 (ms) | 10.582 | 10.594 |
| 首 kernel 开始至末 kernel 结束 / 16 (ms) | 20.929 | 11.139 |
| kernel 时间区间并集之外的间隙 / 16 (ms) | 10.347 | 0.545 |

两组 kernel 区间在本窗口内无重叠，耗时和与区间并集一致。
间隙由原始时间线计算，不通过相减重叠流水线占比得到。
采集给 eager 引入很大额外发射开销，不能把 20.929→11.139 当作生产加速比，
也不能将长请求客户端 TPOT 与短采集窗口 kernel 和之差全部称为 CPU 开销。

图采集中存在 16 次 `aclmdlRIExecuteAsync`。
host 侧 `aclnnMatmul` 从 eager 的 2896 次降为 16 次，但设备侧仍是 2896 次 matmul。
这与“图重放减少逐算子主机处理、kernel 本身基本不变”的解释一致。
图中还有每步 stream synchronize，它维持注意力参数更新与 replay 的正确顺序；
其主机耗时与设备执行可能重叠，不能直接加到设备耗时，也不能直接删除同步。

图执行采集的 kernel 时间构成（均为每 decode 步；占比以 10.594 ms 为分母）：

| 部分 | 时间 (ms) | 约占比 |
| --- | ---: | ---: |
| MLP gate/up + down 矩阵乘 | 4.422 | 41.7% |
| 9 层全局注意力 | 2.302 | 21.7% |
| QKV 矩阵乘 | 0.944 | 8.9% |
| attention 输出投影 | 0.723 | 6.8% |
| LM head | 0.491 | 4.6% |
| 27 层滑窗注意力 | 0.387 | 3.7% |
| headwise gate 矩阵乘 | 0.114 | 1.1% |
| 其他 kernel | 1.211 | 11.4% |

全局/滑窗分类依据模型源码顺序：36 层中每第 4 层为全局注意力，
对按开始时间排序的每组 36 次 FIA 调用按此映射。
图执行中全局注意力每层约 255.8 us，滑窗每层约 14.35 us。
差异与窗口实际起效一致，但没有直接测得窗口外 HBM 读取字节数。

MLP gate/up 的 MTE2 时间 / AIC 时间约 97.6%，MAC 约 12.0%；
down 约 94.4% / 11.4%；LM head 约 99.5% / 12.8%。
全局注意力约 86.9% / 32.3%。这些是各任务对应 pipe time 之和除以对应 AIC time 之和，
不同流水线会重叠，不能相加成占比或推算“空泡”。它们支持优先检查数据搬运、
batch=1 矩阵向量访问和 KV 读取路径，不能证明 HBM 已达到物理带宽上限。
本轮没有 HBM 实际流量或本机可持续带宽校准，所以仍不给出理论最小 TPOT。

后续优先级：先优化 MLP 的真实形状（gate/up 权重 20480×2560、down 2560×10240），
再分析全局注意力的 KV 访问和分块。单独列量化实验，验证精度与真实反量化成本。
融合 Norm、RoPE、GELU、门控等小算子可以减少访存和任务数量，但必须报告整个请求收益；
不能预期单靠小算子融合就能把目前约 11 ms/token 减半。
10.594 ms 是当前 kernel 在采集窗口的实测总时间，不是硬件下限。

### 复现与分析

每种模式分开启动，均先检查目标 NPU 空闲。使用新的结果目录，避免覆盖本次证据。

```bash
source /data1/t00995472/spark/activate_ascend.sh
python -B benchmark_spark_vllm.py --npu 4 --enforce-eager --result /data1/t00995472/spark-benchmark/eager-new.json
python -B benchmark_spark_vllm.py --npu 4 --result /data1/t00995472/spark-benchmark/graph-new.json
python -B benchmark_spark_vllm.py --npu 4 --enforce-eager --runs 1 --profile-dir /data1/t00995472/spark-benchmark/profile-eager-new --result /data1/t00995472/spark-benchmark/profile-eager-new.json
python -B benchmark_spark_vllm.py --npu 4 --runs 1 --profile-dir /data1/t00995472/spark-benchmark/profile-graph-new --result /data1/t00995472/spark-benchmark/profile-graph-new.json
python -B analyze_spark_profiles.py /data1/t00995472/spark-benchmark/eager-graph-20261016
```

`--profile-output-tokens` 默认 128，只改变采集请求长度，完整预热仍使用 `--output-tokens`。
`--profile-delay` / `--profile-steps` 默认均为 16。汇总工具固定核验本轮的 16 步；
若改变步数，需同步调整核验值，不能把不通过的核验当作完整采集。
汇总工具直接读取原始 CSV，输出 `analysis.json`；
`launch-analysis.json` 保存本轮另行提取的 runtime API 数量。
Chrome/Perfetto 时间线位于各采集目录的 `ASCEND_PROFILER_OUTPUT/trace_view.json`。
