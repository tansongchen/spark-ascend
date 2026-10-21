# Spark-X2.5-4B 昇腾推理

本仓库面向 Ascend 950PR，为 Spark-X2.5-4B 提供可直接编译的融合 decode 算子、vLLM 接入和统一 benchmark。prefill 沿用原模型路径，decode 每 token 调用 36 个融合层，LM head 与采样仍由 vLLM 执行。

仅保留三个算子版本：

| 构建名称 | 源码 | 历史完整模型结果 |
| --- | --- | --- |
| `stage_prefetch` | [BF16 原始版](kernels/decode_layer_stage_prefetch.asc) | 约 7.56 ms/token |
| `lossless_prefetch` | [BF16 无损版](kernels/decode_layer_lossless_prefetch.asc) | 约 6.82 ms/token，权重逐位还原 |
| `w4a16_v10b` | [W4A16 v10b](kernels/decode_layer_w4a16_v10b.asc) | 约 4.165 ms/token；完整模型 logits 未通过既定门限 |

这些数值来自既有 32768 输入 / 4096 输出、单请求实验，并非一次统一交错对照。方案与验收边界分别见 [BF16 无损压缩](docs/LOSSLESS_DECODE.md) 和 [W4A16](docs/W4A16.md)。

## 目录

- `kernels/`：3 个静态 `.asc` 和 5 个依赖头文件，构建与运行均不生成 kernel 源码。
- `utils/`：模型接入、权重打包、构建、验证和分析工具。
- `benchmark.py`：完整模型 TTFT/TPOT benchmark，直接接收模型参数。
- `patches/`：Spark 插件与 vLLM Ascend 的本机兼容补丁。
- `pyproject.toml`：本地 W4A16 插件安装配置。
- `docs/`：两篇优化思路文档；[AGENTS.md](AGENTS.md) 保存工作与维护约定。

## 环境与安装

本机已验证环境：Ascend 950PR（`Ascend950PR_957b`）、CANN/NNAL 9.1.0、Python 3.12、torch 2.10.0+cpu、torch-npu 2.10.0.post4、vLLM 0.23.0+empty、vLLM Ascend 0.23.0.post1。需要 950 系列算子包及 Spark 官方模型插件。

在仓库根目录执行：

```bash
source /data1/t00995472/spark/activate_ascend.sh
unset DEBUG
python -m pip install --no-build-isolation --no-deps -e .
npu-smi info
```

使用本机 venv，默认物理 NPU 2；若被占用，运行命令中改用其他空闲卡，禁止物理 NPU 0。设置可见设备后，进程内 `npu:0` 对应所选物理卡。模型、缓存和构建产物均放在 `/data1/t00995472/`。

本机已经应用以下补丁，现有环境无需重复应用；重新部署时需同步处理：

| 补丁 | 目标与用途 |
| --- | --- |
| [spark-plugin-vllm023.patch](patches/spark-plugin-vllm023.patch) | Spark 插件：权重加载 API 与连续 V 张量兼容 |
| [ascend-rope-head-tiling.patch](patches/ascend-rope-head-tiling.patch) | vLLM Ascend：950PR RoPE 分块修复 |
| [spark-fused-register.patch](patches/spark-fused-register.patch) | Spark 插件：按开关注册 `utils.spark_fused_decode` |

仓库本身是 editable 安装源，移动后需要重新安装。benchmark 自动设置模型开关；未启用融合或量化时保留原模型路径。

## 构建三个版本

为每个版本使用新的空目录。以下路径用于示例，重跑时换一个目录：

```bash
export SPARK_BUILD=/data1/t00995472/spark-builds/example
export SPARK_RESULTS=/data1/t00995472/spark-benchmark/example
mkdir -p "$SPARK_RESULTS"

bash utils/build.sh "$SPARK_BUILD/stage" --kernel stage_prefetch
bash utils/build.sh "$SPARK_BUILD/lossless" --kernel lossless_prefetch
bash utils/build.sh "$SPARK_BUILD/w4a16" --kernel w4a16_v10b
```

BF16 两版生成 `libdecode_layer_stage_prefetch.so`，W4A16 生成 `libdecode_layer_w4a16.so`。三个源码分别编译，不能合并链接；部署时必须一起复制共享库和同目录的 `execution.json`，模型适配器据此选择权重布局与工作区。

## 运行 benchmark

先验证单层真实权重，再进行 NPU event 计时：

```bash
python -B utils/benchmark_kernel.py --kernel lossless_prefetch \
  --build "$SPARK_BUILD/lossless" --npu 2 --layer 3 --length 32768 \
  --output "$SPARK_RESULTS/lossless-layer.json"
```

测其他版本时同时修改 `--kernel`、`--build` 和输出路径。单层计时用于筛选，不能替代完整模型 TPOT。

以下三个完整模型命令默认均为输入 32768 token、输出 4096 token、完整预热一次后测三次；短请求检查可追加 `--input-tokens 128 --output-tokens 8 --runs 1`。

```bash
# BF16 原始融合版
python -B benchmark.py --npu 2 --enforce-eager --fused-decode \
  --fused-library "$SPARK_BUILD/stage/libdecode_layer_stage_prefetch.so" \
  --result "$SPARK_RESULTS/bf16.json"

# BF16 无损版，使用同一份 BF16 checkpoint
python -B benchmark.py --npu 2 --enforce-eager --fused-decode \
  --fused-library "$SPARK_BUILD/lossless/libdecode_layer_stage_prefetch.so" \
  --result "$SPARK_RESULTS/lossless.json"

# W4A16 v10b，使用本地 G128 量化 checkpoint
python -B benchmark.py --npu 2 --enforce-eager --w4a16 --fused-decode \
  --model /data1/t00995472/models/Spark-X2.5-4B-W4A16-G128 \
  --fused-library "$SPARK_BUILD/w4a16/libdecode_layer_w4a16.so" \
  --result "$SPARK_RESULTS/w4a16.json"
```

BF16 默认模型为 `/data1/t00995472/models/Spark-X2.5-4B`，可用 `--model` 修改。W4A16 权重导出方法见 [量化方案](docs/W4A16.md)。结果路径应使用新名称，普通模型 benchmark 会覆盖同名结果。

根目录 `benchmark.py` 只运行完整模型，没有子命令。其他工具直接运行对应脚本：

| 脚本 | 用途 |
| --- | --- |
| `utils/benchmark_kernel.py` | 单层正确性检查和计时 |
| `utils/run_model_recorded.py` | 包装完整模型 benchmark，额外记录源码身份和设备采样 |
| `utils/benchmark_paired_model.py` | 同一引擎内 BF16 原始/无损库交错 AB/BA 对照 |
| `utils/run_spark_fused_benchmarks.py` | 原模型、融合及图执行等模式的顺序对照 |

例如 `python -B utils/run_model_recorded.py --directory /data1/t00995472/spark-benchmark/recorded-example -- --npu 2 --enforce-eager`；`--` 后直接传完整模型参数。上述脚本及 `benchmark.py` 均支持 `--help`。TTFT 为提交请求到首 token 的时间，TPOT 为 `(末 token 时间 − 首 token 时间) / (输出 token 数 − 1)`。计时包含引擎调度与进程通信，排除加载、分词、预热及 HTTP；profiler、logits 导出运行不可用作性能结果。输入为重复中文文本，不代表全部业务负载。

## 适用范围与验证

融合路径限定 TP=PP=1、单请求顺序执行、BF16 激活与 KV cache、block size=128、输入加输出不超过 40960，关闭前缀缓存和推测解码。无损版只支持融合 eager；W4A16 文档及示例也采用 eager。滑窗沿用本机 FIA 的 `[pos-512,pos]`、最多 513 个位置，与 Transformers 的 512 位置语义有差异。

BF16 融合接入额外持有重排权重和私有 KV cache，原始实验约增加 13.38 GiB；无损版为支持对照切换，再持有约 4.07 GiB 编码权重。因此编码数据量下降不等于整体显存下降。W4A16 的量化质量与融合实现一致性需要分别评估。

目录精简后的三版构建、单层校验和短请求检查记录在 `/data1/t00995472/spark-three-kernels-check/verification.json`。详细实验流水可从 Git 历史查阅，原始数据仍在 `/data1/t00995472/spark-tpot7/`、`/data1/t00995472/spark-w4a16/` 等目录；本次文档合并未重测性能。
