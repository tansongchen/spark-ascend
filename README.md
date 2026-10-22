# Spark-X2.6-4B-INT4 昇腾推理

本仓库在 Ascend 950PR 上直接运行上游 [XHToken/Spark-X2.6-4B-INT4](https://huggingface.co/XHToken/Spark-X2.6-4B-INT4)，提供 vLLM Ascend 接入、正确性检查和完整模型 TTFT/TPOT benchmark。默认模型目录为 `/data1/t00995472/models/Spark-X2.6-4B-INT4`。

官方 checkpoint 是 `compressed-tensors` 混合精度：

| 模块 | 官方权重与计算 |
| --- | --- |
| QKV、MLP gate/up/down | 非对称 INT4，group size 32，BF16 激活 |
| attention out_proj、独立 LM head | INT8 权重、FP32 scale、逐 token 动态 INT8 激活 |
| embedding、attention gate、norm | BF16 |

本地适配器只转换 INT4 存储布局，保留上游整数码、零点和 scale；不做校准、重新量化或 checkpoint 导出。旧的自制 G128 量化、`v10b`、`reborn`、运行时 INT4 head 和 FP8 KV 实验已移除，历史代码可从 Git 查阅。格式与验证细节见 [官方 INT4 接入](docs/INT4.md)。

## 环境与安装

本机环境：Ascend 950PR（`Ascend950PR_957b`）、CANN/NNAL 9.1.0、Python 3.12、torch 2.10.0+cpu、torch-npu 2.10.0.post4、vLLM 0.23.0+empty、vLLM Ascend 0.23.0.post1。需要 950 系列算子包及 Spark 官方模型插件。

在仓库根目录执行：

```bash
source /data1/t00995472/spark/activate_ascend.sh
unset DEBUG
# 从旧版升级时卸载旧 distribution，防止残留插件入口。
python -m pip uninstall -y spark-w4a16-ascend
python -m pip install --no-build-isolation --no-deps -e .
npu-smi info
```

默认物理 NPU 2；如果被占用，使用 `--npu` 选择空闲卡，禁止物理 NPU 0。设置可见设备后，进程内 `npu:0` 对应所选物理卡。模型、缓存、构建产物和实验结果统一放在 `/data1/t00995472/`。本仓库是 editable 安装源，移动后需要重新安装。

下载本次核对的上游版本；下载目录直接用于推理，无需转换：

```bash
hf download XHToken/Spark-X2.6-4B-INT4 \
  --revision 4cd05c9e3bd23b08331fcc70fea6dbd5139ef29f \
  --local-dir /data1/t00995472/models/Spark-X2.6-4B-INT4 \
  --exclude 'images/*'
```

本机已经应用以下兼容补丁，现有环境无需重复应用；重新部署时需同步处理：

| 补丁 | 用途 |
| --- | --- |
| [spark-plugin-vllm023.patch](patches/spark-plugin-vllm023.patch) | 官方 Spark 插件的 MLP 分片加载 API、连续 V 张量 |
| [ascend-rope-head-tiling.patch](patches/ascend-rope-head-tiling.patch) | vLLM Ascend 的 950PR RoPE 分块修复 |
| [spark-fused-register.patch](patches/spark-fused-register.patch) | 仅 BF16 参考路径需要的融合注册 |

官方 INT4 由本仓库 `spark-ascend` 插件接入。benchmark 根据 `config.json` 自动设置 `SPARK_INT4=1`；直接调用 vLLM 时需自行设置该变量，并使用 `enforce_eager=True`。若设置了 `VLLM_PLUGINS` 白名单，须包含 `spark_int4` 和环境原有的必要插件。

## 验证与 benchmark

本次已在物理 NPU7 通过生成、分块 prefill 和短请求 benchmark，并完成 22 项真实权重量化算子检查。物理 NPU5 曾出现首次请求停滞，原因未确定；详细结果与限制见 [验证记录](docs/INT4.md#本次验证记录2026-10-22)。

```bash
export SPARK_RESULTS=/data1/t00995472/spark-int4/example
mkdir -p "$SPARK_RESULTS"

# CPU 位级转换与 scale 防精度损失检查
python -B utils/validate_spark_int4.py --packing-only

# 真实权重 INT4/INT8 算子检查，以及中文、算术、代码生成和分块 prefill
python -B utils/validate_spark_int4.py --npu 2 --check-kernels \
  --result "$SPARK_RESULTS/validation.json"

# 短请求检查
python -B benchmark.py --npu 2 --enforce-eager \
  --input-tokens 128 --output-tokens 8 --runs 1 \
  --result "$SPARK_RESULTS/smoke.json"

# 完整模型：默认输入 32768、输出 4096，完整预热一次、测量三次
python -B benchmark.py --npu 2 --enforce-eager \
  --result "$SPARK_RESULTS/benchmark.json"
```

可用 `--model` 指定另一份同格式本地 checkpoint。不再使用 `--w4a16` 开关或 `quant_model_description.json`。benchmark 结果记录配置哈希、插件版本、108 个 INT4 层及 37 个 INT8 层的运行时 dtype 与调用次数。

`utils/run_model_recorded.py --directory 新目录 -- --npu 2 --enforce-eager` 包装同一 benchmark，额外保存源码身份与设备采样；`utils/analyze_spark_int4.py 记录目录` 核对标准 32768/4096、一次预热三次测量的结果。其他参数见各入口 `--help`。

TTFT 为提交请求到首 token 的时间，TPOT 为 `(末 token 时间 − 首 token 时间) / (输出 token 数 − 1)`。计时包含引擎调度与进程通信，排除加载、分词、预热及 HTTP。profiler、logits 导出运行不能作为性能结果。输入使用重复中文文本，生成冒烟检查也不等于模型质量评测。旧 X2.5 的性能数字不适用于新 checkpoint。

## 保留的 BF16 参考算子

`kernels/` 保留 `stage_prefetch` 与 `lossless_prefetch` 两个静态源码及公共头文件，相关原理与历史结果见 [BF16 无损压缩](docs/LOSSLESS_DECODE.md)。它们要求显式 BF16 checkpoint，**不能用于官方混合 INT4/INT8 模型**；benchmark 会在初始化设备前拒绝该组合。官方 INT4 当前使用原生 Ascend 量化算子，无需构建这两个融合库。

```bash
export SPARK_BUILD=/data1/t00995472/spark-builds/bf16-example
export SPARK_BF16_MODEL=/data1/t00995472/models/your-bf16-checkpoint
bash utils/build.sh "$SPARK_BUILD/stage" --kernel stage_prefetch
bash utils/build.sh "$SPARK_BUILD/lossless" --kernel lossless_prefetch

python -B utils/benchmark_kernel.py --kernel lossless_prefetch \
  --model "$SPARK_BF16_MODEL" --build "$SPARK_BUILD/lossless" \
  --npu 2 --layer 3 --length 128 --output "$SPARK_RESULTS/bf16-layer.json"
```

构建目录必须为空，共享库和同目录 `execution.json` 一起部署。BF16 融合要求 TP=PP=1、单请求、BF16 激活与 KV、block size=128、总上下文不超过 40960、eager、关闭前缀缓存和推测解码。滑窗沿用本机 FIA 的 `[pos-512,pos]`，最多 513 个位置，与 Transformers 的 512 位置语义有差异。

## 目录

- `benchmark.py`：完整模型 benchmark 的直接入口。
- `utils/`：官方模型接入、布局转换、构建、验证、单层 benchmark 和分析。
- `kernels/`：两个 BF16 参考 `.asc` 及必需头文件；不动态生成源码。
- `docs/`：官方 INT4 接入与 BF16 无损压缩说明。
- `patches/`：官方 Spark 插件与 vLLM Ascend 的本机兼容补丁。
- `pyproject.toml`：`spark-ascend` 插件安装配置；[AGENTS.md](AGENTS.md) 保存目录与工作约定。
