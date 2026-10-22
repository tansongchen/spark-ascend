# 仓库目录说明

本仓库提供 Spark-X2.6-4B-INT4 在 Ascend 950PR 上的官方量化模型接入和 benchmark，并保留独立 BF16 融合 decode 参考算子。

| 路径 | 职责 |
| --- | --- |
| [README.md](README.md) | 仓库目的、环境安装、构建与使用方法 |
| [benchmark.py](benchmark.py) | 完整模型 TTFT/TPOT benchmark 的直接入口 |
| `kernels/` | BF16 参考 `stage_prefetch`、`lossless_prefetch` 两版静态 `.asc` 及必需头文件 |
| `utils/` | 模型接入、插件注册、官方量化布局转换、构建、验证和分析工具 |
| [docs/LOSSLESS_DECODE.md](docs/LOSSLESS_DECODE.md) | BF16 无损压缩的思路、结果和验证方法 |
| [docs/INT4.md](docs/INT4.md) | 官方 mixed INT4/INT8 格式、Ascend 接入与验证 |
| `patches/` | Spark 插件与 vLLM Ascend 的兼容及注册补丁 |
| [pyproject.toml](pyproject.toml) | 本地插件的可编辑安装配置，Python 包为 `utils` |

## 算子开发规范

以下规范适用于后续所有算子开发工作：

1. 在 `kernels/` 中编写可直接编译的 `.h` 和 `.asc` 静态源码，不在构建或运行时通过字符串拼接生成。
2. 在 `docs/` 中编写对应算子的思路说明。
3. 除根目录的完整模型 `benchmark.py` 外，构建、验证、单层 benchmark、分析等脚本和工具统一放在 `utils/` 中，直接调用各自入口。写新的脚本前应该先通读这个目录，尽可能复用或修改现有脚本，避免重复实现功能。

## 整理约定

- 始终使用中文回复。
- 当前项目文档为上述三篇；后续新算子的思路说明放在 `docs/`，本文件仅说明目录与工作约定。
- 模型、缓存、构建产物和实验结果放在 `/data1/t00995472/`，不放进仓库。
- 本仓库是插件的可编辑安装源；移动仓库或调整模块路径时，同步更新安装和注册入口。
- 运行环境、设备选择和具体命令统一参照 README。
