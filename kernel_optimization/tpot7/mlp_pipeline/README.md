# MLP 局部生产与消费实验

目标为完整模型 32K/4K、BF16、单卡单请求的 7.0 ms/token。此目录是独立候选，未修改默认 kernel、模型适配器或安装包。

实现保持 gate/up 的原 GEMV、GELU LUT、BF16 激活边界：每个 AIV 产生其 180/184 个中间通道后，立即计算对应 W2 列块对全部 2560 输出的 FP32 部分和。56 个部分和按固定次序归约，再恢复 BF16 down 输出并执行原最终 norm。没有原 up→down 的全局依赖，新增部分和归约。

ABI 的参数列表不变，输入格式有两项变化，因此不能直接将本库传给原适配器：

- W2 为原 `[2560,10240]` BF16 权重的 `.T.contiguous()`，即 `[10240,2560]`。
- workspace 至少为 704512 字节（原 131072 字节后追加 56×2560×4 字节）。

`prepare.py` 从未修改的默认源码生成候选，`local_down.h` 实现局部 down 和归约，`bench.py` 使用已有 CPU 参考，增加上述 packing、扩展 workspace 和同卡 AB/BA。`build.sh` 调用 Bisheng 技能构建脚本后构建 `.so`。

```bash
bash kernel_optimization/tpot7/mlp_pipeline/build.sh /data1/t00995472/spark-tpot7/mlp_pipeline/local_down8
source /data1/t00995472/spark/activate_ascend.sh
export ASCEND_RT_VISIBLE_DEVICES=6  # 运行前检查空闲，禁止物理 NPU 0
python kernel_optimization/tpot7/mlp_pipeline/bench.py \
  --build /data1/t00995472/spark-tpot7/mlp_pipeline/local_down8 \
  --layer 0 --length 32769 --repeat 3 \
  --output /data1/t00995472/spark-tpot7/mlp_pipeline/local_down8/l0-32769.json
```

2026-10-16 NPU 6 结果：

| 候选 | 长度 | baseline μs | candidate μs | 结果 |
| --- | ---: | ---: | ---: | --- |
| 初版单组外积 | 513 | 164.5112 | 165.0175 | 慢 0.308% |
| 八组展开外积 | 32769 | 164.2102 | 164.2280 | 持平（慢 0.011%） |

两组均通过 CPU 参考（NRMSE≤0.03、cosine≥0.999）、workspace/output guard、重复 3 次；最终 y/resout 与 baseline 逐位一致。基线预热 3000 次，8 轮 baseline/null/candidate 及反向顺序，各组 1000 次。start event 提交后同步当前流，再通过 ctypes 启动原生 kernel。固定单层微测仅筛选，不代表完整模型 TPOT。

为解释无收益，另采集 baseline/candidate 各 3 次 PipeUtilization，核对捕获 6/6。baseline AIV 约 164.8 μs、Vector 35.53 μs、MTE2 148.35 μs；候选 AIV 约 164.46 μs、Vector 35.89 μs、MTE2 149.16 μs。MTE2 活动约占 AIV 时间 90%，新结构未减少主要读取活动。管线可能重叠，不能把各项相加或相减推算气泡，也不能据此宣布 HBM 上限。后续优先减少权重读取量的逐位可逆 BF16 编码，未接入本候选。

证据目录 `/data1/t00995472/spark-tpot7/mlp_pipeline/`；包括 `baseline-check.json`、两个候选的逐阶段数值/ABBA JSON、原始 profile、`profile-summary.json`。初版头文件备份在 `local_down/initial_local_down.h`，当前源码对应 `local_down8`。

本分支随后承担可逆 BF16 编码的独立验证，生产编码/解码实现位于相邻的 `lossless/` 与 `weight_path/`：

- `scan_repairs.py` 扫描全部 36 层，确认 W1 每 tile 最大 60 个异常、W2 最大 234 个异常（layer 11、tile 148）。32/64 修复槽不足；生产布局因此采用 256 槽。
- `verify_model_codec.py` 对全部 72 个实际 W1/W2 矩阵按正式 gate/up 四行交错和 tile packing 编码后解码，逐位比较及 SHA256 全部一致。MLP 字节从 5662310400 降到 4368384000（减少 22.85%）；这是存储字节数，不是性能预测。
- `verify_codec.py` 对两种 tile（20480、40960 元素）各生成 496 个合法测试 tile，覆盖全部 65536 个 BF16 原始码点、修复 count 0..256、首尾及向量边界索引。CPU 与最终融合库的独立 VF 导出均逐位通过；NaN、正负零、无穷大均比较原始 int16，不使用浮点相等。NPU 6 的双侧输出 guard、输入不变与 10 轮复用均通过。CPU 还验证非法容量/头部/index 拒绝，及 base 0、1、108、124、240 的合法窗口。

256 槽结果为 `codec-all-model.json`、`codec256-cpu.json`、`codec256-npu.json`；已验证 CPU codec SHA256 为 `3bcf56d5cdfb4e84eef3aa2b01b8afef27c4e40a435c9680f5ade5f26818bc45`，融合库为 `7840fdf30e6258f48020dac4da20adbb5ead1d5c79fad53b6bddd9600293d3fe`。旧 32 槽码点校验证据保留为 `codec32-cpu.json`、`codec32-npu.json`，不能据此认为 32 槽可覆盖实际完整模型。

```bash
source /data1/t00995472/spark/activate_ascend.sh
python kernel_optimization/tpot7/mlp_pipeline/scan_repairs.py
python kernel_optimization/tpot7/mlp_pipeline/verify_model_codec.py
export ASCEND_RT_VISIBLE_DEVICES=6  # 先查空闲
python kernel_optimization/tpot7/mlp_pipeline/verify_codec.py \
  --library /data1/t00995472/spark-tpot7/weight_path/lossless256/libdecode_layer_stage_prefetch.so \
  --output /data1/t00995472/spark-tpot7/mlp_pipeline/codec256-npu.json
```

快速解包版本也完成独立验收：`lossless_fast/libdecode_layer_stage_prefetch.so` SHA256 为 `d33b8dc1a428a71afa4cd33afb492bd720db6dbaf7c4404c7f161e7c97d40243`，相同全码点/count/guard/重放覆盖通过，证据 `codec256-fast-npu.json`。

为验证解包优化机制，NPU 6 分别采集 baseline/稳定编码、baseline/快速编码各 3 次完整滑窗层（layer 0，长度 32769），每个窗口均核对 6/6 kernel，单层参考和 guard 通过：

| 独立 profile 版本 | AIV 时间 μs | Vector 活动 μs | MTE2 活动 μs |
| --- | ---: | ---: | ---: |
| 稳定编码窗口 baseline | 164.407 | 35.535 | 148.578 |
| 稳定编码 | 150.437 | 105.749 | 116.001 |
| 快速编码窗口 baseline | 164.093 | 35.505 | 148.643 |
| 快速编码 | 143.203 | 79.257 | 116.786 |

快速编码的 Vector 活动比稳定版本少约 26.49 μs，MTE2 活动基本不变，对应 AIV 时间约少 7.23 μs。这是独立 profile 的机制证据；完整模型 TPOT 和验收由主线程记录。原始捕获为 `prof-lossless256/`、`prof-lossless-fast/`，摘要为 `profile-lossless256-summary.json`、`profile-lossless-fast-summary.json`。`profile_lossless.py` 复用正式 packing 和已有 CPU 参考，`--profile <目录>` 开启采集。
