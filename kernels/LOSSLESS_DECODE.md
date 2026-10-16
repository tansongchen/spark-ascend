# BF16 无损编码融合 decode

`decode_layer_lossless_prefetch.asc` 是达到约 6.82 ms/token 的 `lossless_fast` 已验收版本；`decode_bf16_lossless_vf.h` 是其精简整数解码器。两者从已验证产物整理，计算代码保持一致，只添加说明并修改本地解码头的 include 名称。

依赖同目录已有的 `decode_cube_matmul.h`、`decode_sigmoid_lut_vf.h` 和本机 CANN 头文件，不需要运行实验目录的源码生成器。保留独立 `launch_decode_tile` 入口用于无损解码验证。

## 接口与布局

与原 `decode_layer_stage_prefetch.asc` 使用相同的 `launch_decode_layer_stage_prefetch` ABI，因此两份 `.asc` 应分别编译成独立库，不能一起链接。

- 参数数组 `a[4]` / `a[5]` 分别为 MLP up / down 的**编码权重**，不能直接传入原始 BF16 packing。
- CPU 打包器为 `kernel_optimization/tpot7/lossless/codec.py::pack_weight`，输入先经过原 `tile_weights`。up 使用 `(K=2560,M=8)`，down 使用 `(K=10240,M=4)`。
- 固定 base=109、256 个异常修补项。每 tile 的编码字节数分别为 31776 / 62496；计算前逐位还原完整 BF16，没有量化。
- QKV、O、其他参数与 128 KiB workspace 保持原合同。仅支持 BF16、TP=PP=1、单请求、block=128、context≤40960、融合 eager，关闭前缀缓存与推测解码。
- `integration/spark_fused_decode.py` 读取库旁的 `execution.json` 后自动选择正确打包。复制库时必须同时携带该 manifest。

## 构建与使用

构建脚本只写入新的空目录，不修改默认 kernel 或已有产物；同时保存源码、依赖头和文件哈希。所有构建产物放在数据盘。

```bash
bash kernel_validation/build_lossless.sh \
  /data1/t00995472/spark-kernel-validation/lossless-prefetch

source /data1/t00995472/spark/activate_ascend.sh
npu-smi info
# 确认物理 NPU 2 空闲；禁止使用物理 NPU 0
python -B benchmark_spark_vllm.py --npu 2 --enforce-eager --fused-decode \
  --fused-library /data1/t00995472/spark-kernel-validation/lossless-prefetch/libdecode_layer_stage_prefetch.so \
  --result /data1/t00995472/spark-tpot7/new-kernels-lossless.json
```

原始性能、逐位精度与限制见 [完整实验记录](../docs/TPOT7_OPTIMIZATION.md)。当前模型适配器保留原重排权重，额外持有约 4.07 GiB 编码权重；首次 CPU 打包属于初始化开销。源码整理本身不代表新增性能收益。

## 本次整理检查

已在本机编译通过，产物为 `/data1/t00995472/spark-kernel-validation/lossless-prefetch/`。去掉新增说明并还原 include 名称后，两个文件的计算主体与已验收源码逐字一致。

新旧设备 ELF 的 `.text`、重定位、kernel 元数据等 13 个段逐字节一致；解析后的符号表仅有源码文件名变化。整个二进制哈希因此不同，不能将新库冒充为旧库。检查记录为构建目录的 `organization-verification.json`，本次未重新运行性能测量。
