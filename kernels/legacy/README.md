# 原始 mega 算子

`decode_layer_mega.asc` 是用户提供的原始版本，编译成功但完整入口曾超时，未用于完整模型加速。保留原始源码及其哈希，不再作为默认构建目标。当前可用版本为上一级的 `decode_layer_stage_prefetch.asc`。

历史诊断记录见 [单层验证说明](../../kernel_validation/README.md)，数据盘 `/data1/t00995472/spark-kernel-validation/` 保存原始库、提前返回副本及超时日志。手动构建此原始文件时，公共头文件目录为 `kernels/`。
