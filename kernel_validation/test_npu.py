# test_npu.py

import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())

x = torch.randn(128, 128, device="npu")
y = torch.randn(128, 128, device="npu")

z = x @ y

print("device:", z.device)
print("shape:", z.shape)
print("result:", z[0, :5])
