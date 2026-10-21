"""Select an explicitly idle nonzero physical NPU before importing torch_npu."""
import os
import subprocess


def select_idle_npu(physical):
    if not 1 <= physical <= 7:
        raise ValueError('本机只允许使用物理 NPU 1～7，禁止 NPU 0')
    env = os.environ.copy()
    env.pop('ASCEND_RT_VISIBLE_DEVICES', None)
    info = subprocess.check_output(['npu-smi', 'info'], text=True, env=env)
    if f'No running processes found in NPU {physical}' not in info:
        raise RuntimeError(f'NPU {physical} 未确认空闲，请先检查占用并选择空闲非 0 卡')
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(physical)
