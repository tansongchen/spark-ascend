"""验证本机兼容补丁的 MLP 分片加载与 RoPE 数值（使用基准脚本的设备预检查）。"""
from types import SimpleNamespace
import faulthandler

from benchmark_spark_vllm import arguments, preflight


def main():
    faulthandler.dump_traceback_later(60, repeat=True)
    args = arguments()
    preflight(args)
    import torch
    import torch_npu  # noqa: F401
    from vllm_spark2_5_plugin.spark2_5 import Spark2_5ForCausalLM
    from vllm_ascend.ops.triton.rope import rope_forward_triton

    # 使用真实 torch 参数验证 gate/up 分片位置与普通参数加载。
    module = torch.nn.Module()
    module.config = SimpleNamespace(tie_word_embeddings=True)
    module.mlp = torch.nn.Module()
    packed = torch.nn.Parameter(torch.zeros(8, 3), requires_grad=False)
    packed.weight_loader = lambda p, w, shard: p.data[shard * 4:(shard + 1) * 4].copy_(w)
    module.mlp.gate_up_proj = torch.nn.Linear(3, 8, bias=False)
    module.mlp.gate_up_proj.weight = packed
    module.norm = torch.nn.LayerNorm(3, bias=False)
    loaded = Spark2_5ForCausalLM.load_weights(module, [
        ('mlp.gate_proj.weight', torch.ones(4, 3)),
        ('mlp.up_proj.weight', torch.full((4, 3), 2.0)),
        ('norm.weight', torch.full((3,), 3.0)),
        ('lm_head.weight', torch.empty(0)),
    ])
    assert loaded == {'mlp.gate_up_proj.weight', 'norm.weight'}
    torch.testing.assert_close(packed[:4], torch.ones(4, 3))
    torch.testing.assert_close(packed[4:], torch.full((4, 3), 2.0))
    torch.testing.assert_close(module.norm.weight, torch.full((3,), 3.0))
    print('MLP 分片加载检查通过', flush=True)

    torch.npu.set_device(0)  # 预检查已将物理卡映射为逻辑 0。
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    init_device_properties_triton()
    torch.manual_seed(7)
    for dim in (64, 256):
        for tokens in (1, 128):
            print(f'RoPE 检查：rotary_dim={dim}, tokens={tokens}', flush=True)
            qkv = torch.randn(tokens, 24, 256, dtype=torch.bfloat16)
            q, k, _ = qkv.split((16, 4, 4), dim=1)
            angles = torch.randn(tokens, dim // 2)
            cos, sin = angles.cos(), angles.sin()
            cache = torch.cat((cos, sin), dim=-1).to(torch.bfloat16)
            cos, sin = cache.float().chunk(2, dim=-1)

            def reference(x):
                x = x.float()
                left, right = x[..., :dim].chunk(2, dim=-1)
                rot = torch.cat((left * cos[:, None] - right * sin[:, None],
                                 right * cos[:, None] + left * sin[:, None]), dim=-1)
                return torch.cat((rot, x[..., dim:]), dim=-1).to(torch.bfloat16)

            expected = reference(q), reference(k)
            actual = rope_forward_triton(qkv.npu()[:, :16], qkv.npu()[:, 16:20],
                                        cos_sin_cache=cache.npu(),
                                        positions=torch.arange(tokens, device='npu'), rope_dim=dim)
            for value, ref in zip(actual, expected):
                torch.testing.assert_close(value.cpu(), ref, atol=0.03125, rtol=0.016)
    print('MLP 分片加载、完整/部分 RoPE、单 token/多 token 数值检查通过')
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
