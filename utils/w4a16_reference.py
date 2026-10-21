"""W4A16 checkpoint oracle and exact fused-kernel packing."""
import json
from safetensors import safe_open
from utils.w4a16_packing import unpack_checkpoint, pack_tile, pack_qkv, interleave_gate_up, pack_outer, OUTER_FORMAT
import torch


def make_reference(model, encoding):
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    lut = "float32" if encoding.endswith("float_lut_v5") else encoding.endswith("lut_v2")
    quantized = {}
    def read(name):
        with safe_open(model / index[name], framework="pt") as f:
            return f.get_tensor(name)
    def load(layer):
        prefix = f"model.layers.{layer}."
        out = {}
        for key in index:
            if not key.startswith(prefix) or key.endswith("weight_scale"):
                continue
            short = key.removeprefix(prefix)
            if key.replace(".weight", ".weight_scale") in index:
                q, scales = unpack_checkpoint(read(key), read(key.replace(".weight", ".weight_scale")))
                quantized[short] = (q, scales)
                out[short] = (q.float() * scales.float().repeat_interleave(128, 1)).bfloat16()
            else:
                out[short] = read(key)
        nextkey = f"model.layers.{layer+1}.input_layernorm.weight" if layer < 35 else "model.norm.weight"
        out[nextkey] = read(nextkey)
        return out, nextkey
    def pack(inputs, layer, rd):
        if encoding == OUTER_FORMAT:
            def outer(q,s): return pack_outer(q,s)
            q,s = quantized['self_attn.q_k_v_proj.weight']
            gate = read(f'model.layers.{layer}.self_attn.g_proj.weight')
            inputs[2] = torch.cat((outer(q,s),gate.contiguous().view(torch.uint8).flatten()))
            inputs[3] = outer(*quantized['self_attn.out_proj.weight'])
            g,gs = quantized['mlp.gate_proj.weight']; u,us = quantized['mlp.up_proj.weight']
            inputs[4] = outer(torch.cat((g,u)),torch.cat((gs,us)))
            inputs[5] = outer(*quantized['mlp.down_proj.weight'])
            return inputs
        q, s = quantized["self_attn.q_k_v_proj.weight"]
        inputs[2] = pack_qkv(q, s, read(f"model.layers.{layer}.self_attn.g_proj.weight"), rd, lut=lut)
        q, s = quantized["self_attn.out_proj.weight"]
        inputs[3] = pack_tile(q, s, 8, lut=lut)
        gq, gs = quantized["mlp.gate_proj.weight"]
        uq, us = quantized["mlp.up_proj.weight"]
        inputs[4] = pack_tile(interleave_gate_up(gq, uq), interleave_gate_up(gs, us), 8, lut=lut)
        q, s = quantized["mlp.down_proj.weight"]
        inputs[5] = pack_tile(q, s, 4, lut=lut)
        return inputs
    return load, pack
