"""Check a captured failing model step against independent CPU MLP arithmetic."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, help="Context directory with logits-base/logits-fused")
    p.add_argument("--model", type=Path, default=Path("/data1/t00995472/models/Spark-X2.5-4B-W4A16-G128"))
    p.add_argument("--layers", type=int, nargs="+", default=[18,19,20,21])
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from utils.w4a16_packing import unpack_checkpoint
    torch.set_num_threads(8)
    index = json.loads((a.model/"model.safetensors.index.json").read_text())["weight_map"]
    def read(name):
        with safe_open(a.model/index[name],framework="pt") as f:
            return f.get_tensor(name)
    def load(mode,layer,name):
        return torch.load(a.directory/f"logits-{mode}"/f"layer-{layer:02d}-{name}.pt",weights_only=True).flatten()
    def weight(layer,name):
        prefix=f"model.layers.{layer}.mlp.{name}"
        q,s=unpack_checkpoint(read(prefix+".weight"),read(prefix+".weight_scale"))
        return (q.float()*s.float().repeat_interleave(128,1)).bfloat16()
    def linear(x,w):
        return F.linear(x.float(),w.float()).bfloat16()
    def nrmse(x,y):
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        return float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12))
    rows=[]
    for i in a.layers:
        if i < 1:
            raise ValueError("Residual reconstruction requires layer >=1")
        residual=(load("fused",i-1,"down").float()+load("fused",i-1,"resmid").float()).bfloat16()
        s=load("fused",i,"oproj").float()+residual.float()
        gamma=read(f"model.layers.{i}.post_attention_layernorm.weight").float()
        x=(s*torch.rsqrt(s.square().mean()+1e-6)*gamma).bfloat16()
        gw,uw,dw=weight(i,"gate_proj"),weight(i,"up_proj"),weight(i,"down_proj")
        g,u=linear(x,gw),linear(x,uw)
        mid=(F.gelu(g.float()).bfloat16()*u).bfloat16()
        bg,bu=load("base",i,"gateup").chunk(2)
        bn=load("base",i,"postnorm")
        bmid=(F.gelu(bg.float()).bfloat16()*bu).bfloat16()
        rows.append({
            "layer":i,
            "postnorm_native_vs_fused_nrmse":nrmse(x,bn),
            "fused_mid_vs_own_input_cpu_nrmse":nrmse(load("fused",i,"mid"),mid),
            "fused_down_vs_own_input_cpu_nrmse":nrmse(load("fused",i,"down"),linear(load("fused",i,"mid"),dw)),
            "native_gate_vs_own_input_cpu_nrmse":nrmse(bg,linear(bn,gw)),
            "native_up_vs_own_input_cpu_nrmse":nrmse(bu,linear(bn,uw)),
            "native_down_vs_own_input_cpu_nrmse":nrmse(load("base",i,"down"),linear(bmid,dw)),
        })
    result={"reference":"CPU FP32 GEMM, exact GELU, BF16 boundaries and original INT4 scales",
            "purpose":"Local arithmetic versus accumulated input differences; does not waive full-model checks",
            "layers":rows}
    (a.directory/"local-mlp-reference.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
