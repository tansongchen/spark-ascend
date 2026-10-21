"""Report BF16 exponent distributions for selected model MLP matrices."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('/data1/t00995472/models/Spark-X2.5-4B'))
    p.add_argument('--layers', type=int, nargs='+', choices=range(36), default=[0, 3, 17, 35])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output path; existing evidence is immutable')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model=a.model;index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];result=[]
    for layer in a.layers:
     for proj in ('gate_proj','up_proj','down_proj'):
      key=f'model.layers.{layer}.mlp.{proj}.weight'
      with safe_open(model/index[key],framework='pt',device='cpu') as f:w=f.get_tensor(key)
      bits=w.view(torch.int16).to(torch.int32)&65535;exp=(bits>>7)&255;hist=torch.bincount(exp.flatten(),minlength=256)
      row={'key':key,'elements':w.numel(),'zero':int((bits&32767==0).sum()),'exponent_histogram':{str(i):int(x) for i,x in enumerate(hist) if x},'blocks':{}}
      for block in (16,32,64,128,256,2560):
       x=exp.reshape(-1,block);width=x.max(1).values-x.min(1).values
       row['blocks'][str(block)]={'fit4_fraction':float((width<16).float().mean()),'fit5_fraction':float((width<32).float().mean()),'range_max':int(width.max())}
      result.append(row);print(key,row['blocks'],flush=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
