"""Compare teacher-forced full-model logits; run under the target venv on CPU."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
a=p.parse_args();root=a.directory
torch.set_num_threads(8)
base=root/'logits-base';fused=root/'logits-fused'
files=sorted(base.glob('step-*.pt'))
if not files or {x.name for x in files}!={x.name for x in fused.glob('step-*.pt')}:
    raise RuntimeError('Logit capture counts differ or empty')
rows=[]
for path in files:
    b=torch.load(path,weights_only=True).float().flatten()
    f=torch.load(fused/path.name,weights_only=True).float().flatten()
    if b.shape!=f.shape or not torch.isfinite(f).all():
        raise RuntimeError(f'Invalid fused logits: {path.name}')
    topb=b.topk(5).indices;topf=f.topk(5).indices
    rows.append({'step':int(path.stem.split('-')[1]),
                 'nrmse':float((b-f).norm()/b.norm()),
                 'cosine':float(F.cosine_similarity(b[None],f[None])),
                 'max_abs':float((b-f).abs().max()),
                 'top1_agree':bool(topb[0]==topf[0]),
                 'top5_overlap':len(set(topb.tolist())&set(topf.tolist()))/5,
                 'baseline_top1':int(topb[0]),'fused_top1':int(topf[0]),
                 'baseline_top1_margin':float(b.topk(2).values.diff().abs()[0])})
decode=[r for r in rows if r['step']>0]
out={'steps':rows,'decode_steps':len(decode),
     'decode_top1_agreement':sum(r['top1_agree'] for r in decode)/len(decode),
     'decode_max_nrmse':max(r['nrmse'] for r in decode),
     'decode_min_cosine':min(r['cosine'] for r in decode),
     'decode_mean_top5_overlap':sum(r['top5_overlap'] for r in decode)/len(decode),
     'reference':'same prompt, forced baseline token prefix; logits captured before forcing',
     'quality_benchmark':False}
(root/'logit-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='steps'},indent=2))
