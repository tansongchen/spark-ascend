"""Write an additive complete binary/source identity without editing old evidence."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('build',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[3]
files={name:hashlib.sha256((a.build/name).read_bytes()).hexdigest() for name in ('decode_layer_stage_prefetch.asc','lossless_vf.h','libdecode_layer_stage_prefetch.so','execution.json')}
result=dict(files=files,baseline_source_sha256=hashlib.sha256((root/'kernels/decode_layer_stage_prefetch.asc').read_bytes()).hexdigest(),generated_source_sha256=files['decode_layer_stage_prefetch.asc'],decoder_header_sha256=files['lossless_vf.h'],shared_library_sha256=files['libdecode_layer_stage_prefetch.so'])
(a.build/'identity.json').write_text(json.dumps(result,indent=2)+'\n')
