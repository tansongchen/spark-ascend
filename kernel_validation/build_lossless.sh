#!/usr/bin/env bash
# Build the standalone lossless variant; never replace an existing build.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-/data1/t00995472/spark-kernel-validation/lossless-prefetch}
source /data1/t00995472/spark/activate_ascend.sh
set -u
unset DEBUG
export ASCEND_SET_ENV=/data1/t00995472/Ascend/cann/set_env.sh
mkdir -p "$out"
python - "$out" <<'PY'
from pathlib import Path
import sys
if any(Path(sys.argv[1]).iterdir()):
    raise SystemExit('Use an empty build directory; existing artifacts are not overwritten.')
PY
name=decode_layer_lossless_prefetch
cp "$root/kernels/$name.asc" "$root/kernels/decode_bf16_lossless_vf.h" \
   "$root/kernels/decode_cube_matmul.h" "$root/kernels/decode_sigmoid_lut_vf.h" "$out/"
bash "$root/.agents/skills/bisheng/build.sh" --arch dav-3510 \
  --kernel "$out/$name.asc" --host "$root/kernel_validation/build_host.c" \
  --out "$out/$name" --objdir "$out/obj"
bisheng --npu-arch=dav-3510 -fPIC -c "$out/$name.asc" -o "$out/device_pic.o"
# Preserve the existing loader ABI and validation-harness library basename.
bisheng --npu-arch=dav-3510 -shared "$out/device_pic.o" \
  -o "$out/libdecode_layer_stage_prefetch.so" -L"$ASCEND_HOME_PATH/lib64" -lascendcl
python - "$out" "$root" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

out, root = map(Path, sys.argv[1:])
files = ['decode_layer_lossless_prefetch.asc', 'decode_bf16_lossless_vf.h',
         'decode_cube_matmul.h', 'decode_sigmoid_lut_vf.h',
         'libdecode_layer_stage_prefetch.so']
manifest = dict(candidate='lossless_mlp_fast_kernels', weight_encoding='bf16_lossless12',
                workspace_bytes=131072, codec_base=109, codec_repairs=256,
                unchanged_abi=True, unchanged_qkv_o_packing=True,
                source='decode_layer_lossless_prefetch.asc',
                target='dav-3510', files_sha256={
                    name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files},
                cpu_codec='kernel_optimization/tpot7/lossless/codec.py',
                cpu_codec_sha256=hashlib.sha256(
                    (root/'kernel_optimization/tpot7/lossless/codec.py').read_bytes()).hexdigest())
(out/'execution.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f'Library and required manifest: {out}')
PY
