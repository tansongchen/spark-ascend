#!/usr/bin/env bash
set -eo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
out=${1:?output under data1}
source /data1/t00995472/spark/activate_ascend.sh
unset DEBUG
export ASCEND_SET_ENV=/data1/t00995472/Ascend/cann/set_env.sh
python "$root/kernel_optimization/tpot7/attention_pipeline/prepare_reuse.py" --output "$out"
export CPATH="$root/kernels${CPATH:+:$CPATH}"
bash "$root/.agents/skills/bisheng/build.sh" --arch dav-3510 --kernel "$out/decode_layer_stage_prefetch.asc" --host "$root/kernel_validation/build_host.c" --out "$out/standalone" --objdir "$out/obj"
bisheng --npu-arch=dav-3510 -fPIC -c "$out/decode_layer_stage_prefetch.asc" -o "$out/device_pic.o"
bisheng --npu-arch=dav-3510 -shared "$out/device_pic.o" -o "$out/libdecode_layer_stage_prefetch.so" -L"$ASCEND_HOME_PATH/lib64" -lascendcl
