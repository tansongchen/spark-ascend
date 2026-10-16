#!/usr/bin/env bash
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-/data1/t00995472/spark-kernel-validation/build}
source /data1/t00995472/spark/activate_ascend.sh
set -u
unset DEBUG
export ASCEND_SET_ENV=/data1/t00995472/Ascend/cann/set_env.sh
mkdir -p "$out"
for name in decode_layer_mega decode_layer_stage_prefetch; do
  bash "$root/.agents/skills/bisheng/build.sh" --arch dav-3510 --kernel "$root/$name.asc" --host "$root/kernel_validation/build_host.c" --out "$out/$name" --objdir "$out/$name-obj"
  bisheng --npu-arch=dav-3510 -fPIC -c "$root/$name.asc" -o "$out/$name-obj/device_${name}_pic.o"
  bisheng --npu-arch=dav-3510 -shared "$out/$name-obj/device_${name}_pic.o" -o "$out/lib$name.so" -L"$ASCEND_HOME_PATH/lib64" -lascendcl
done
