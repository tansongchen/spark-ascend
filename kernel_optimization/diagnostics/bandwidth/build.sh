#!/usr/bin/env bash
set -eo pipefail
source /data1/t00995472/spark/activate_ascend.sh
unset DEBUG
set -u
root=$(cd "$(dirname "$0")/../../.." && pwd)
src="$root/kernel_optimization/diagnostics/bandwidth"
out=${1:-/data1/t00995472/spark-bf16-opt/bandwidth-build}
name=stream_read
mkdir -p "$out"
export ASCEND_SET_ENV=/data1/t00995472/Ascend/cann/set_env.sh
bash "$root/.agents/skills/bisheng/build.sh" --arch dav-3510 --kernel "$src/$name.asc" --host "$root/kernel_validation/build_host.c" --out "$out/probe_$name" --objdir "$out/obj"
bisheng --npu-arch=dav-3510 -fPIC -c "$src/$name.asc" -o "$out/pic_$name.o"
bisheng --npu-arch=dav-3510 -shared "$out/pic_$name.o" -o "$out/lib$name.so" -L"$ASCEND_HOME_PATH/lib64" -lascendcl
