#!/usr/bin/env bash
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source /data1/t00995472/spark/activate_ascend.sh
exec python -B "$root/utils/build_kernel.py" --output "${1:-/data1/t00995472/spark-kernel-validation/build}" "${@:2}"
