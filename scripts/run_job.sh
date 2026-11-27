#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
workspace_dir="${ANOMALYDINO_WORKSPACE:-$project_dir}"
stage="${1:?extract or evaluate}"
config="${2:?config path}"
output="${3:?new output directory}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="$workspace_dir/work/matplotlib"
case "$stage" in
  extract) script=extract_features.py ;;
  evaluate) script=evaluate.py ;;
  *) exit 2 ;;
esac
"$workspace_dir/work/venv/bin/python" -u "$project_dir/scripts/$script" --config "$config" --output "$output"
