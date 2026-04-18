#!/usr/bin/env bash
set -euo pipefail
#
# Usage:
#   bash compute_metrics.sh <input_csv> <output_metrics_csv>
#
# Requires:
#   CONDA_ROOT        -> e.g. /path/to/miniconda3
#   PROTEIN_CHAI_ENV  -> (optional) conda env name/path; defaults to "${CONDA_ROOT}/envs/protein_chai"

if [ "$#" -ne 2 ]; then
  echo "Usage: bash $0 <input_csv> <output_csv>"
  exit 1
fi

INPUT_CSV="$1"
OUTPUT_CSV="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CONDA_ROOT="${CONDA_ROOT:?Please export CONDA_ROOT=/absolute/path/to/miniconda3}"
PROTEIN_CHAI_ENV="${PROTEIN_CHAI_ENV:-${CONDA_ROOT}/envs/protein_chai}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${PROTEIN_CHAI_ENV}"

python "${PROJECT_ROOT}/eval_tools/compute_metrics_multi_gpu.py" \
  --input_csv "$INPUT_CSV" \
  --output_csv "$OUTPUT_CSV"

conda deactivate

echo "[DONE] $OUTPUT_CSV"
