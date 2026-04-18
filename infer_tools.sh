#!/usr/bin/env bash
# set -euo pipefail
#
# End-to-end evaluation pipeline:
#   1) infer_tools.py   -> JSONL of multi-turn dialogs with tool calls
#   2) extract_answer   -> CSV with final <answer> sequences
#   3) compute_metrics  -> metrics CSV (ProTrek / Chai / etc.)
#
# Configure paths via environment variables or edit the defaults below.

########################
#      Config          #
########################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"

# Conda (external). Export CONDA_ROOT before running.
CONDA_ROOT="${CONDA_ROOT:?Please export CONDA_ROOT=/absolute/path/to/miniconda3}"
OPENAGENT_ENV="${OPENAGENT_ENV:-${CONDA_ROOT}/envs/OpenAgentRL}"
PROTEIN_CHAI_ENV="${PROTEIN_CHAI_ENV:-${CONDA_ROOT}/envs/protein_chai}"

# Base model or RL checkpoint to evaluate. Export MODEL_DIR before running.
MODEL_DIR="${MODEL_DIR:?Please export MODEL_DIR=/absolute/path/to/checkpoint}"
MODEL_NAME="${MODEL_NAME:-model}"

# Results directory (inside repo by default)
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/baseline_results}"
mkdir -p "${RESULTS_DIR}/metrics"

JSONL="${JSONL:-${RESULTS_DIR}/${MODEL_NAME}.jsonl}"
CSV="${CSV:-${RESULTS_DIR}/${MODEL_NAME}.csv}"
METRICS="${METRICS:-${RESULTS_DIR}/metrics/${MODEL_NAME}_metrics.csv}"

# Optional extra args for compute_metrics_multi_gpu (e.g. --skip_chai)
METRICS_EXTRA_ARGS=()

########################
#      Pipeline        #
########################

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
cd "${PROJECT_ROOT}"

echo "[Step 1] Inference: infer_tools.py"
conda activate "${OPENAGENT_ENV}"
python "${PROJECT_ROOT}/infer_tools.py" \
  --model_dir "${MODEL_DIR}" \
  --out "${JSONL}"

echo "[Step 2] Extract answers: extract_answer.py"
python "${PROJECT_ROOT}/eval_tools/extract_answer.py" \
  --jsonl "${JSONL}" \
  --out_csv "${CSV}"

conda deactivate

echo "[Step 3] Compute metrics: compute_metrics_multi_gpu.py"
conda activate "${PROTEIN_CHAI_ENV}"

python "${PROJECT_ROOT}/eval_tools/compute_metrics_multi_gpu.py" \
  --input_csv "${CSV}" \
  --output_csv "${METRICS}" \
  "${METRICS_EXTRA_ARGS[@]}"

conda deactivate

echo "[DONE] metrics -> ${METRICS}"
