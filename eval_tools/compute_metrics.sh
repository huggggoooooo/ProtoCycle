#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash step3_metrics.sh /abs/path/input.csv /abs/path/output_metrics.csv
# 示例：
#   bash step3_metrics.sh \
#     /path/to/ProtoCycle/baseline_results/Qwen2.5_7B_rl_step_10_CAMEO.csv \
#     /path/to/ProtoCycle/baseline_results/Qwen2.5_7B_rl_step_10_CAMEO_metrics.csv

if [ "$#" -ne 2 ]; then
  echo "Usage: bash $0 <input_csv_abs_path> <output_csv_abs_path>"
  exit 1
fi

INPUT_CSV="$1"
OUTPUT_CSV="$2"

# 1) source conda
source /path/to/miniconda3/etc/profile.d/conda.sh

# 2) activate env
conda activate /path/to/miniconda3/envs/protein_chai

# 3) compute metrics
python /path/to/ProtoCycle/eval_tools/compute_metrics_multi_gpu.py \
  --input_csv "$INPUT_CSV" \
  --output_csv "$OUTPUT_CSV"

conda deactivate

echo "[DONE] $OUTPUT_CSV"
