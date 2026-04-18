#!/bin/bash
# set -x
source /path/to/miniconda3/etc/profile.d/conda.sh
cd /path/to/ProtoCycle
conda activate /path/to/miniconda3/envs/OpenAgentRL

export SWANLAB_API_KEY="TBDQtjpnnI095QKDuzhrP"
swanlab login
export VLLM_USE_V1=1

# ================= 配置 =================
nnodes=1
nproc_per_node=4
project_name=desc2seq-agentic-rl-no_reflect
experiment_name=llama3_1-8b-desc2seq-no_reflect-ep5
DATA_ROOT=${DATA_ROOT:-$PWD}
TRAIN_DATA=/path/to/ProtoCycle/data/proteinllm/desc2seq_agentic_conversations_2000_str2.parquet
EVAL_DATA=/path/to/ProtoCycle/data/proteinllm/desc2seq_agentic_conversations_2000_str2.parquet
MODEL_PATH=/path/to/models/Meta-Llama-3.1-8B-Instruct
SAVE_PATH=/path/to/ProtoCycle/results/$experiment_name

# Infer 配置
CONDA_ROOT="/path/to/miniconda3"
OPENAGENT_ENV="${CONDA_ROOT}/envs/OpenAgentRL"
PROTEIN_CHAI_ENV="${CONDA_ROOT}/envs/protein+chai"
PROJECT_ROOT="/path/to/ProtoCycle"
RESULTS_DIR="${PROJECT_ROOT}/baseline_results"
MODEL_NAME="Llama3_1_8B_no_reflect"
JSONL="${RESULTS_DIR}/${MODEL_NAME}.jsonl"
CSV="${RESULTS_DIR}/${MODEL_NAME}.csv"
METRICS="${RESULTS_DIR}/${MODEL_NAME}_metrics.csv"

# ================= Step 1: SFT 训练 =================
torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=32768 \
    data.truncation=right \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","swanlab"]' \
    trainer.total_epochs=5 \
    trainer.save_freq=1000 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true\
    data.custom_cls.path=/path/to/ProtoCycle/recipe/protein/protein_dataset.py \
    data.custom_cls.name=ParquetJSONAdapter \

# ================= Step 2: Merge 模型 =================
echo "[Step 2] Merge 模型..."

# 找到最新的 checkpoint
LATEST_CKPT=$(ls -td $SAVE_PATH/global_step_* 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "Error: No checkpoint found in $SAVE_PATH"
    exit 1
fi

echo "Using checkpoint: $LATEST_CKPT"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CKPT \
    --target_dir $SAVE_PATH/huggingface

MERGED_MODEL_DIR=$SAVE_PATH/huggingface

# ================= Step 3: Inference =================
echo "[Step 3] Inference..."

# Step 3.1: 推理
python infer_tools.py \
  --model_dir "$MERGED_MODEL_DIR" \
  --out "$JSONL"

# Step 3.2: 抽取 answer
cd "${PROJECT_ROOT}/eval_tools"
python "${PROJECT_ROOT}/eval_tools/extract_answer.py" \
  --jsonl "${JSONL}" \
  --out_csv "${CSV}"

# ================= Step 4: 计算指标 =================
echo "[Step 4] 计算指标..."
conda activate "${PROTEIN_CHAI_ENV}"

python "${PROJECT_ROOT}/eval_tools/compute_metrics_multi_gpu.py" \
  --input_csv "${CSV}" \
  --output_csv "${METRICS}"

conda deactivate

echo "[完成] 指标已写入: ${METRICS}"