#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

nnodes=1
nproc_per_node=8
project_name=desc2seq-agentic-rl-clever
experiment_name=qwen2.5-7b-desc2seq-clever-ep5

# -------- data / model / output --------
# Override any of these via environment variables if your layout differs.
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/proteinllm/desc2seq_agentic_clever_1000.parquet}"
EVAL_DATA="${EVAL_DATA:-${PROJECT_ROOT}/data/proteinllm/desc2seq_agentic_clever_1000.parquet}"
# External base model (HF snapshot). Set MODEL_PATH before running.
MODEL_PATH="${MODEL_PATH:?Please export MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct}"
SAVE_PATH="${SAVE_PATH:-${PROJECT_ROOT}/results/${experiment_name}}"

CUSTOM_CLS_PATH="${PROJECT_ROOT}/recipe/protein/protein_dataset.py"

cd "${PROJECT_ROOT}"

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
    use_remove_padding=true \
    data.custom_cls.path=${CUSTOM_CLS_PATH} \
    data.custom_cls.name=ParquetJSONAdapter \
    $@
