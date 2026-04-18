#!/bin/bash
source /path/to/miniconda3/etc/profile.d/conda.sh
cd /path/to/ProtoCycle
conda activate /path/to/miniconda3/envs/OpenAgentRL

export SWANLAB_API_KEY="TBDQtjpnnI095QKDuzhrP"
swanlab login
export VLLM_USE_V1=1

nnodes=1
nproc_per_node=8
project_name=desc2seq-agentic-rl-no_reflect
experiment_name=qwen2.5-7b-desc2seq-no_reflect-ep5
DATA_ROOT=${DATA_ROOT:-$PWD}
TRAIN_DATA=/path/to/ProtoCycle/data/proteinllm/desc2seq_agentic_conversations_2000_str2.parquet
EVAL_DATA=/path/to/ProtoCycle/data/proteinllm/desc2seq_agentic_conversations_2000_str2.parquet
MODEL_PATH=/path/to/models/Qwen2.5-7B-Instruct
SAVE_PATH=/path/to/ProtoCycle/results/$experiment_name

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