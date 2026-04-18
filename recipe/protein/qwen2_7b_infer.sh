#!/bin/bash
set -xeuo pipefail

# ============ 用户配置 ============
nnodes=1
nproc_per_node=8

# 项目信息
project_name=desc2seq-agentic-rl
experiment_name=qwen2.5-7b-desc2seq-eval

# 模型路径（可以是训练完的 checkpoint，也可以是原模型）
CKPT_PATH=/path/to/ProtoCycle/results/qwen2.5-7b-desc2seq-sft01_ep15_correct2
# 或者用：MODEL_PATH=/path/to/models/Qwen2.5-7B-Instruct

# 评测数据
EVAL_DATA=/path/to/ProtoCycle/data/proteinllm/desc2seq_agentic_conversations_2000_correct.parquet

# 评测结果保存目录
SAVE_PATH=/path/to/ProtoCycle/results/eval_results_$(date +%Y%m%d_%H%M)

mkdir -p "$SAVE_PATH"

# ============ 启动评测 ============
torchrun --nnodes=$nnodes \
  --nproc_per_node=$nproc_per_node \
  -m verl.trainer.main_eval \
  evaluation.model.partial_pretrain=$CKPT_PATH \
  data.path=$EVAL_DATA \
  evaluation.output_dir=$SAVE_PATH \
  evaluation.max_new_tokens=4096 \
  evaluation.batch_size=2 \
  evaluation.temperature=0.2 \
  evaluation.top_p=0.8 \
  evaluation.num_return_sequences=1 \
  evaluation.use_tool_schema=true \
  evaluation.save_outputs=true \
  trainer.project_name=$project_name \
  trainer.experiment_name=$experiment_name \
  trainer.logger='["console","swanlab"]'
