#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

export VLLM_USE_V1=1

# ================= data / model / tool =================
# Override via environment variables if your layout differs.
open_agent_rl="${TRAIN_DATA:-${PROJECT_ROOT}/data/proteinllm/desc2seq_agent_grpo_10000.parquet}"
eval_rl="${EVAL_DATA:-${PROJECT_ROOT}/data/proteinllm/desc2seq_agent_eval_10.parquet}"
# SFT checkpoint to start RL from. Export MODEL_PATH if yours is elsewhere.
model_path="${MODEL_PATH:-${PROJECT_ROOT}/results/qwen2.5-7b-desc2seq-clever-ep5}"

train_files="['$open_agent_rl']"
test_files="['$eval_rl']"

# tool config
tool_config_path="${TOOL_CONFIG_PATH:-${PROJECT_ROOT}/recipe/protein/tool_config.yaml}"

# experiment metadata
project_name=protein_grpo
experiment_name=Qwen2.5-7B_simple_reward
default_local_dir="${SAVE_PATH:-${PROJECT_ROOT}/results/qwen2.5-7b-desc2seq-clever-ep5/rl/${experiment_name}}"

# ================= algorithm =================
adv_estimator=grpo

# remove KL divergence
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# clip higher
clip_ratio_low=0.2
clip_ratio_high=0.28

# loss agg
loss_agg_mode="token-mean"

# Dynamic sampling
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# Overlong reward shaping
reward_manager=protein
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

max_turns=16
max_prompt_length=8192
max_response_length=20480
actor_lr=1e-6

train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=8
n_resp_per_prompt_val=1

# ================= performance =================
infer_tp=4 # vllm
train_sp=4 # train
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

# ================= save rollouts =================
ROLLOUT_SAVE_PATH="${default_local_dir}/rollout"
VAL_SAVE_PATH="${default_local_dir}/validation"

mkdir -p "$ROLLOUT_SAVE_PATH" "$VAL_SAVE_PATH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/protein/reward.py \
    data.custom_cls.name=ProteinToolPromptDataset \
    custom_reward_function.path=recipe/protein/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.log_val_generations=20 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@
