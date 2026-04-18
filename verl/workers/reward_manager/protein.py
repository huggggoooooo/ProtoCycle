# Copyright 2025
# ProteinRewardManager for protein design RL in Verl

from collections import defaultdict
from typing import Any, Dict

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# 你自己的评分函数，按我们之前约定的版本，注意它们要支持:
# compute_score_protein(data_source, solution_str, reward_model, extra_info) -> {"score": float, "pred": str}
# compute_score_protein_outcome_reward(data_source, solution_str, reward_model, extra_info) -> {"score": float, "pred": str}

from recipe.protein.reward import compute_score




@register("protein")
class ProteinRewardManager(AbstractRewardManager):
    """
    ProteinRewardManager:
    - API and behavior mirrors DAPORewardManager, so trainer code doesn't change.
    - But scoring logic is for protein design, not math/code.

    What it does per sample:
      1. Decode prompt and generated response (like DAPO).
      2. Pull reference sequence from reward_model["ground_truth"].
      3. Call compute_score(data_source, response_str, ground_truth, extra_info).
      4. Drop that reward on the last valid response token.
      5. Optionally apply overlong penalty like DAPO.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        step_coef: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Arguments are intentionally aligned with DAPORewardManager so that
        load_reward_manager() can pass the same kwargs.

        tokenizer:            tokenizer to decode ids to text
        num_examine:          how many debug samples to print per data_source
        compute_score:        callable(data_source, solution_str, ground_truth, extra_info)
                              If None, we'll use our default protein_compute_score.
                              IMPORTANT: we will wrap it to also pass step_coef.
        reward_fn_key:        which key in non_tensor_batch identifies "task" or "domain"
        max_resp_len:         used for length penalty (matches DAPO semantics)
        overlong_buffer_cfg:  same structure DAPO expects (len, penalty_factor, enable, log)
        step_coef:            weight for shaping reward (plan/think/answer compliance)
        kwargs:               swallow any future args without crashing
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.step_coef = step_coef
        self._extra_kwargs = kwargs  # just keep for debugging if needed

        # If trainer didn't pass a scorer, default to our protein scorer.
        # We wrap protein_compute_score so it's signature-compatible with dapo
        # (i.e. manager will call scorer(data_source, solution_str, ground_truth, extra_info)).
        if compute_score is None:
            def _scorer(data_source, solution_str, ground_truth, extra_info):
                return compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    step_coef=self.step_coef,
                )
            self.compute_score = _scorer
        else:
            # If trainer *did* pass a compute_score, we assume it follows the dapo convention:
            # compute_score(data_source, solution_str, ground_truth, extra_info) -> dict or float
            self.compute_score = compute_score

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Mirror DAPORewardManager.__call__.

        Input: DataProto batch with:
            batch["prompts"], batch["responses"], batch["attention_mask"]
            non_tensor_batch["reward_model"]["ground_truth"] (string AA sequence)
            non_tensor_batch[self.reward_fn_key] (like "protein_design_stage3")
            non_tensor_batch["extra_info"]

        Output:
            If return_dict=False:
                reward_tensor  (FloatTensor [batch, seq_len])
            If return_dict=True:
                {
                  "reward_tensor": reward_tensor,
                  "reward_extra_info": reward_extra_info_dict
                }
        """

        # Shortcut: external rm_scores wins (same as DAPO)
        # breakpoint()
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch[key] for key in reward_extra_keys
                }
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return data.batch["rm_scores"]

        # Initialize
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        printed_per_source: Dict[str, int] = {}

        # Iterate each item in the batch
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids    = data_item.batch["prompts"]
            response_ids  = data_item.batch["responses"]
            attn_mask     = data_item.batch["attention_mask"]

            prompt_len = prompt_ids.shape[-1]

            # figure out valid prompt / response lengths (like DAPO)
            valid_prompt_len = attn_mask[:prompt_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            valid_resp_len = attn_mask[prompt_len:].sum()
            valid_resp_ids = response_ids[:valid_resp_len]

            # decode text
            prompt_str   = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)

            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            # non-tensor metadata
            ntb = data_item.non_tensor_batch

            # training data needs to have "reward_model": {"ground_truth": "..."}
            reward_model = ntb["reward_model"]
            ground_truth_seq = reward_model.get("ground_truth", "")

            data_source = ntb.get(
                self.reward_fn_key,
                ntb.get("data_source", "protein_design_stage3"),
            )
            extra_info = ntb.get("extra_info", {})

            # compute score dict using our scorer
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth_seq,
                extra_info=extra_info,
            )

            # Accept dict or raw float, same as DAPO
            if isinstance(result, dict):
                score_val      = float(result["score"])
                step_score_val = float(result.get("step_score", 0.0))
                out_score_val  = float(result.get("outcome_score", score_val))
                pred_text      = result.get("pred_text", "")
                pred_seq       = result.get("pred_seq", "")
            else:
                score_val      = float(result)
                step_score_val = 0.0
                out_score_val  = score_val
                pred_text      = ""
                pred_seq       = ""

            # Overlong penalty (same contract as DAPO)
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_resp_len - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(
                    -exceed_len / overlong_buffer_len * overlong_penalty_factor,
                    0,
                )
                score_val += overlong_reward

                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # Put reward at last valid token of the response (Dapo convention)
            reward_tensor[i, valid_resp_len - 1] = score_val

            # For logging
            reward_extra_info["final_reward"].append(score_val)
            reward_extra_info["step_score"].append(step_score_val)
            reward_extra_info["outcome_score"].append(out_score_val)
            reward_extra_info["pred_text"].append(pred_text)
            reward_extra_info["pred_seq"].append(pred_seq)
            reward_extra_info["data_source"].append(data_source)

            # Print a few debug lines per data_source (like DAPO)
            if data_source not in printed_per_source:
                printed_per_source[data_source] = 0
            if printed_per_source[data_source] < self.num_examine:
                printed_per_source[data_source] += 1
                print("====== [ProteinRewardManager sample dump] ======")
                print("[prompt]        ", prompt_str)
                print("[response]      ", response_str)
                print("[ground_truth]  ", ground_truth_seq)
                print("[step_score]    ", step_score_val)
                print("[outcome_score] ", out_score_val)
                print("[final_reward]  ", score_val)
                print("[pred_seq]      ", pred_seq)
                print("===============================================")

        # match DAPO's return contract
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
