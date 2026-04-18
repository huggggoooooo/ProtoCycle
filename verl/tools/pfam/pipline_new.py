#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
from typing import Any, Dict, List, Optional


import os
import subprocess
from typing import List
import ray
import json
import os
from typing import List, Dict, Any
import requests  # 记得在模块顶部 import

PROTREK_API_URL = os.getenv(
    "PROTREK_API_URL",
    "http://101.126.67.113:8863/protrek_score_35m",  # 改成你真实地址
)


class AgentRuntime:
    """
    A lightweight executor that replays tool_calls one by one.

    新版特性（为 SFT / agent 环境准备）：
    - 每个 AgentRuntime 对应一次 desc2seq 设计任务（一个 requirement_text）。
    - 在 work_dir 下维护所有中间文件（scaffold.json, *_constraints.json, inpaint_results.json, tmp_ranked.json 等）。
    - 每次工具调用后：
        * 调用真实脚本生成/更新序列或约束；
        * 用 _caculate_protrek_score_mini 对“本轮新序列”做 ProTrek 打分；
        * 维护全局 state（round, best_score, best_seq 等）；
        * 返回统一风格的 observation 文本，给 LLM 当 tool 输出。
    """

    def __init__(
        self,
        work_dir: str,
        requirement_text: Optional[str] = None,
        protrek_env_python: str = "/path/to/miniconda3/envs/protrek/bin/python",
    ):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        # canonical file paths that form the "blackboard"
        self.scaffold_json = os.path.join(self.work_dir, "scaffold.json")
        self.inpaint_results_json = os.path.join(self.work_dir, "inpaint_results.json")

        # 旧的 stage1/final ranked 文件（保留路径，但不再用 stage1_ranked.json / similarity_score 逻辑）
        self.ranked_json = os.path.join(self.work_dir, "ranked.json")
        self.stage1_ranked_json = os.path.join(self.work_dir, "stage1_ranked.json")

        # ProTrek mini scoring 临时文件 & 全局 top_k 文件
        self.tmp_scoring_json = os.path.join(self.work_dir, "tmp_for_scoring.json")
        self.mini_ranked_json = os.path.join(self.work_dir, "mini_ranked.json")
        self.tmp_ranked_json = os.path.join(self.work_dir, "tmp_ranked.json")

        self.protrek_env_python = protrek_env_python
        self.default_topk = 3   # how many entries to keep globally at most

        # 全局 state（对同一个 requirement）
        self.state: Dict[str, Any] = {
            "requirement_text": requirement_text or "",
            "round": 0,
            "best_score": float("-inf"),
            "prev_best": float("-inf"),
            "best_accession": None,
            "best_seq": None,
        }

        # script registry: siblings inside verl/tools/pfam/
        _PFAM_DIR = os.path.dirname(os.path.abspath(__file__))
        self.scripts = {
            "function2seq": os.path.join(_PFAM_DIR, "function2seq.py"),
            "pathway2seq": os.path.join(_PFAM_DIR, "pathway2seq.py"),
            "dna_binding2seq": os.path.join(_PFAM_DIR, "dna_binding2seq.py"),
            "domain2seq": os.path.join(_PFAM_DIR, "domain2seq.py"),
            "go2seq": os.path.join(_PFAM_DIR, "go2seq.py"),

            "build_constraints_from_uniprot": os.path.join(_PFAM_DIR, "build_constraints_from_uniprot.py"),
            "cofactor2constraints": os.path.join(_PFAM_DIR, "cofactor2constraints.py"),
            "motif2constraints": os.path.join(_PFAM_DIR, "motif2constraints.py"),
            "signal2constraints": os.path.join(_PFAM_DIR, "signal2constraints.py"),

            "esm_inpaint": os.path.join(_PFAM_DIR, "esm/esm_constrain.py"),

            # ProTrek scoring script (reuses caculate_similarity_text_seq_35M.py)
            "similarity_score": os.path.join(_PFAM_DIR, "ProTrek/caculate_similarity_text_seq_35M.py"),
        }

        # map tool name -> handler function
        self.handlers = {
            # Stage1-like generators
            "function2seq": self._handle_scaffold_generator,
            "pathway2seq": self._handle_scaffold_generator,
            "dna_binding2seq": self._handle_scaffold_generator,
            "domain2seq": self._handle_scaffold_generator,
            "go2seq": self._handle_scaffold_generator,

            # 显式 get_score：对当前所有已知序列再做一轮聚合打分
            "get_score": self._handle_get_score,

            # constraints builders
            "cofactor2constraints": self._handle_update_constraints,
            "motif2constraints": self._handle_update_constraints,
            "signal2constraints": self._handle_update_constraints,

            # esm inpaint
            "esm_inpaint": self._handle_esm_inpaint,
        }

    # --------------- small utilities ---------------

    def set_requirement_text(self, text: str):
        """可以在 init 后再设置/覆盖 requirement_text。"""
        self.state["requirement_text"] = text or ""

    def _run_subprocess(
        self,
        cmd: List[str],
        use_protrek_env: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        If use_protrek_env=True, we replace leading 'python' with self.protrek_env_python
        """
        if use_protrek_env and len(cmd) > 0 and cmd[0] == "python":
            cmd = [self.protrek_env_python] + cmd[1:]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result

    def _load_json_if_exists(self, path: str, default: Any) -> Any:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    def _save_json(self, path: str, obj: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # helpful for constraints
    def _constraints_path_for_accession(self, accession: str) -> str:
        return os.path.join(self.work_dir, f"{accession}_constraints.json")
    
    def _build_error_observation(
        self,
        tool_name: str,
        error_message: str,
        args: Dict[str, Any],
    ) -> str:
        """
        在工具执行失败 / 未产生候选时构造一个 observation：
        - 不改变 best_score
        - num_scored = 0, delta_best = 0
        - 明确写出：哪个工具 + 用了什么参数 + 因此没产生新序列
        """

        round_idx = int(self.state.get("round", 0))

        # 当前全局 best_score
        best = float(self.state.get("best_score", float("-inf")))
        best_val = None if best == float("-inf") else best

        # 这些字段主要是为了 Summary 一致性
        num_scored = 0
        delta_best = 0.0

        # 不同工具的“主参数”字段名
        primary_arg_map = {
            # scaffold
            "function2seq": ("text", "text"),
            "pathway2seq": ("pathway", "pathway"),
            "dna_binding2seq": ("text", "text"),
            "domain2seq": ("domain", "domain"),
            "go2seq": ("go_term", "go_term"),
            # constraints
            "motif2constraints": ("motif", "motif"),
            "cofactor2constraints": ("cofactor", "cofactor"),
            # signal2constraints 没有主参数
        }

        # 取出主参数（如果有）
        arg_key, arg_label = primary_arg_map.get(tool_name, (None, None))
        primary_val = None
        if arg_key is not None:
            raw_val = args.get(arg_key)
            if isinstance(raw_val, str):
                primary_val = raw_val.strip()
            elif raw_val is not None:
                primary_val = str(raw_val)

        # 太长就截断一下，避免 observation 爆长
        if primary_val is not None and len(primary_val) > 120:
            primary_val = primary_val[:117] + "..."

        scaffold_tools = {
            "function2seq",
            "pathway2seq",
            "dna_binding2seq",
            "domain2seq",
            "go2seq",
        }
        constraint_tools = {
            "cofactor2constraints",
            "motif2constraints",
            "signal2constraints",
        }

        # 第一行自然语言：点名工具 + 参数 + 失败
        if tool_name in scaffold_tools:
            if primary_val is not None:
                first = (
                    f"Round {round_idx}: scaffold-generation tool '{tool_name}' was called "
                    f"with {arg_label}=\"{primary_val}\", but this call did not retrieve any "
                    f"candidate sequences, so no new sequences were scored in this round."
                )
            else:
                first = (
                    f"Round {round_idx}: scaffold-generation tool '{tool_name}' was called, "
                    f"but it did not retrieve any candidate sequences, so no new sequences "
                    f"were scored in this round."
                )
        elif tool_name in constraint_tools:
            if primary_val is not None:
                first = (
                    f"Round {round_idx}: constraint tool '{tool_name}' was called with "
                    f"{arg_label}=\"{primary_val}\", but it could not produce usable "
                    f"constrained designs, so no new constrained sequences were scored in this round."
                )
            else:
                first = (
                    f"Round {round_idx}: constraint tool '{tool_name}' was called, but it "
                    f"could not produce usable constrained designs, so no new constrained "
                    f"sequences were scored in this round."
                )
        else:
            # 其它工具（例如 esm_inpaint / get_score 等）统一一个描述
            if primary_val is not None:
                first = (
                    f"Round {round_idx}: tool '{tool_name}' was called with "
                    f"{arg_label}=\"{primary_val}\", but the call failed and no new sequences "
                    f"were scored in this round."
                )
            else:
                first = (
                    f"Round {round_idx}: tool '{tool_name}' failed with the current arguments, "
                    f"and no new sequences were scored in this round."
                )

        if best_val is not None:
            first += f" Global best ProTrek remains {best_val:.3f}."

        # Summary 部分，结构和正常 obs 尽量保持一致
        lines: list[str] = [first, "Summary:"]
        lines.append(f"- round: {round_idx}")
        lines.append(f"- tool: {tool_name}")
        lines.append(f"- num_sequences_scored: {num_scored}")
        if best_val is not None:
            lines.append(f"- global_best_ProTrek: {best_val:.3f}")
            lines.append(f"- prev_global_best_ProTrek: {best_val:.3f}")
        lines.append(f"- delta_best: {delta_best:+.3f}")

        if primary_val is not None and arg_key is not None:
            lines.append(f"- argument_{arg_key}: {primary_val}")

        # 可以选择性地附上当前 best_accession，方便模型知道“全局最佳是谁”
        best_acc = self.state.get("best_accession")
        if best_acc:
            lines.append(f"- current_best_accession: {best_acc}")

        return "\n".join(lines)




    # --------------- public API ---------------

    def run_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        tool_call format (similar to OpenAI function calling):
        {
          "id": "call_9_0",
          "type": "function",
          "function": {
            "name": "go2seq",
            "arguments": {
              "go_term": "DNA-binding transcription activator",
              "size": 30,
              "timeout": 30,
              "organism": null,
              "include_unreviewed": false
            }
          }
        }

        Returns:
           {
             "content": "<observation string>",
             "ok": True/False,
             "extra": {... anything you want ...}
           }
        """
        fn = tool_call.get("function", {})
        name = fn.get("name")
        raw_args = fn.get("arguments", {}) or {}

        # arguments 既可能是 dict，也可能是 JSON 字符串，统一转成 dict
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args

        handler = self.handlers.get(name)
        if handler is None:
            return {
                "content": f"ERROR: unknown tool '{name}'",
                "ok": False,
                "extra": {},
            }

        # 每次调用视为一轮
        self.state["round"] = int(self.state.get("round", 0)) + 1

        try:
            obs, extra = handler(name, args)
            return {
                "content": obs,
                "ok": True,
                "extra": extra,
            }

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or str(e)
            obs = self._build_error_observation(name, error_msg, args)
            return {
                "content": obs,
                "ok": False,
                "extra": {
                    "error_type": "subprocess_error",
                    "raw_error": error_msg,
                },
            }

        except Exception as e:
            error_msg = repr(e)
            obs = self._build_error_observation(name, error_msg, args)
            return {
                "content": obs,
                "ok": False,
                "extra": {
                    "error_type": "python_error",
                    "raw_error": error_msg,
                },
            }

    def pick_accessions_from_scaffold(self, topn: int = 3) -> List[str]:
        """
        Read self.scaffold_json and return up to `topn` accessions
        in order. Assumes each entry in scaffold.json looks like:
            {
              "accession": "P0A8Q0",
              "sequence":  "MKK....",
              ...
            }
        If scaffold.json doesn't exist or is empty, returns [].
        """
        data = self._load_json_if_exists(self.scaffold_json, default=[])
        accs: List[str] = []
        for entry in data:
            acc = entry.get("accession")
            if acc:
                accs.append(acc)
            if len(accs) >= topn:
                break
        return accs

    # --------------- ProTrek mini scoring & state ---------------

    # def _caculate_protrek_score_mini(
    #     self,
    #     sequences: List[Dict[str, str]],
    # ) -> List[Dict[str, Any]]:
    #     """
    #     用 ProTrek 对给定的 sequences 列表打分。

    #     sequences: list of {"accession": str, "sequence": str}
    #     返回值: list of {"accession", "sequence", "score", ...}

    #     逻辑：
    #     - requirement_text 从 self.state["requirement_text"] 里拿；
    #     - 把 sequences 写到 tmp_for_scoring.json；
    #     - 调用 caculate_similarity_text_seq.py（similarity_score）；
    #     - 输出写到 self.mini_ranked_json；
    #     - 解析后返回完整列表（不限制 topk，topk 合并留到 _update_tmp_ranked()）。
    #     """
    #     if not sequences:
    #         return []

    #     design_text = (self.state or {}).get("requirement_text", "")
    #     if not design_text:
    #         raise RuntimeError(
    #             "requirement_text is empty. Please provide it via AgentRuntime(..., requirement_text=...) "
    #             "or runtime.set_requirement_text(...) before scoring."
    #         )

    #     # 写临时输入
    #     with open(self.tmp_scoring_json, "w", encoding="utf-8") as f_tmp:
    #         json.dump(sequences, f_tmp, ensure_ascii=False, indent=2)

    #     # 用 ProTrek 脚本打分。topk 设为 len(sequences)，保证全保留。
    #     script_path = self.scripts["similarity_score"]
    #     cmd = [
    #         "python",
    #         script_path,
    #         "--text", design_text,
    #         "--json", self.tmp_scoring_json,
    #         "--topk", str(len(sequences)),
    #         "--out", self.mini_ranked_json,
    #         "--device", "cpu"
    #     ]
    #     self._run_subprocess(cmd, use_protrek_env=True, check=True)

    #     ranked_raw = self._load_json_if_exists(self.mini_ranked_json, default=[])
    #     if isinstance(ranked_raw, dict):
    #         if "results" in ranked_raw and isinstance(ranked_raw["results"], list):
    #             ranked_entries = ranked_raw["results"]
    #         else:
    #             ranked_entries = [ranked_raw]
    #     elif isinstance(ranked_raw, list):
    #         ranked_entries = ranked_raw
    #     else:
    #         ranked_entries = []

    #     # 确保包含 accession, sequence, score
    #     scored: List[Dict[str, Any]] = []
    #     for r in ranked_entries:
    #         scored.append({
    #             "accession": r.get("accession"),
    #             "sequence": r.get("sequence", ""),
    #             "score": r.get("score", None),
    #         })
    #     return scored




    def _caculate_protrek_score_mini(
        self,
        sequences: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        用远端 ProTrek API 对给定的 sequences 列表打分。

        sequences: list of {"accession": str, "sequence": str}
        返回值: list of {"accession", "sequence", "score", ...}
        """
        if not sequences:
            return []

        design_text = (self.state or {}).get("requirement_text", "")
        if not design_text:
            raise RuntimeError(
                "requirement_text is empty. Please provide it via AgentRuntime(..., requirement_text=...) "
                "or runtime.set_requirement_text(...) before scoring."
            )

        # 1) 仍然把输入 sequences 写到本机 tmp_scoring_json（方便 debug / 兼容老逻辑）
        os.makedirs(os.path.dirname(self.tmp_scoring_json), exist_ok=True)
        with open(self.tmp_scoring_json, "w", encoding="utf-8") as f_tmp:
            json.dump(sequences, f_tmp, ensure_ascii=False, indent=2)

        # 2) 构造发给 API 的 payload
        #    API 期望的结构：{"text": text, "sequences": [{accession, sequence}], "topk": K}
        payload = {
            "text": design_text,
            "sequences": sequences,        # 直接用原结构即可
            "topk": len(sequences),        # 保证全保留；想截断就传更小
        }

        # 如果你在 config 里有 self.protrek_api_url，就用那个；否则用环境变量默认
        api_url = getattr(self, "protrek_api_url", None) or PROTREK_API_URL

        try:
            resp = requests.post(api_url, json=payload, timeout=600)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"ProTrek API request failed: {e}") from e

        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError(f"ProTrek API returned error: {data}")

        # 3) 得到 API 返回的结果列表（已经按 score 降序）
        #    形如 [{"sequence": "...", "accession": "...", "score": 12.3}, ...]
        results = data.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError(f"ProTrek API returned invalid 'results': {results!r}")

        # 4) 把结果写回本机 mini_ranked_json，保持和原脚本输出尽量一致
        os.makedirs(os.path.dirname(self.mini_ranked_json), exist_ok=True)
        with open(self.mini_ranked_json, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=2)
            f_out.write("\n")  # 原来脚本也是 dump 后加一个换行

        # 5) 返回 scored 列表（保持原格式）
        scored: List[Dict[str, Any]] = []
        for r in results:
            scored.append({
                "accession": r.get("accession"),
                "sequence": r.get("sequence", ""),
                "score": r.get("score", None),
            })
        return scored

    
    def _update_tmp_ranked(
        self,
        new_scored: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        把新打分结果并入全局 tmp_ranked.json，只保留全局 top_k。
        """
        if not new_scored:
            return self._load_json_if_exists(self.tmp_ranked_json, default=[])

        old_list = self._load_json_if_exists(self.tmp_ranked_json, default=[])
        if not isinstance(old_list, list):
            old_list = []

        # 合并 + 去重（按 accession + sequence）
        combined = old_list + new_scored
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for entry in combined:
            acc = entry.get("accession")
            seq = entry.get("sequence", "")
            key = (acc, seq)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)

        # 按 score 降序
        deduped_sorted = sorted(
            deduped,
            key=lambda x: float(x.get("score", float("-inf"))),
            reverse=True,
        )
        topk = deduped_sorted[: self.default_topk]

        self._save_json(self.tmp_ranked_json, topk)
        return topk

    def _score_and_update_state(
        self,
        tool_name: str,
        sequences: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        对本轮新序列打分，并更新全局 state + tmp_ranked.json。

        返回：
          {
            "best_this_round": float or None,
            "global_best": float or None,
            "prev_global_best": float or None,
            "delta_best": float,
            "num_scored": int,
            "new_scored": [...],
            "global_topk": [...],
          }
        """
        prev_best = float(self.state.get("best_score", float("-inf")))

        if not sequences:
            # 本轮没新序列，不更新 best
            self.state["prev_best"] = prev_best
            return {
                "best_this_round": None,
                "global_best": prev_best if prev_best != float("-inf") else None,
                "prev_global_best": prev_best if prev_best != float("-inf") else None,
                "delta_best": 0.0,
                "num_scored": 0,
                "new_scored": [],
                "global_topk": self._load_json_if_exists(self.tmp_ranked_json, default=[]),
            }

        new_scored = self._caculate_protrek_score_mini(sequences)
        if not new_scored:
            self.state["prev_best"] = prev_best
            return {
                "best_this_round": None,
                "global_best": prev_best if prev_best != float("-inf") else None,
                "prev_global_best": prev_best if prev_best != float("-inf") else None,
                "delta_best": 0.0,
                "num_scored": 0,
                "new_scored": [],
                "global_topk": self._load_json_if_exists(self.tmp_ranked_json, default=[]),
            }

        global_topk = self._update_tmp_ranked(new_scored)

        best_this_round = max(
            float(e.get("score", float("-inf"))) for e in new_scored
        )

        if global_topk:
            global_best = float(global_topk[0].get("score", float("-inf")))
            best_acc = global_topk[0].get("accession")
            best_seq = global_topk[0].get("sequence", "")
        else:
            global_best = prev_best
            best_acc = None
            best_seq = None

        # 更新 state
        self.state["prev_best"] = prev_best
        self.state["best_score"] = global_best
        self.state["best_accession"] = best_acc
        self.state["best_seq"] = best_seq

        if prev_best == float("-inf"):
            delta = global_best
        else:
            delta = global_best - prev_best

        return {
            "best_this_round": best_this_round,
            "global_best": global_best,
            "prev_global_best": prev_best if prev_best != float("-inf") else None,
            "delta_best": delta,
            "num_scored": len(new_scored),
            "new_scored": new_scored,
            "global_topk": global_topk,
        }

    def _build_summary_lines(
        self,
        tool_name: str,
        score_info: Dict[str, Any],
        extra_lines: Optional[list[str]] = None,
    ) -> list[str]:
        """公共的 summary 部分，不含第一行 natural language。"""
        round_idx = int(self.state.get("round", 0))

        num_scored = score_info.get("num_scored", 0)
        best_this = score_info.get("best_this_round")
        global_best = score_info.get("global_best")
        prev_best = score_info.get("prev_global_best")
        delta = score_info.get("delta_best")

        lines: list[str] = [
            "Summary:",
            f"- round: {round_idx}",
            f"- tool: {tool_name}",
            f"- num_sequences_scored: {num_scored}",
        ]
        if best_this is not None:
            lines.append(f"- best_ProTrek_this_round: {best_this:.3f}")
        if global_best is not None and global_best != float("-inf"):
            lines.append(f"- global_best_ProTrek: {global_best:.3f}")
        if prev_best is not None and prev_best != float("-inf"):
            lines.append(f"- prev_global_best_ProTrek: {prev_best:.3f}")
        if delta is not None:
            lines.append(f"- delta_best: {delta:+.3f}")

        # 这里再加上额外的 summary 句子（比如 scaffold pool 大小、约束成功条数等）
        # if extra_lines:
        #     for e in extra_lines:
        #         lines.append(f"- {e}")

        # 可以选择性地暴露 best_accession（sequence 不再放进 obs）
        best_acc = self.state.get("best_accession")
        if best_acc:
            lines.append(f"- current_best_accession: {best_acc}")

        return lines

    def _build_observation(
        self,
        tool_name: str,
        score_info: Dict[str, Any],
        extra_lines: Optional[list[str]] = None,
    ) -> str:
        """
        正常执行（无异常）时的 observation：
        - 第一行：natural language 描述本轮工具执行结果
        - 后面：Summary block，给出数字摘要
        """
        round_idx = int(self.state.get("round", 0))

        num_scored = score_info.get("num_scored", 0)
        best_this = score_info.get("best_this_round")
        global_best = score_info.get("global_best")
        prev_best = score_info.get("prev_global_best")
        delta = score_info.get("delta_best")

        # 第一句（自然语言）
        if num_scored > 0 and best_this is not None:
            first = (
                f"Round {round_idx}: tool '{tool_name}' ran successfully and scored "
                f"{num_scored} new sequence(s). Best ProTrek this round: {best_this:.3f}. "
            )
        else:
            first = (
                f"Round {round_idx}: tool '{tool_name}' ran but did not add or score "
                f"any new sequences in this round. "
            )

        if global_best is not None and global_best != float("-inf"):
            if prev_best is not None and prev_best != float("-inf") and delta is not None:
                if abs(delta) < 1e-6:
                    first += (
                        f"Global best ProTrek remains {global_best:.3f} "
                        f"(Δbest = {delta:+.3f} relative to the previous round)."
                    )
                elif delta > 0:
                    first += (
                        f"Global best ProTrek improved to {global_best:.3f} "
                        f"(Δbest = {delta:+.3f})."
                    )
                else:
                    first += (
                        f"Global best ProTrek decreased to {global_best:.3f} "
                        f"(Δbest = {delta:+.3f})."
                    )
            else:
                first += f"Current global best ProTrek is {global_best:.3f}."

        lines = [first]
        lines.extend(self._build_summary_lines(tool_name, score_info, extra_lines))
        return "\n".join(lines)


    # --------------- handlers ---------------

    def _handle_scaffold_generator(self, tool_name: str, args: Dict[str, Any]):
        """
        Handles function2seq / pathway2seq / dna_binding2seq / domain2seq / go2seq.

        Expected args keys:
          - one query arg that differs by tool:
              function2seq: text
              pathway2seq: pathway
              dna_binding2seq: text
              domain2seq: domain
              go2seq: go_term
          - optional: size, timeout, organism, include_unreviewed

        我们会：
        - 调用对应脚本，把结果写入 scaffold.json（追加或合并）。
        - 读取 scaffold.json，抽出所有条目中新的 {accession, sequence}。
        - 调用 ProTrek mini 打分 + 更新全局 state。
        - 返回统一的 observation。
        """
        script_path = self.scripts[tool_name]

        # map tool -> its main query arg name
        main_arg_name_map = {
            "function2seq": "text",
            "pathway2seq": "pathway",
            "dna_binding2seq": "text",
            "domain2seq": "domain",
            "go2seq": "go_term",
        }
        main_arg_name_map_tool = {
            "function2seq": "query",
            "pathway2seq": "text",
            "dna_binding2seq": "keywords",
            "domain2seq": "text",
            "go2seq": "text",
        }

        main_arg = main_arg_name_map[tool_name]
        main_arg_tool = main_arg_name_map_tool[tool_name]
        main_val = args.get(main_arg, "")

        size = str(args.get("size", 20))
        timeout = str(args.get("timeout", 60))
        organism = args.get("organism")
        include_unreviewed = args.get("include_unreviewed", False)

        # 记录调用前的 scaffold pool，方便找 new entries
        prev_scaffolds = self._load_json_if_exists(self.scaffold_json, default=[])
        prev_keys = {
            (e.get("accession"), e.get("sequence", ""))
            for e in prev_scaffolds if isinstance(e, dict)
        }

        if tool_name == 'go2seq':
            cmd = [
                "python",
                script_path,
                f"--{main_arg_tool}", str(main_val),
                "--size", str(int(size) // 10),
                "--timeout", timeout,
                "--json", self.scaffold_json,
            ]
        else:
            cmd = [
                "python",
                script_path,
                f"--{main_arg_tool}", str(main_val),
                "--size", size,
                "--timeout", timeout,
                "--json", self.scaffold_json,
            ]            
        if organism:
            cmd += ["--organism", organism]
        if include_unreviewed:
            cmd += ["--unreviewed"]

        result = self._run_subprocess(cmd, use_protrek_env=False, check=True)

        # 读取新的 scaffold pool
        scaffolds = self._load_json_if_exists(self.scaffold_json, default=[])
        new_sequences: List[Dict[str, str]] = []
        for entry in scaffolds:
            if not isinstance(entry, dict):
                continue
            acc = entry.get("accession")
            seq = entry.get("sequence", "")
            key = (acc, seq)
            if key in prev_keys:
                continue
            if acc and seq:
                new_sequences.append({"accession": acc, "sequence": seq})

        score_info = self._score_and_update_state(tool_name, new_sequences)

        obs = self._build_observation(
            tool_name,
            score_info,
            extra_lines=[
                f"scaffold_pool_size={len(scaffolds)}",
                f"stdout_summary={result.stdout.strip()[:200]}",
            ],
        )
        extra = {
            "scaffold_count": len(scaffolds),
            "stdout": result.stdout.strip(),
            "score_info": score_info,
        }
        return obs, extra

    # def _handle_get_score(self, tool_name: str, args: Dict[str, Any]):
    #     """
    #     显式 get_score：聚合当前所有已知序列（scaffold + inpaint），再做一轮 ProTrek 打分，
    #     并直接返回“当前全局最好的设计”和它的分数。

    #     对于模型来说，这更像是一个“最终/阶段性评估”的工具，
    #     observation 主要内容就是 best sequence + score。
    #     """
    #     # 收集 scaffold 序列
    #     sequences: List[Dict[str, str]] = []
    #     scaffolds = self._load_json_if_exists(self.scaffold_json, default=[])
    #     for entry in scaffolds:
    #         if not isinstance(entry, dict):
    #             continue
    #         acc = entry.get("accession")
    #         seq = entry.get("sequence", "")
    #         if acc and seq:
    #             sequences.append({"accession": acc, "sequence": seq})

    #     # 收集 inpaint 序列
    #     inpaint_list = self._load_json_if_exists(self.inpaint_results_json, default=[])
    #     for entry in inpaint_list:
    #         if not isinstance(entry, dict):
    #             continue
    #         acc = entry.get("accession")
    #         seq = entry.get("sequence", "")
    #         if acc and seq:
    #             sequences.append({"accession": acc, "sequence": seq})

    #     # 去重
    #     uniq = {}
    #     for e in sequences:
    #         key = (e["accession"], e["sequence"])
    #         uniq[key] = e
    #     sequences = list(uniq.values())

    #     # 如果压根没有序列可评估，直接告诉模型
    #     round_idx = int(self.state.get("round", 0))
    #     if not sequences:
    #         obs = (
    #             f"Round {round_idx}: tool 'get_score' was called, "
    #             "but there are no sequences available to score. "
    #             "No best design can be reported."
    #         )
    #         extra = {
    #             "sequence_count": 0,
    #             "score_info": None,
    #             "best_entry": None,
    #         }
    #         return obs, extra

    #     # 正常打分 + 更新全局 state / tmp_ranked.json
    #     score_info = self._score_and_update_state(tool_name, sequences)
    #     global_topk = score_info.get("global_topk") or []

    #     if not global_topk:
    #         # 理论上不该发生（因为 sequences 非空），保险兜底一下
    #         obs = (
    #             f"Round {round_idx}: tool 'get_score' ran, "
    #             "but scoring did not produce a ranked list. "
    #             "No best design can be reported."
    #         )
    #         extra = {
    #             "sequence_count": len(sequences),
    #             "score_info": score_info,
    #             "best_entry": None,
    #         }
    #         return obs, extra

    #     # 取当前全局最优的一条
    #     best_entry = global_topk[0]
    #     acc = best_entry.get("accession") or "N/A"
    #     seq = best_entry.get("sequence", "") or ""
    #     score = best_entry.get("score", None)

    #     if score is None:
    #         score_str = "unknown"
    #     else:
    #         score_str = f"{float(score):.3f}"

    #     # 这里就按你说的，只专注于“最好的序列和 score”
    #     obs_lines = [
    #         f"Round {round_idx}: tool 'get_score' aggregated and re-scored all known sequences "
    #         f"and selected the current best design.",
    #         "",
    #         "Best design:",
    #         f"- accession: {acc}",
    #         f"- ProTrek_score: {score_str}",
    #         f"- sequence: {seq}",
    #     ]
    #     obs = "\n".join(obs_lines)

    #     extra = {
    #         "sequence_count": len(sequences),
    #         "score_info": score_info,
    #         "best_entry": best_entry,
    #     }
    #     return obs, extra

    def _handle_get_score(self, tool_name: str, args: Dict[str, Any]):
        """
        显式 get_score：聚合当前“已排序的全局 top-k 序列”再做一轮 ProTrek 打分，
        并直接返回“当前全局最好的设计”和它的分数。

        新逻辑：
        - 如果 tmp_ranked.json 已经存在且非空：
            * 只取其中前 default_topk 条的 {accession, sequence} 重新打分；
        - 否则（比如还从没打过分）：
            * 回退到旧逻辑：从 scaffold.json + inpaint_results.json 里收集所有序列来打分。
        """

        round_idx = int(self.state.get("round", 0))

        sequences: List[Dict[str, str]] = []

        # 优先用 tmp_ranked.json 中的全局 top-k
        ranked_list = self._load_json_if_exists(self.tmp_ranked_json, default=[])
        if isinstance(ranked_list, dict):
            # 理论上不会是 dict，这里只是防守式兜底
            ranked_list = ranked_list.get("results", []) or []

        if isinstance(ranked_list, list) and len(ranked_list) > 0:
            # 只取前 default_topk 条
            topk = ranked_list[: getattr(self, "default_topk", 5)]
            for entry in topk:
                if not isinstance(entry, dict):
                    continue
                acc = entry.get("accession")
                seq = entry.get("sequence", "")
                if acc and seq:
                    sequences.append({"accession": acc, "sequence": seq})
        else:
            # fallback：如果 tmp_ranked.json 还没建立，就退回到“全局所有序列”逻辑
            # 收集 scaffold 序列
            scaffolds = self._load_json_if_exists(self.scaffold_json, default=[])
            for entry in scaffolds:
                if not isinstance(entry, dict):
                    continue
                acc = entry.get("accession")
                seq = entry.get("sequence", "")
                if acc and seq:
                    sequences.append({"accession": acc, "sequence": seq})

            # 收集 inpaint 序列
            inpaint_list = self._load_json_if_exists(self.inpaint_results_json, default=[])
            for entry in inpaint_list:
                if not isinstance(entry, dict):
                    continue
                acc = entry.get("accession")
                seq = entry.get("sequence", "")
                if acc and seq:
                    sequences.append({"accession": acc, "sequence": seq})

        # 去重（按 accession + sequence）
        uniq = {}
        for e in sequences:
            key = (e["accession"], e["sequence"])
            uniq[key] = e
        sequences = list(uniq.values())

        # 如果压根没有序列可评估，直接告诉模型
        if not sequences:
            obs = (
                f"Round {round_idx}: tool 'get_score' was called, "
                "but there are no sequences available to score. "
                "No best design can be reported."
            )
            extra = {
                "sequence_count": 0,
                "score_info": None,
                "best_entry": None,
            }
            return obs, extra

        # 正常打分 + 更新全局 state / tmp_ranked.json
        score_info = self._score_and_update_state(tool_name, sequences)
        global_topk = score_info.get("global_topk") or []

        if not global_topk:
            # 理论上不该发生（因为 sequences 非空），保险兜底一下
            obs = (
                f"Round {round_idx}: tool 'get_score' ran, "
                "but scoring did not produce a ranked list. "
                "No best design can be reported."
            )
            extra = {
                "sequence_count": len(sequences),
                "score_info": score_info,
                "best_entry": None,
            }
            return obs, extra

        # 取当前全局最优的一条
        best_entry = global_topk[0]
        acc = best_entry.get("accession") or "N/A"
        seq = best_entry.get("sequence", "") or ""
        score = best_entry.get("score", None)

        if score is None:
            score_str = "unknown"
        else:
            score_str = f"{float(score):.3f}"

        obs_lines = [
            f"Round {round_idx}: tool 'get_score' re-scored the current global top-k sequences "
            f"and selected the best design.",
            "",
            "Best design:",
            f"- accession: {acc}",
            f"- ProTrek_score: {score_str}",
            f"- sequence: {seq}",
        ]
        obs = "\n".join(obs_lines)

        extra = {
            "sequence_count": len(sequences),
            "score_info": score_info,
            "best_entry": best_entry,
        }
        return obs, extra


    def _ensure_constraints_for_accession(self, acc: str) -> str:
        """
        Make sure <work_dir>/<acc>_constraints.json exists.
        If missing, call build_constraints_from_uniprot.py once to create it.

        Returns the path to the constraints json.
        """
        cfile = self._constraints_path_for_accession(acc)
        if os.path.exists(cfile):
            return cfile  # already there

        script_path = self.scripts["build_constraints_from_uniprot"]
        cmd = [
            "python",
            script_path,
            "--accession", acc,
            "--out", self.work_dir,
        ]
        print(f"[ensure_constraints] creating base constraints for {acc}")
        self._run_subprocess(cmd, use_protrek_env=False, check=True)
        return cfile

    def _handle_update_constraints(self, tool_name: str, args: Dict[str, Any]):
        """
        Apply cofactor/motif/signal constraints to top-k accessions from tmp_ranked.json，
        对每个 accession:
          - 尝试应用约束（cofactor2constraints/motif2constraints/signal2constraints）
          - 不因单个 accession 报错而中断其它 accession
        然后：
          - 对所有约束成功的 accession 自动跑一轮 esm_inpaint
          - 对本轮 inpaint 产生的新序列做 ProTrek mini 打分 + 更新 state
        """

        topk = getattr(self, "default_topk", 5)

        ranked_list = self._load_json_if_exists(self.tmp_ranked_json, default=[])
        if not ranked_list:
            raise RuntimeError(
                "tmp_ranked.json is empty; you should generate & score some scaffolds "
                "before applying constraints."
            )

        ranked_list = ranked_list[:topk]
        acc_list = [entry.get("accession") for entry in ranked_list if entry.get("accession")]
        if not acc_list:
            raise RuntimeError("No valid accession entries found in tmp_ranked.json.")

        success_accs: list[str] = []
        failed_accs: list[Dict[str, Any]] = []

        for acc in acc_list:
            cfile = self._ensure_constraints_for_accession(acc)

            try:
                # 根据 tool_name 构建命令
                if tool_name == "cofactor2constraints":
                    script = self.scripts["cofactor2constraints"]
                    cofactor = args.get("cofactor", "")
                    cmd = [
                        "python",
                        script,
                        "--in_json", cfile,
                        "--cofactor", cofactor,
                    ]

                elif tool_name == "motif2constraints":
                    script = self.scripts["motif2constraints"]
                    desc = args.get("motif", "")
                    cmd = [
                        "python",
                        script,
                        "--json", cfile,
                        "--desc", desc,
                    ]

                elif tool_name == "signal2constraints":
                    script = self.scripts["signal2constraints"]
                    cmd = [
                        "python",
                        script,
                        "--json", cfile,
                    ]

                else:
                    raise ValueError(f"Unsupported tool: {tool_name}")

                print(f"[constraints] running {tool_name} for {acc}")
                self._run_subprocess(cmd, use_protrek_env=False, check=True)

                # 这个 accession 约束成功
                success_accs.append(acc)

            except subprocess.CalledProcessError as e:
                failed_accs.append({
                    "accession": acc,
                    "error": e.stderr or str(e),
                    "type": "subprocess_error",
                })
                # 不中断循环，继续处理下一个 accession
                continue

            except Exception as e:
                failed_accs.append({
                    "accession": acc,
                    "error": repr(e),
                    "type": "python_error",
                })
                continue

        # 如果一个都没成功，就给一个统一的错误 observation，让模型知道需要调整约束/策略
        if not success_accs:
            obs = self._build_error_observation(
                tool_name,
                f"Constraint tool {tool_name} failed for all {len(acc_list)} candidate accessions.",
            )
            extra = {
                "updated_accessions": [],
                "failed_accessions": failed_accs,
            }
            return obs, extra

        # === 约束成功的 accession：自动接 esm_inpaint + 打分 ===
        sequences_for_scoring, new_batch_entries, all_results = self._run_esm_inpaint_for_accessions(success_accs)

        score_info = self._score_and_update_state(tool_name, sequences_for_scoring)

        extra_lines = [
            f"applied_constraints_to={len(success_accs)} accessions (from_topk={len(acc_list)})",
            f"esm_inpaint_new_entries={len(new_batch_entries)}",
            f"inpaint_results_total_entries={len(all_results)}",
        ]
        if failed_accs:
            # 简单地提示有部分 accession 失败并被跳过
            failed_str = ",".join(a["accession"] for a in failed_accs if a.get("accession"))
            extra_lines.append(
                f"constraint_failed_for_accessions={failed_str} (these were skipped for this round)."
            )

        obs = self._build_observation(
            tool_name,
            score_info,
            extra_lines=extra_lines,
        )
        extra = {
            "updated_accessions": success_accs,
            "failed_accessions": failed_accs,
            "inpaint_added_entries": new_batch_entries,
            "score_info": score_info,
        }
        return obs, extra

    # ---- shared helper for esm_inpaint ----

    # def _run_esm_inpaint_for_accessions(
    #     self,
    #     acc_list: List[str],
    # ) -> (List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, Any]]):
    #     """
    #     内部 helper：对给定 accession 列表运行 esm_inpaint，
    #     更新 inpaint_results.json，并返回：
    #       sequences_for_scoring: [{accession, sequence}]
    #       new_batch_entries:     [{accession, sequence, summary}]
    #       all_results:           完整 inpaint_results 列表
    #     """
    #     script_path = self.scripts["esm_inpaint"]

    #     all_results = self._load_json_if_exists(self.inpaint_results_json, default=[])
    #     if not isinstance(all_results, list):
    #         all_results = []

    #     new_batch_entries: List[Dict[str, Any]] = []

    #     for acc in acc_list:
    #         cfile = self._ensure_constraints_for_accession(acc)

    #         out_file = os.path.join(self.work_dir, f"{acc}_inpaint.json")
    #         cmd = [
    #             "python",
    #             script_path,
    #             "--json", cfile,
    #             "--out", out_file,
    #         ]
    #         print(f"[Stage3:esm_inpaint] running for {acc}")
    #         self._run_subprocess(cmd, use_protrek_env=False, check=True)

    #         # load per-accession result JSON
    #         try:
    #             with open(out_file, "r", encoding="utf-8") as f_in:
    #                 payload = json.load(f_in)
    #         except Exception as e:
    #             payload = {
    #                 "sequence": "",
    #                 "original_sequence": "",
    #                 "debug_info": {"error": f"failed to load {out_file}: {repr(e)}"},
    #             }

    #         final_seq = payload.get("sequence", "")
    #         original_seq = payload.get("original_sequence", "")
    #         debug_info = payload.get("debug_info", {})

    #         entry_record = {
    #             "accession": acc,
    #             "sequence": final_seq,
    #             "summary": {
    #                 "original_sequence_len": len(original_seq) if original_seq else None,
    #                 "final_sequence_len": len(final_seq) if final_seq else None,
    #                 "debug_info": debug_info,
    #             },
    #         }

    #         new_batch_entries.append(entry_record)
    #         all_results.append(entry_record)

    #     self._save_json(self.inpaint_results_json, all_results)

    #     sequences_for_scoring = [
    #         {"accession": e["accession"], "sequence": e["sequence"]}
    #         for e in new_batch_entries
    #         if e.get("accession") and e.get("sequence")
    #     ]

    #     return sequences_for_scoring, new_batch_entries, all_results

    # def _handle_esm_inpaint(self, tool_name: str, args: Dict[str, Any]):
    #     """
    #     单独调用 esm_inpaint 工具：

    #     - 如果 args 里提供 "accessions"，就用这批；
    #     - 否则从 tmp_ranked.json 里取前 default_topk 个 accession；
    #     - 对这些 accession 跑 inpaint；
    #     - 对本轮 inpaint 结果做 ProTrek mini 打分 + 更新 state；
    #     - 返回统一 observation。
    #     """
    #     # 决定要 inpaint 的 accession 列表
    #     if "accessions" in args and args["accessions"]:
    #         acc_list = [a for a in args["accessions"] if a]
    #     else:
    #         ranked_list = self._load_json_if_exists(self.tmp_ranked_json, default=[])
    #         acc_list = [
    #             entry.get("accession")
    #             for entry in ranked_list[: getattr(self, "default_topk", 5)]
    #             if entry.get("accession")
    #         ]

    #     if not acc_list:
    #         raise RuntimeError(
    #             "No accession available to run esm_inpaint. "
    #             "You should generate & score scaffolds first (so that tmp_ranked.json is non-empty), "
    #             "or pass explicit 'accessions' in tool arguments."
    #         )

    #     sequences_for_scoring, new_batch_entries, all_results = self._run_esm_inpaint_for_accessions(acc_list)

    #     score_info = self._score_and_update_state(tool_name, sequences_for_scoring)

    #     obs = self._build_observation(
    #         tool_name,
    #         score_info,
    #         extra_lines=[
    #             f"esm_inpaint_executed_for={len(acc_list)} accessions",
    #             f"inpaint_results_total_entries={len(all_results)}",
    #         ],
    #     )
    #     extra = {
    #         "accessions": acc_list,
    #         "added_entries": new_batch_entries,
    #         "score_info": score_info,
    #     }
    #     return obs, extra


    def _run_esm_inpaint_for_accessions(
        self,
        acc_list: List[str],
    ) -> (List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, Any]]):
        """
        内部 helper：对给定 accession 列表运行 esm_inpaint，
        更新 inpaint_results.json，并返回：
        sequences_for_scoring: [{accession, sequence}]
        new_batch_entries:     [{accession, sequence, summary}]
        all_results:           完整 inpaint_results 列表

        现在不再在本机起子进程跑 esm_constrain.py，
        而是调用远程 ESM API (/esm_constrain)。
        """

        # 原来的 script_path 保留（即使不用），避免其他地方依赖 self.scripts["esm_inpaint"]
        script_path = self.scripts.get("esm_inpaint")

        # 远程 API 地址：
        # 优先用实例属性 self.esm_constrain_api_url，其次用环境变量，最后用默认值
        esm_api_url = getattr(self, "esm_constrain_api_url", None) \
                    or os.getenv("ESM_CONSTRAIN_API_URL", "http://101.126.67.113:8863/esm_constrain")

        all_results = self._load_json_if_exists(self.inpaint_results_json, default=[])
        if not isinstance(all_results, list):
            all_results = []

        new_batch_entries: List[Dict[str, Any]] = []

        for acc in acc_list:
            # 原逻辑：确保本地 constraints.json 存在
            cfile = self._ensure_constraints_for_accession(acc)

            # ⭐ 1) 读取本地 constraints.json 内容
            try:
                with open(cfile, "r", encoding="utf-8") as f_cfg:
                    constraints = json.load(f_cfg)
            except Exception as e:
                raise RuntimeError(f"Failed to load constraints json for {acc} from {cfile}: {e}") from e

            # 本机仍然保留一个 out_file 路径，后面照旧写结果，保证行为兼容
            out_file = os.path.join(self.work_dir, f"{acc}_inpaint.json")

            # ⭐ 2) 调用远程 /esm_constrain API
            payload = {
                "constraints": constraints,
                # "model_dir": 可选，一般不需要传，远端用默认的 ESM checkpoint
            }

            print(f"[Stage3:esm_inpaint] calling remote ESM API for {acc} -> {esm_api_url}")
            try:
                resp = requests.post(esm_api_url, json=payload, timeout=600)
                resp.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Remote ESM API request failed for accession {acc}: {e}") from e

            try:
                data = resp.json()
            except Exception as e:
                raise RuntimeError(f"Remote ESM API returned non-JSON for accession {acc}: {e}") from e

            if data.get("status") != "success":
                raise RuntimeError(f"Remote ESM API error for {acc}: {data}")

            result_payload = data.get("result", {})               # == 原来 args.out 写出的 JSON
            updated_constraints = data.get("updated_constraints") # == 写回后的 constraints.json

            if not isinstance(result_payload, dict) or not isinstance(updated_constraints, dict):
                raise RuntimeError(f"Remote ESM API returned invalid payload for {acc}: {data}")

            # ⭐ 3) 把 updated_constraints 写回原来的 cfile
            #    行为等价于原来的 esm_constrain.py 在本地写回 sequence_inpaint
            try:
                with open(cfile, "w", encoding="utf-8") as f_cfg_out:
                    json.dump(updated_constraints, f_cfg_out, ensure_ascii=False, indent=2)
            except Exception as e:
                raise RuntimeError(f"Failed to write updated constraints back to {cfile} for {acc}: {e}") from e

            # ⭐ 4) 把 result_payload 写到本地 out_file，保持后续逻辑兼容
            try:
                with open(out_file, "w", encoding="utf-8") as f_out:
                    json.dump(result_payload, f_out, ensure_ascii=False, indent=2)
            except Exception as e:
                raise RuntimeError(f"Failed to write local inpaint result json to {out_file} for {acc}: {e}") from e

            # ⭐ 5) 后面解析 payload 的逻辑保持不变
            final_seq = result_payload.get("sequence", "")
            original_seq = result_payload.get("original_sequence", "")
            debug_info = result_payload.get("debug_info", {})

            entry_record = {
                "accession": acc,
                "sequence": final_seq,
                "summary": {
                    "original_sequence_len": len(original_seq) if original_seq else None,
                    "final_sequence_len": len(final_seq) if final_seq else None,
                    "debug_info": debug_info,
                },
            }

            new_batch_entries.append(entry_record)
            all_results.append(entry_record)

        # 写回 inpaint_results.json（与原行为一致）
        self._save_json(self.inpaint_results_json, all_results)

        # sequences_for_scoring 结构不变
        sequences_for_scoring = [
            {"accession": e["accession"], "sequence": e["sequence"]}
            for e in new_batch_entries
            if e.get("accession") and e.get("sequence")
        ]

        return sequences_for_scoring, new_batch_entries, all_results
    def _handle_esm_inpaint(self, tool_name: str, args: Dict[str, Any]):
        """
        占位版 esm_inpaint 处理逻辑：

        - 不再真正调用 ESM inpaint 脚本；
        - 对于指定的 accessions，直接复用当前 ranked（tmp_ranked.json）里的原始 scaffold 序列；
        - 仍然更新 inpaint_results.json（追加占位记录）；
        - 仍然调用 _score_and_update_state，对这些序列做 ProTrek mini 打分并更新 state；
        - 返回的 obs / extra 结构与原实现保持兼容。

        注意：如果某个 accession 在 tmp_ranked.json 中找不到对应序列，会抛错。
        """

        # 1) 决定要处理的 accession 列表（保持原逻辑）
        if "accessions" in args and args["accessions"]:
            acc_list = [a for a in args["accessions"] if a]
        else:
            ranked_list = self._load_json_if_exists(self.tmp_ranked_json, default=[])
            acc_list = [
                entry.get("accession")
                for entry in ranked_list[: getattr(self, "default_topk", 5)]
                if entry.get("accession")
            ]

        if not acc_list:
            raise RuntimeError(
                "No accession available to run esm_inpaint (placeholder). "
                "You should generate & score scaffolds first (so that tmp_ranked.json is non-empty), "
                "or pass explicit 'accessions' in tool arguments."
            )

        # 2) 从 tmp_ranked.json 里构建 accession -> sequence 映射
        ranked_list = self._load_json_if_exists(self.tmp_ranked_json, default=[])
        acc2seq: Dict[str, str] = {}
        if isinstance(ranked_list, list):
            for entry in ranked_list:
                acc = entry.get("accession")
                seq = entry.get("sequence")
                if acc and seq and acc not in acc2seq:
                    acc2seq[acc] = seq

        # 确保所有 acc 都能找到对应的 sequence
        missing = [acc for acc in acc_list if acc not in acc2seq]
        if missing:
            raise RuntimeError(
                f"esm_inpaint placeholder: some accessions have no sequence in tmp_ranked.json: {missing}. "
                "Please ensure scaffolds are generated and scored before calling esm_inpaint."
            )

        # 3) 载入已有的 inpaint_results.json，准备追加占位记录
        all_results = self._load_json_if_exists(self.inpaint_results_json, default=[])
        if not isinstance(all_results, list):
            all_results = []

        new_batch_entries: List[Dict[str, Any]] = []

        for acc in acc_list:
            seq = acc2seq[acc]

            entry_record = {
                "accession": acc,
                "sequence": seq,
                "summary": {
                    "original_sequence_len": len(seq) if seq else None,
                    "final_sequence_len": len(seq) if seq else None,
                    "debug_info": {
                        "placeholder": True,
                        "note": "esm_inpaint disabled in _handle_esm_inpaint; reused original scaffold sequence",
                    },
                },
            }

            new_batch_entries.append(entry_record)
            all_results.append(entry_record)

        # 写回 inpaint_results.json（文件副作用保持）
        self._save_json(self.inpaint_results_json, all_results)

        # 4) 构造 sequences_for_scoring，仍然是 [{accession, sequence}]
        sequences_for_scoring = [
            {"accession": e["accession"], "sequence": e["sequence"]}
            for e in new_batch_entries
            if e.get("accession") and e.get("sequence")
        ]

        # 5) 用这些原始 scaffold 序列做 ProTrek mini 打分并更新 state
        score_info = self._score_and_update_state(tool_name, sequences_for_scoring)

        # 6) 构造 obs / extra，结构尽量和原版一致，多加一行占位说明
        obs = self._build_observation(
            tool_name,
            score_info,
            extra_lines=[
                f"esm_inpaint_placeholder_mode=True",
                f"esm_inpaint_executed_for={len(acc_list)} accessions (no actual inpainting, reused original sequences)",
                f"inpaint_results_total_entries={len(all_results)}",
            ],
        )
        extra = {
            "accessions": acc_list,
            "added_entries": new_batch_entries,
            "score_info": score_info,
        }
        return obs, extra


# -------------------------
# quick demo of how you'd use it in a loop
# -------------------------
if __name__ == "__main__":
    requirement = (
        "Designed enzyme should bind (6S)-10-formyltetrahydrofolate "
        "and dTDP-4-amino-4,6-dideoxyglucose, act as an N-formyltransferase "
        "on dTDP-sugars, and maintain selectivity against off-target sugar substrates."
    )

    runtime = AgentRuntime(
        work_dir="./agent_workspace_test",
        requirement_text=requirement,
        protrek_env_python="/path/to/miniconda3/envs/protrek/bin/python",
    )
    runtime.default_topk = 5  # 你想保留多少条进入后续阶段

    def run_and_print(tc: dict, title: str):
        """统一跑一轮工具，只打印 obs + state。"""
        print(f"\n===== {title} =====")
        result = runtime.run_tool_call(tc)
        # 只打印要给模型看的 observation
        print("OBSERVATION:")
        print(result["content"])
        # 再打印当前 state
        print("STATE:")
        print(json.dumps(runtime.state, ensure_ascii=False, indent=2))

    
    # 1. function2seq
    tc_function2seq = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "function2seq",
            "arguments": {
                "text": "sugar N-formyltransferase that uses (6S)-10-formyltetrahydrofolate",
                "size": 16,
                "timeout": 30,
                "organism": None,
                "include_unreviewed": False,
            },
        },
    }
    run_and_print(tc_function2seq, "Stage1 - function2seq")

    # 2. pathway2seq
    tc_pathway2seq = {
        "id": "call_2",
        "type": "function",
        "function": {
            "name": "pathway2seq",
            "arguments": {
                "pathway": "dTDP-4-amino-4,6-dideoxyglucose biosynthesis",
                "size": 16,
                "timeout": 30,
                "organism": None,
                "include_unreviewed": False,
            },
        },
    }
    run_and_print(tc_pathway2seq, "Stage1 - pathway2seq")

    # 3. dna_binding2seq
    tc_dna_binding2seq = {
        "id": "call_3",
        "type": "function",
        "function": {
            "name": "dna_binding2seq",
            "arguments": {
                "text": "transcriptional regulator of sugar N-formyltransferase operon",
                "size": 16,
                "timeout": 30,
                "organism": None,
                "include_unreviewed": False,
            },
        },
    }
    run_and_print(tc_dna_binding2seq, "Stage1 - dna_binding2seq")

    # 4. domain2seq
    tc_domain2seq = {
        "id": "call_4",
        "type": "function",
        "function": {
            "name": "domain2seq",
            "arguments": {
                "domain": "C-terminal",
                "size": 16,
                "timeout": 30,
                "organism": None,
                "include_unreviewed": False,
            },
        },
    }
    run_and_print(tc_domain2seq, "Stage1 - domain2seq")

    # 5. go2seq
    tc_go2seq = {
        "id": "call_5",
        "type": "function",
        "function": {
            "name": "go2seq",
            "arguments": {
                "go_term": "sugar N-formyltransferase activity",
                "size": 16,
                "timeout": 30,
                "organism": None,
                "include_unreviewed": False,
            },
        },
    }
    run_and_print(tc_go2seq, "Stage1 - go2seq")

    # ---------- Stage 2: constraints（每个 constraint 工具都过一遍） ----------

    # 6. cofactor2constraints（自动接 inpaint + 打分）
    tc_cofactor = {
        "id": "call_6",
        "type": "function",
        "function": {
            "name": "cofactor2constraints",
            "arguments": {
                "cofactor": "(6S)-10-formyltetrahydrofolate; Mg2+",
            },
        },
    }
    run_and_print(tc_cofactor, "Stage2 - cofactor2constraints (+esm_inpaint)")

    # 7. motif2constraints（自动接 inpaint + 打分）
    tc_motif = {
        "id": "call_7",
        "type": "function",
        "function": {
            "name": "motif2constraints",
            "arguments": {
                "motif": "HXXXXD",
            },
        },
    }
    run_and_print(tc_motif, "Stage2 - motif2constraints (+esm_inpaint)")

    # 8. signal2constraints（自动接 inpaint + 打分）
    tc_signal = {
        "id": "call_8",
        "type": "function",
        "function": {
            "name": "signal2constraints",
            "arguments": {
                # 这里如果你的脚本支持 localization 之类的参数，可以补上
                # "localization": "periplasmic space",
            },
        },
    }
    run_and_print(tc_signal, "Stage2 - signal2constraints (+esm_inpaint)")

    # ---------- Stage 3: refinement + global scoring ----------

    # 9. 单独跑一次 esm_inpaint（从 tmp_ranked.json 里取 top_k accession）
    tc_inpaint = {
        "id": "call_9",
        "type": "function",
        "function": {
            "name": "esm_inpaint",
            "arguments": {}
        },
    }
    run_and_print(tc_inpaint, "Stage3 - esm_inpaint only")

    # 10. get_score：聚合当前所有已知序列再打分
    tc_get_score = {
        "id": "call_10",
        "type": "function",
        "function": {
            "name": "get_score",
            "arguments": {}
        },
    }
    run_and_print(tc_get_score, "Stage3 - get_score (aggregate scoring)")
