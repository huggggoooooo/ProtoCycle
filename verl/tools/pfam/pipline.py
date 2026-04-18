#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
from typing import Any, Dict, List, Optional
from ray.util import rpdb

class AgentRuntime:
    """
    A lightweight executor that replays tool_calls one by one.

    - Maintains shared workspace files in work_dir.
    - Each tool_call is a dict like the model would emit in tool_calls[i].
    - We route to a handler based on function.name.
    - Each handler:
        * builds CLI args for the real script
        * runs it
        * updates files (scaffold.json, *_constraints.json, etc.)
        * returns a short natural-language observation string
          that you can stuff back into the conversation.
    """

    def __init__(
        self,
        work_dir: str,
        protrek_env_python: str = "/path/to/miniconda3/envs/protrek/bin/python",
    ):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        # canonical file paths that form the "blackboard"
        self.scaffold_json = os.path.join(self.work_dir, "scaffold.json")
        self.inpaint_results_json = os.path.join(self.work_dir, "inpaint_results.json")
        self.ranked_json = os.path.join(self.work_dir, "ranked.json")
        self.stage1_ranked_json = os.path.join(self.work_dir, "stage1_ranked.json")

        self.protrek_env_python = protrek_env_python
        self.default_topk = 5   # how many scaffold entries to process at most


        # script registry (fill with your real absolute paths)
        self.scripts = {
            "function2seq": "./verl/tools/pfam/function2seq.py",
            "pathway2seq": "./verl/tools/pfam/pathway2seq.py",
            "dna_binding2seq": "./verl/tools/pfam/dna_binding2seq.py",
            "domain2seq": "./verl/tools/pfam/domain2seq.py",
            "go2seq": "./verl/tools/pfam/go2seq.py",

            "build_constraints_from_uniprot": "./verl/tools/pfam/build_constraints_from_uniprot.py",
            "cofactor2constraints": "./verl/tools/pfam/cofactor2constraints.py",
            "motif2constraints": "./verl/tools/pfam/motif2constraints.py",
            "signal2constraints": "./verl/tools/pfam/signal2constraints.py",

            "esm_inpaint": "./verl/tools/pfam/esm/esm_constrain.py",
            "similarity_score": "./verl/tools/pfam/ProTrek/caculate_similarity_text_seq.py",

            # protrek-based ranking for stage1 (topk scaffolds)
            "protrek_stage1_rank": "./verl/tools/pfam/ProTrek/caculate_similarity_text_seq.py",
        }

        # map tool name -> handler function
        self.handlers = {
            # Stage1-like generators
            "function2seq": self._handle_scaffold_generator,
            "pathway2seq": self._handle_scaffold_generator,
            "dna_binding2seq": self._handle_scaffold_generator,
            "domain2seq": self._handle_scaffold_generator,
            "go2seq": self._handle_scaffold_generator,

            # rank/filter scaffolds
            "get_score": self._handle_stage1_rank,   # naming matches what you described
                                                     # ("protrek score and keep topk")

            # constraints builders (per accession)
            "cofactor2constraints": self._handle_update_constraints,
            "motif2constraints": self._handle_update_constraints,
            "signal2constraints": self._handle_update_constraints,

            # esm inpaint
            "esm_inpaint": self._handle_esm_inpaint,

            # final similarity scoring / ranking
            "similarity_score": self._handle_final_rank,
        }

    # --------------- small utilities ---------------

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
             "content": "<natural language observation string>",
             "ok": True/False,
             "extra": {... anything you want ...}
           }
        """
        fn = tool_call.get("function", {})
        name = fn.get("name")
        args = fn.get("arguments", {}) or {}

        handler = self.handlers.get(name)
        if handler is None:
            return {
                "content": f"ERROR: unknown tool '{name}'",
                "ok": False,
                "extra": {},
            }

        try:
            obs, extra = handler(name, args)
            return {
                "content": obs,
                "ok": True,
                "extra": extra,
            }
        except subprocess.CalledProcessError as e:
            return {
                "content": f"ERROR running {name}: {e}\nSTDERR:\n{e.stderr}",
                "ok": False,
                "extra": {},
            }
        except Exception as e:
            return {
                "content": f"ERROR (python-side) running {name}: {repr(e)}",
                "ok": False,
                "extra": {},
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


    # --------------- handlers ---------------

    def _handle_scaffold_generator(self, tool_name: str, args: Dict[str, Any]):
        """
        Handles function2seq / pathway2seq / dna_binding2seq / domain2seq / go2seq.

        Expected args keys:
          - one query arg that differs by tool:
              function2seq: query
              pathway2seq: pathway
              dna_binding2seq: text
              domain2seq: domain
              go2seq: go_term
          - optional: size, timeout, organism, include_unreviewed
        We always pass --json self.scaffold_json.
        Each tool must be able to: create (if missing) or append/merge into scaffold.json.
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

        size = str(args.get("size", 30))
        timeout = str(args.get("timeout", 30))
        organism = args.get("organism")
        include_unreviewed = args.get("include_unreviewed", False)

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

        # after running, read scaffold.json length for summary
        scaffolds = self._load_json_if_exists(self.scaffold_json, default=[])
        obs = (
            f"Tool call '{tool_name}' executed successfully. "
            f"Current scaffold pool has {len(scaffolds)} candidates."
        )
        extra = {
            "scaffold_count": len(scaffolds),
            "stdout": result.stdout.strip(),
        }
        return obs, extra

    # def _handle_stage1_rank(self, tool_name: str, args: Dict[str, Any]):
    #     """
    #     'get_score' in your description, i.e. run protrek_stage1_rank
    #     Args:
    #         topk: int
    #     Behavior:
    #         python protrek_stage1_rank.py --json scaffold.json --topk N --out scaffold.json
    #     Then we summarize best score etc (we'll assume the script prints that).
    #     """
    #     script_path = self.scripts["protrek_stage1_rank"]
    #     topk = str(args.get("topk", 2))
    #     text_arg = args.get("text", "")


    #     cmd = [
    #         "python",
    #         script_path,
    #         "--text", text_arg,
    #         "--json", self.scaffold_json,
    #         "--topk", topk,
    #         "--out", self.stage1_ranked_json,
    #     ]
    #     # run in protrek env because it's using ProTrek scoring
    #     result = self._run_subprocess(cmd, use_protrek_env=True, check=True)

    #     # reload scaffold pool after filtering
    #     scaffolds = self._load_json_if_exists(self.scaffold_json, default=[])
    #     obs = (
    #         f"Tool call 'get_score' executed successfully. "
    #         f"The operation completed and produced usable results. "
    #         f"We obtained {len(scaffolds)} candidate scaffolds ranked by score."
    #     )
    #     extra = {
    #         "kept": len(scaffolds),
    #         "stdout": result.stdout.strip(),
    #     }
    #     return obs, extra

    def _handle_stage1_rank(self, tool_name: str, args: Dict[str, Any]):
        """
        Stage-1 排序（你之前称为 get_score / protrek_stage1_rank）。
        现在的策略：
        - 如果 self.stage1_ranked_json 已存在，则认为 Stage-1 已完成，
            当前的 get_score 调用应当指向“最终排序”（_handle_final_rank），
            直接跳到最终打分并返回其 obs；
        - 否则按原逻辑执行 stage1_rank，并把观测基于 stage1_ranked_json。
        """
        # --- 0) 若已存在 stage1 排序产物，直接跳最终打分 ---
        if os.path.exists(self.stage1_ranked_json):
            # 复用传入的 args（允许携带 text/topk 等）
            return self._handle_final_rank(tool_name, args)

        # --- 1) 真正执行 stage1 排序 ---
        script_path = self.scripts["protrek_stage1_rank"]
        topk = str(args.get("topk", 2))
        text_arg = args.get("text", "")

        cmd = [
            "python",
            script_path,
            "--text", text_arg,
            "--json", self.scaffold_json,
            "--topk", topk,
            "--out", self.stage1_ranked_json,   # ★ 输出写到 stage1_ranked_json
        ]
        # 使用 ProTrek 环境执行
        result = self._run_subprocess(cmd, use_protrek_env=True, check=True)

        # --- 2) 基于 ranked 结果做观测（不再读 scaffold.json） ---
        ranked_list = self._load_json_if_exists(self.stage1_ranked_json, default=[])
        if isinstance(ranked_list, dict) and "results" in ranked_list:
            ranked_list = ranked_list["results"]

        kept = len(ranked_list)
        best_score = None
        if kept > 0:
            # 约定第一项为最高分；若脚本未排序，可自行再排序
            best_score = ranked_list[0].get("score")

        obs = (
            f"Stage-1 ranking completed. We obtained {kept} candidate scaffolds ranked by score."
            + (f" Top score: {best_score}." if best_score is not None else "")
        )
        extra = {
            "kept": kept,
            "stdout": (result.stdout or "").strip() if hasattr(result, "stdout") and result.stdout else "",
            "stage": "stage1_rank",
            "ranked_path": self.stage1_ranked_json,
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
        Apply cofactor/motif/signal constraints to up to self.default_topk
        accessions from stage1_ranked.json (NOT raw scaffold.json).

        For each accession in stage1_ranked.json:
          1. Ensure <acc>_constraints.json exists (lazy init from UniProt).
          2. Run the appropriate constraint updater script
             - cofactor2constraints   -> ... --in_json <file> --cofactor ...
             - motif2constraints      -> ... --json    <file> --desc ...
             - signal2constraints     -> ... --json    <file>

        We DO NOT require tool_call to pass accession.
        We DO take `cofactor` / `motif` text from tool_call args if relevant.
        """

        topk = getattr(self, "default_topk", 5)

        # read from ranked file now
        ranked_list = self._load_json_if_exists(self.stage1_ranked_json, default=[])
        if not ranked_list:
            print("stage1_ranked.json is empty; did you call get_score first?")
            raise RuntimeError("stage1_ranked.json is empty; did you call get_score first?")

        ranked_list = ranked_list[:topk]
        results = []

        for entry in ranked_list:
            acc = entry.get("accession")
            if not acc:
                continue

            # 1. make sure constraints exist (lazy init)
            cfile = self._ensure_constraints_for_accession(acc)

            # 2. build specific command based on tool_name
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
            result = self._run_subprocess(cmd, use_protrek_env=False, check=True)

            results.append({
                "accession": acc,
                "stdout": result.stdout.strip(),
            })

        obs = (
            f"Applied {tool_name} to {len(results)} ranked scaffold accessions "
            f"(source=stage1_ranked.json, max={topk}). "
            "Constraints were lazily initialized as needed."
        )
        extra = {
            "updated_accessions": [r["accession"] for r in results],
        }
        return obs, extra


    def _handle_esm_inpaint(self, tool_name: str, args: Dict[str, Any]):
        """
        Run esm_inpaint for accessions.
        Default accession list comes from stage1_ranked.json, not scaffold.json.
        Writes per-accession <acc>_inpaint.json (from esm_constrain.py --out),
        then merges into inpaint_results.json with explicit final 'sequence'.
        """

        script_path = self.scripts["esm_inpaint"]

        # Figure out which accessions to process:
        if "accessions" in args and args["accessions"]:
            acc_list = [a for a in args["accessions"] if a]
        else:
            # NEW: fallback to stage1_ranked.json, not scaffold.json
            ranked_list = self._load_json_if_exists(self.stage1_ranked_json, default=[])
            acc_list = [
                entry.get("accession")
                for entry in ranked_list[: getattr(self, "default_topk", 5)]
                if entry.get("accession")
            ]

        all_results = self._load_json_if_exists(self.inpaint_results_json, default=[])
        if not isinstance(all_results, list):
            all_results = []

        new_batch_entries = []

        for acc in acc_list:
            # ensure constraints exist
            cfile = self._ensure_constraints_for_accession(acc)

            out_file = os.path.join(self.work_dir, f"{acc}_inpaint.json")
            cmd = [
                "python",
                script_path,
                "--json", cfile,
                "--out", out_file,
            ]
            print(f"[Stage3:esm_inpaint] running for {acc}")
            self._run_subprocess(cmd, use_protrek_env=False, check=True)

            # load per-accession result JSON (the one we added with --out)
            try:
                with open(out_file, "r", encoding="utf-8") as f_in:
                    payload = json.load(f_in)
            except Exception as e:
                payload = {
                    "sequence": "",
                    "original_sequence": "",
                    "debug_info": {"error": f"failed to load {out_file}: {repr(e)}"},
                }

            final_seq = payload.get("sequence", "")
            original_seq = payload.get("original_sequence", "")
            debug_info = payload.get("debug_info", {})

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

        self._save_json(self.inpaint_results_json, all_results)

        obs = (
            f"esm_inpaint executed for {len(acc_list)} ranked accession(s). "
            f"inpaint_results.json now has {len(all_results)} total entries."
        )
        extra = {
            "added_entries": new_batch_entries,
        }
        return obs, extra


    def _handle_final_rank(self, tool_name: str, args: Dict[str, Any]):
        """
        Final similarity ranking between design_text and generated sequences.

        Steps:
        1. Load inpaint_results.json.
        2. Build tmp_for_scoring.json with accession+sequence.
        3. Call similarity_score script (protrek env).
        4. Read ranked.json produced by the script.
        5. Merge back summary info for context.
        6. Overwrite ranked.json with enriched info.
        7. RETURN: a dict whose 'content' is exactly the final picked sequence in plain text,
            so the model will see that as observation.
        """

        script_path = self.scripts["similarity_score"]

        design_text = args.get("text", "")
        topk = str(args.get("topk", 2))

        # 1. read inpaint_results.json
        inpaint_list = self._load_json_if_exists(self.inpaint_results_json, default=[])
        if not isinstance(inpaint_list, list):
            inpaint_list = []

        # 2. build tmp_for_scoring.json
        scoring_payload = []
        for entry in inpaint_list:
            acc = entry.get("accession")
            seq = entry.get("sequence", "")
            if acc and seq:
                scoring_payload.append({
                    "accession": acc,
                    "sequence": seq,
                })
        tmp_scoring_json = os.path.join(self.work_dir, "tmp_for_scoring.json")
        with open(tmp_scoring_json, "w", encoding="utf-8") as f_tmp:
            json.dump(scoring_payload, f_tmp, indent=2)

        # 3. run similarity_score script
        cmd = [
            "python",  # swapped to protrek python in _run_subprocess
            script_path,
            "--text", design_text,
            "--json", tmp_scoring_json,
            "--topk", topk,
            "--out", self.ranked_json,
        ]
        print("[Stage3:final_score] running:", " ".join(cmd))
        self._run_subprocess(cmd, use_protrek_env=True, check=True)

        # 4. load ranked.json from scorer
        ranked_raw = self._load_json_if_exists(self.ranked_json, default=[])
        if isinstance(ranked_raw, dict):
            if "results" in ranked_raw and isinstance(ranked_raw["results"], list):
                ranked_entries = ranked_raw["results"]
            else:
                ranked_entries = [ranked_raw]
        elif isinstance(ranked_raw, list):
            ranked_entries = ranked_raw
        else:
            ranked_entries = []

        # 5. merge summary info back in
        summary_by_acc = {e.get("accession"): e.get("summary", {}) for e in inpaint_list}

        enriched = []
        for r in ranked_entries:
            acc = r.get("accession")
            seq = r.get("sequence", "")
            score = r.get("score", None)
            enriched.append({
                "accession": acc,
                "sequence": seq,
                "score": score,
                "summary": summary_by_acc.get(acc, {}),
            })

        # 6. overwrite ranked.json with enriched info (keep nice merged view)
        self._save_json(self.ranked_json, enriched)

        # ---- 7. Build model-visible message ----
        # pick top1 (best-scoring)
        if len(enriched) > 0:
            best_seq = enriched[0].get("sequence", "")
            best_acc = enriched[0].get("accession", "")
            best_score = enriched[0].get("score", None)

            # 这是要喂回模型当 observation 的主要文字
            final_msg_lines = [
                "The final selected sequence after scoring is:",
                best_seq,
                ""
                f"Score: {best_score}"
            ]
            final_msg = "\n".join([line for line in final_msg_lines if line is not None])
        else:
            final_msg = (
                "No valid sequence was selected. Ranking produced no candidates."
            )

        # 构造给 ProteinDesignTool 的 result
        # 注意：只有 'content' 实际会被 ToolResponse 传回模型。
        obs = final_msg

        extra = {
            "ranked": enriched,
            "best_score": enriched[0].get("score", None) if enriched else None,
            "best_accession": enriched[0].get("accession", "") if enriched else "",
        }

        # 你想要的结构里还有 role / tool_calls，如果你喜欢也可以一起返回。
        # ProteinDesignTool.execute() 目前不会用它们，但可以保留以防后续扩展。
        wrapped_result_content = obs
        wrapped_extra = extra

        # 这里按你的需求，把这三个键暴露出来
        # run_tool_call() 期望 handler 返回 (obs, extra)
        # 然后 run_tool_call() 包成:
        # {
        #   "content": obs,
        #   "ok": True,
        #   "extra": extra,
        # }
        #
        # 所以我们只需要确保 obs 是 final_msg。
        # 如果你也希望 role/tool_calls 后面可能被用到，可以把它拼进 extra。
        return obs, {
            "role": "user",
            "tool_calls": None,
            **wrapped_extra
        }




# -------------------------
# quick demo of how you'd use it in a loop
# -------------------------
if __name__ == "__main__":
    runtime = AgentRuntime(
        work_dir="./agent_workspace_test",
        protrek_env_python="/path/to/miniconda3/envs/protrek/bin/python",
    )
    runtime.default_topk = 5  # 你想保留多少条进入后续阶段

    # 1. 生成初始 scaffold
    tc_go2seq = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "go2seq",
            "arguments": {
                "go_term": "sugar N-formyltransferase activity",
                "size": 30,
                "timeout": 30,
                "organism": None,
                "include_unreviewed": False,
            },
        },
    }
    print(json.dumps(runtime.run_tool_call(tc_go2seq), ensure_ascii=False, indent=2))

    # 2. get_score -> 写 stage1_ranked.json
    tc_filter = {
        "id": "call_2",
        "type": "function",
        "function": {
            "name": "get_score",
            "arguments": {}
        },
    }
    print(json.dumps(runtime.run_tool_call(tc_filter), ensure_ascii=False, indent=2))

    # 3a. 加 cofactor 约束 (lazy init constraints from UniProt if needed)
    tc_cofactor = {
        "id": "call_3",
        "type": "function",
        "function": {
            "name": "cofactor2constraints",
            "arguments": {
                "cofactor": "orotate"
            },
        },
    }
    print(json.dumps(runtime.run_tool_call(tc_cofactor), ensure_ascii=False, indent=2))

    # 3b. 加 motif 约束
    tc_motif = {
        "id": "call_4",
        "type": "function",
        "function": {
            "name": "motif2constraints",
            "arguments": {
                "motif": "PTB"
            },
        },
    }
    print(json.dumps(runtime.run_tool_call(tc_motif), ensure_ascii=False, indent=2))

    # 3c. 加 signal 约束
    tc_signal = {
        "id": "call_5",
        "type": "function",
        "function": {
            "name": "signal2constraints",
            "arguments": {}
        },
    }
    print(json.dumps(runtime.run_tool_call(tc_signal), ensure_ascii=False, indent=2))

    # 4. esm_inpaint（默认会读 stage1_ranked.json -> N条 accession）
    tc_inpaint = {
        "id": "call_6",
        "type": "function",
        "function": {
            "name": "esm_inpaint",
            "arguments": {}
        },
    }
    print(json.dumps(runtime.run_tool_call(tc_inpaint), ensure_ascii=False, indent=2))

    # 5. 最终打分+合并 summary
    tc_final = {
        "id": "call_7",
        "type": "function",
        "function": {
            "name": "similarity_score",
            "arguments": {
                "text": (
                    "Designed enzyme should bind (6S)-10-formyltetrahydrofolate "
                    "and dTDP-4-amino-4,6-dideoxyglucose, act as an N-formyltransferase "
                    "on dTDP-sugars, and maintain selectivity against off-target sugar substrates."
                ),
                "topk": 2
            },
        },
    }
    print(json.dumps(runtime.run_tool_call(tc_final), ensure_ascii=False, indent=2))


