import datasets
from typing import List
from verl.utils.dataset import RLHFDataset
import json
import datasets
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# 你提供的 tool_prompt（保持逐字一致）
# ========= 工具说明（可以保留你现在这个 TOOL_PROMPT，不改也行） =========
TOOL_PROMPT = """You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "function2seq", "description": "Stage-1 scaffold generator. Given a SHORT functional or enzymatic description, retrieves natural protein sequences that roughly match the described activity.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Concise functional / enzymatic description used to guide scaffold retrieval."}}, "required": ["text"]}}}
{"type": "function", "function": {"name": "pathway2seq", "description": "Stage-1 scaffold generator. Uses pathway or metabolic-context keywords to retrieve sequences involved in the same biological process.", "parameters": {"type": "object", "properties": {"pathway": {"type": "string", "description": "Name or description of the metabolic / biosynthetic pathway."}}, "required": ["pathway"]}}}
{"type": "function", "function": {"name": "dna_binding2seq", "description": "Stage-1 scaffold generator for DNA-binding proteins, based on short DNA-binding keywords or motif names.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Short DNA-binding related term (e.g. motif/family name)."}}, "required": ["text"]}}}
{"type": "function", "function": {"name": "domain2seq", "description": "Stage-1 scaffold generator using a SINGLE domain / region keyword (e.g. a structural domain or C-terminal fragment).", "parameters": {"type": "object", "properties": {"domain": {"type": "string", "description": "Domain / region keyword to match (one concise phrase)."}}, "required": ["domain"]}}}
{"type": "function", "function": {"name": "go2seq", "description": "Stage-1 scaffold generator using GO-like functional phrases (e.g. 'glycolytic process', 'DNA repair'). It is generally robust and can be used as a fallback when other Stage-1 tools fail.", "parameters": {"type": "object", "properties": {"go_term": {"type": "string", "description": "GO-style functional phrase or process description."}}, "required": ["go_term"]}}}
{"type": "function", "function": {"name": "cofactor2constraints", "description": "Stage-2 constraint builder. Starting from top-ranked scaffolds, derives binding / coordination constraints for a specific cofactor or ligand and prepares them for further refinement.", "parameters": {"type": "object", "properties": {"cofactor": {"type": "string", "description": "Name or description of the cofactor / ligand to coordinate."}}, "required": ["cofactor"]}}}
{"type": "function", "function": {"name": "motif2constraints", "description": "Stage-2 constraint builder for sequence or structural motifs (e.g. catalytic motif, binding motif). It encodes the motif into constraints applied to top-ranked scaffolds.", "parameters": {"type": "object", "properties": {"motif": {"type": "string", "description": "Motif description to enforce (e.g. 'HXXXXD', 'glycine-rich loop')."}}, "required": ["motif"]}}}
{"type": "function", "function": {"name": "signal2constraints", "description": "Stage-2 constraint builder for localization / signal peptides (e.g. secretion or targeting signals). It modifies top-ranked scaffolds to include an appropriate signal segment.", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "esm_inpaint", "description": "Stage-3 refinement tool. Inpaints or rewrites parts of the current best scaffolds under all accumulated constraints, producing refined candidate sequences.", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "get_score", "description": "Scoring tool. Re-scores all currently known sequences against the textual design requirement using ProTrek and returns the current best design and its score.", "parameters": {"type": "object", "properties": {}, "required": []}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>""".strip()

# ========= 新版 global agent BASE_PROMPT（和你构造 eval 数据集时保持一致） =========
BASE_PROMPT = """
You are an agentic protein-design assistant. For each conversation, the user gives a natural-language design requirement for a protein, and you must explore tools step by step and finally output one amino-acid sequence.

Tools:
- You have several tools, with names such as:
  function2seq, pathway2seq, dna_binding2seq, domain2seq, go2seq,
  cofactor2constraints, motif2constraints, signal2constraints,
  esm_inpaint, get_score.
- Stage-1 tools (for scaffold generation) are:
  function2seq, pathway2seq, dna_binding2seq, domain2seq, go2seq.
- Stage-2 tools (constraints on top scaffolds) are:
  cofactor2constraints, motif2constraints, signal2constraints.
- Stage-3 tools (refinement / scoring) are:
  esm_inpaint, get_score.

Your message format:
At every assistant turn you must choose EXACTLY ONE of the following patterns:
(1) FIRST STEP (the very first assistant message after the user requirement),
(2) INTERMEDIATE STEP (later steps that still call tools),
(3) FINAL STEP (no more tool calls, only output the sequence).

================================
(1) FIRST ASSISTANT STEP
================================

The first assistant message in the conversation MUST have this structure:

<think>
[Requirement decomposition]
- general function: present/not mentioned/not specified — quote key phrases if present.
- pathway: ...
- co-factor: ...
- reaction: ...
- domain: ...
- dna-binding: ...
- go: ...
- motif: ...
- signal: ...

[Tool mapping]
- Which Stage-1 tools (from {function2seq, pathway2seq, dna_binding2seq, domain2seq, go2seq})
  are potentially useful, and with what type of arguments.
- Which Stage-2 tools (from {cofactor2constraints, motif2constraints, signal2constraints})
  might be used later and why.
- How Stage-3 tools (esm_inpaint, get_score) will be used near the end.

[Initial strategy]
- A high-level multi-step plan:
  * how you will explore scaffolds with Stage-1 tools,
  * when and why you will introduce constraints (Stage-2),
  * how you will use refinement/scoring tools (Stage-3),
  * under what conditions you will stop and output the final sequence.
</think>

<plan>
- For THIS first step only:
  - Choose exactly ONE Stage-1 tool to call now.
  - State clearly which tool you will call and what key arguments you will pass.
</plan>

<tool_call>
{"name": "ONE_STAGE1_TOOL_NAME", "arguments": { ... }}
</tool_call>

Rules for the first step:
- You MUST call exactly ONE Stage-1 tool in the first step.
- You MUST NOT call Stage-2 or Stage-3 tools in the first step.
- You MUST NOT output <answer> in the first step.

================================
(2) INTERMEDIATE STEPS (LATER)
================================

Any later assistant step that still calls a tool MUST follow this structure:

<think>
- Summarize what has happened so far, especially the latest tool OBSERVATION
  (scores, whether sequences were found, whether constraints worked, etc.).
- Decide whether to continue exploring scaffolds, add constraints, refine, or replan.
- Choose exactly ONE tool to call next and explain briefly why it is appropriate now.
- Explain how you choose its key arguments (e.g., simplify terms if previous calls failed).
</think>

<plan>
- A concise description of the NEXT action:
  - which SINGLE tool you will call,
  - what main arguments you will pass,
  - and what you expect to learn or improve.
</plan>

<tool_call>
{"name": "TOOL_NAME", "arguments": { ... }}
</tool_call>

Rules for intermediate steps:
- You MUST include exactly one <think>, one <plan>, and one <tool_call>.
- You MUST call exactly ONE tool per intermediate step.
- You MUST NOT include <answer> in an intermediate step.

================================
(3) FINAL STEP (STOP AND OUTPUT SEQUENCE)
================================

Before the final step:
- You MUST have called the scoring tool `get_score` at least once in this conversation.
- You MUST read its OBSERVATION to know the current best design and its sequence.

The FINAL assistant message MUST contain ONLY:

<answer>
AA_SEQUENCE
</answer>

Rules for the final answer:
- Do NOT include <think>, <plan>, or <tool_call> in the final message.
- The content inside <answer> must be exactly ONE continuous amino-acid sequence
  (letters from ACDEFGHIKLMNPQRSTVWY), typically copied from the best sequence
  reported by the most recent get_score OBSERVATION.
- Do NOT output placeholders like "SEQUENCE_IN_AMINO_ACIDS_ONLY".
- Do NOT add any extra commentary, explanation, or text inside <answer>.

================================
Heuristics and stopping criteria
================================

- Use several intermediate tool steps to:
  * generate scaffolds with Stage-1 tools,
  * optionally refine with Stage-2 constraints,
  * optionally refine with Stage-3 esm_inpaint,
  * and monitor the scores reported in OBSERVATIONs.

- If the global ProTrek score is high (for example > 15), OR appears to have
  plateaued around a reasonable level (for example ≥ 10 with only very small
  improvements over several rounds), you SHOULD:
  1) call `get_score` once to aggregate and re-score all known sequences, and then
  2) in the NEXT turn, produce a FINAL STEP containing only <answer>.

- If tools return OBSERVATIONs like “no new sequences” or num_sequences_scored=0,
  treat that call as FAILED:
  * in <think>, diagnose why (argument too long, wrong type, not a real motif/cofactor, etc.),
  * then adjust arguments (simplify or clean them) or switch to a more robust Stage-1 tool.

Hard constraints:
- NEVER call more than one tool in a single assistant message.
- NEVER mix <answer> with <tool_call>, <think>, or <plan>.
- ALWAYS use the FIRST STEP structure for the first assistant message,
  the INTERMEDIATE STEP structure for later tool-calling steps,
  and the FINAL STEP structure when you are ready to output the sequence.

The following text is the design requirement you must satisfy for this conversation. Design Requirements:

""".strip()


def build_user_prompt(requirement: str) -> str:
    """
    构造 user prompt，顺序：
      1) TOOL_PROMPT（工具说明）
      2) BASE_PROMPT（agent 协议）
      3) 任务说明 + requirement

    最终格式：
      TOOL_PROMPT

      BASE_PROMPT

      You are given design requirements for a protein sequence generation task.

      The following text is the design requirement you must satisfy for this conversation.

      <requirement>
    """
    parts: list[str] = []

    tp = TOOL_PROMPT.strip()
    if tp:
        parts.append(tp)

    # 第二段：agent 协议
    parts.append(BASE_PROMPT.strip())

    # 第三段：任务 & requirement
    parts.append(
        "You are given design requirements for a protein sequence generation task.\n\n"
        "The following text is the design requirement you must satisfy for this conversation.\n\n"
        f"{requirement.strip()}"
    )

    return "\n\n".join(parts).strip()


class ProteinToolPromptDataset(RLHFDataset):
    """
    用 row['requirement'] 重新构建一条标准的 user prompt：
      TOOL_PROMPT + BASE_PROMPT + 任务说明 + requirement
    不再在原有 prompt 前面简单拼接，避免旧 prompt 残留。
    """

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]

            # 对每一行用 map_fn 重写 prompt
            dataframe = dataframe.map(self.map_fn, num_proc=16)

            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        如果这一行有 requirement，就完全重建 prompt：
          [{ "role": "user", "content": build_user_prompt(requirement) }]
        否则保持原样。
        """
        requirement = row.get("requirement")
        if isinstance(requirement, str) and requirement.strip():
            new_user_content = build_user_prompt(requirement)
            row = dict(row)
            row["prompt"] = [{
                "role": "user",
                "content": new_user_content,
            }]
        return row




import re
from typing import Any, Dict

PLAN_RE   = re.compile(r"<plan>\s*(.+?)\s*</plan>", re.DOTALL | re.IGNORECASE)
THINK_RE  = re.compile(r"<think>\s*(.+?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL | re.IGNORECASE)

def _extract_tag(pattern, text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = pattern.search(text)
    return m.group(1).strip() if m else ""

def _has_plan_and_think(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(PLAN_RE.search(text) and THINK_RE.search(text))

def _has_answer(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(ANSWER_RE.search(text))

def _extract_answer_seq(text: str) -> str:
    return _extract_tag(ANSWER_RE, text)

def _seq_identity_ratio(seq_a: str, seq_b: str) -> float:
    if not seq_a or not seq_b:
        return 0.0
    min_len = min(len(seq_a), len(seq_b))
    if min_len == 0:
        return 0.0
    a_sub = seq_a[:min_len]
    b_sub = seq_b[:min_len]
    match_count = sum(1 for i in range(min_len) if a_sub[i] == b_sub[i])
    return match_count / float(min_len)


# def compute_score(
#     data_source: str,
#     solution_str: str,
#     ground_truth: str,
#     extra_info: Dict[str, Any] | None = None,
#     **kwargs,
# ) -> Dict[str, Any]:
#     """
#     自定义 reward 函数，供 verl 的 NaiveRewardManager 调用。

#     参数说明（由 NaiveRewardManager 传入）：
#     - data_source: 这一条样本的 data_source（来自 non_tensor_batch[reward_fn_key]）
#     - solution_str: 模型的 response 文本（已经 decode 好的字符串）
#     - ground_truth: 数据里提供的 ground_truth（这里不用）
#     - extra_info: 数据里的 extra_info dict，加上 num_turns / rollout_reward_scores 等
#     - **kwargs: 来自 config.reward_model.custom_reward_function.reward_kwargs 的额外参数（现在不用）

#     返回值：
#     - 必须包含 "score" 这个键，其它键会进入 reward_extra_info 里，方便你 debug。
#     """
#     # breakpoint()
#     if solution_str is None:
#         solution_str = ""

#     # 判断是否包含 <answer> ... </answer> 这对 tag
#     has_open_tag = "<answer>" in solution_str
#     has_close_tag = "</answer>" in solution_str
#     has_answer_tag = has_open_tag and has_close_tag

#     score = 1.0 if has_answer_tag else 0.0

#     # extra_info 里可以顺手把 num_turns 带出来看看
#     num_turns = None
#     if isinstance(extra_info, dict):
#         num_turns = extra_info.get("num_turns", None)

#     # NaiveRewardManager 要求：
#     # - 如果返回的是 dict，必须有 "score" 键
#     # - 其它键会被收集到 reward_extra_info，用于打印 / 记录
#     return {
#         "score": float(score),
#         "has_answer_tag": bool(has_answer_tag),
#         "num_turns": num_turns,
#         "data_source": data_source,
#         # 这里 ground_truth 也可以顺手带一下，方便 debug
#         # 但注意别太长（比如大段文本），否则 log 会很啰嗦
#     }


import re
from typing import Any, Dict, List, Optional

TOOL_RESP_RE = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL | re.IGNORECASE
)

# 一些简单的正则把有用信息扒出来
NUM_SCORED_RE = re.compile(r"num_sequences_scored:\s*(\d+)", re.IGNORECASE)

# 改成更严格的小数匹配，避免把末尾句号吃进去
DECIMAL_PATTERN = r"[-+]?\d+(?:\.\d+)?"

BEST_PROTREK_RE = re.compile(
    rf"best_ProTrek_this_round:\s*({DECIMAL_PATTERN})", re.IGNORECASE
)
GLOBAL_PROTREK_RE = re.compile(
    rf"global_best_ProTrek:\s*({DECIMAL_PATTERN})", re.IGNORECASE
)

RAN_SUCCESS_RE = re.compile(
    r"tool\s+'(\w+)'\s+ran successfully", re.IGNORECASE
)


def _safe_float(s: Optional[str]) -> Optional[float]:
    """更鲁棒的 float 解析，防止 '19.401.' 之类把进程搞挂。"""
    if s is None:
        return None
    s = s.strip()
    # 再保险一点，把尾部标点去掉
    s = s.rstrip(".,;:)")
    try:
        return float(s)
    except ValueError:
        return None


def _extract_tool_events(solution_str: str) -> List[Dict[str, Any]]:
    """从 solution_str 里抽出每个 <tool_response> 的结构化信息。"""
    events: List[Dict[str, Any]] = []
    for block in TOOL_RESP_RE.findall(solution_str or ""):
        ev: Dict[str, Any] = {"raw": block}

        # 是否成功
        m = RAN_SUCCESS_RE.search(block)
        if m:
            ev["tool_name"] = m.group(1)
            ev["success"] = True
        else:
            # 失败时日志一般有 "was called with ..." / "failed" 之类
            ev["success"] = False

        # num_scored
        m = NUM_SCORED_RE.search(block)
        try:
            ev["num_scored"] = int(m.group(1)) if m else 0
        except ValueError:
            ev["num_scored"] = 0

        # best_protrek_this_round
        m = BEST_PROTREK_RE.search(block)
        ev["best_protrek_round"] = _safe_float(m.group(1)) if m else None

        # global_best_ProTrek
        m = GLOBAL_PROTREK_RE.search(block)
        ev["global_best_protrek"] = _safe_float(m.group(1)) if m else None

        events.append(ev)

    return events



def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    if solution_str is None:
        solution_str = ""

    # ---------- 1) 格式/协议奖励 ----------
    has_think = "<think>" in solution_str and "</think>" in solution_str
    has_plan = "<plan>" in solution_str and "</plan>" in solution_str
    has_answer = "<answer>" in solution_str and "</answer>" in solution_str

    R_format = 0.0
    if has_think and has_plan:
        R_format += 0.5    # 遵守 <think>/<plan> 协议
    if has_answer:
        R_format += 0.5    # 有最终 <answer>

    # ---------- 2) 工具调用奖励（你关心的这块） ----------
    events = _extract_tool_events(solution_str)
    R_tools = 0.0
    best_global_protrek = None

    for ev in events:
        tool = ev.get("tool_name", "")
        success = ev.get("success", False)
        num_scored = ev.get("num_scored", 0)
        best_round = ev.get("best_protrek_round")
        best_global = ev.get("global_best_protrek")

        if best_global is not None:
            best_global_protrek = (
                best_global
                if best_global_protrek is None
                else max(best_global_protrek, best_global)
            )

        # 只看几类 Stage-1 工具
        if tool in {"function2seq", "go2seq", "pathway2seq", "domain2seq", "dna_binding2seq"}:
            if success and num_scored > 0:
                # 成功产出 scaffold：奖励
                R_tools += 0.3
                # 如果这一轮 best ProTrek 挺高，再给一点额外奖励（argument 好 → tool 回应好）
                if best_round is not None:
                    if best_round >= 15:
                        R_tools += 0.3
                    elif best_round >= 10:
                        R_tools += 0.15
            else:
                # 白跑一轮，没有拿到任何序列 → 小罚
                R_tools -= 0.1

        # get_score 成功跑完也奖励一点，鼓励 pipeline 收尾
        if tool == "get_score" and success:
            R_tools += 0.3

    # ---------- 3) ProTrek 作为弱 outcome 信号 ----------
    R_protrek = 0.0
    if best_global_protrek is not None:
        # 把 [0, 30+] 映射到大概 [0, 1]，再裁剪
        R_protrek = max(0.0, min(best_global_protrek / 20.0, 1.0))
    else:
        # 完全没产生任何 ProTrek 分数 → 小罚
        R_protrek = -0.2

    # ---------- 4) 效率奖励：回合太多适当扣一点 ----------
    R_eff = 0.0
    num_turns = None
    if isinstance(extra_info, dict):
        num_turns = extra_info.get("num_turns", None)
    try:
        if num_turns is not None:
            num_turns = int(num_turns)
            # 超过 6 个 turn 开始扣分
            if num_turns > 6:
                R_eff = -0.05 * (num_turns - 6)
    except Exception:
        pass

    # ---------- 5) 合成总分 ----------
    # 这里权重你可以后面慢慢调
    w_format = 0.5
    w_tools = 1.0
    w_protrek = 0.5
    w_eff = 1.0

    total_reward = (
        w_format * R_format
        + w_tools * R_tools
        + w_protrek * R_protrek
        + w_eff * R_eff
    )

    return {
        "score": float(total_reward),

        # 下面这些字段都会进 reward_extra_info，方便你在 log 里排查
        "R_format": float(R_format),
        "R_tools": float(R_tools),
        "R_protrek": float(R_protrek),
        "R_eff": float(R_eff),
        "best_global_protrek": float(best_global_protrek or 0.0),
        "num_tool_events": len(events),
        "num_turns": num_turns,
    }


from typing import Dict, Any, Optional

# def compute_score_has_answer(
#     data_source: str,
#     solution_str: str,
#     ground_truth: str,
#     extra_info: Optional[Dict[str, Any]] = None,
#     **kwargs,
# ) -> Dict[str, Any]:
#     if solution_str is None:
#         solution_str = ""

#     # ---------- 1) 协议检测 ----------
#     has_think = "<think>" in solution_str and "</think>" in solution_str
#     has_plan = "<plan>" in solution_str and "</plan>" in solution_str
#     has_answer = "<answer>" in solution_str and "</answer>" in solution_str

#     R_format = 0.0
#     if has_think and has_plan:
#         R_format += 0.5    # 遵守 <think>/<plan> 协议
#     if has_answer:
#         R_format += 0.5    # 有最终 <answer>

#     # ---------- 2) 工具调用奖励 ----------
#     events = _extract_tool_events(solution_str)
#     R_tools = 0.0
#     best_global_protrek = None

#     # 限制：只对前 N 次成功的 Stage-1 调用给奖励，避免“刷工具”
#     MAX_STAGE1_REWARDED_CALLS = 3
#     stage1_rewarded_calls = 0

#     for ev in events:
#         tool = ev.get("tool_name", "")
#         success = ev.get("success", False)
#         num_scored = ev.get("num_scored", 0)
#         best_round = ev.get("best_protrek_round")
#         best_global = ev.get("global_best_protrek")

#         if best_global is not None:
#             best_global_protrek = (
#                 best_global
#                 if best_global_protrek is None
#                 else max(best_global_protrek, best_global)
#             )

#         # 只看几类 Stage-1 工具
#         if tool in {"function2seq", "go2seq", "pathway2seq", "domain2seq", "dna_binding2seq"}:
#             if success and num_scored > 0:
#                 # 只对前 MAX_STAGE1_REWARDED_CALLS 次成功调用计奖励
#                 if stage1_rewarded_calls < MAX_STAGE1_REWARDED_CALLS:
#                     R_tools += 0.3
#                     if best_round is not None:
#                         if best_round >= 15:
#                             R_tools += 0.3
#                         elif best_round >= 10:
#                             R_tools += 0.15
#                     stage1_rewarded_calls += 1
#                 # 超过的成功调用不再加分，只是为了探索，不鼓励刷分
#             else:
#                 # 白跑一轮，没有拿到任何序列 → 小罚
#                 R_tools -= 0.1

#         # get_score 成功跑完也奖励一点，鼓励 pipeline 收尾
#         if tool == "get_score" and success:
#             R_tools += 0.3

#     # ---------- 3) ProTrek 作为弱 outcome 信号 ----------
#     R_protrek = 0.0
#     if best_global_protrek is not None:
#         # 把 [0, 30+] 映射到大概 [0, 1]，再裁剪
#         R_protrek = max(0.0, min(best_global_protrek / 20.0, 1.0))
#     else:
#         # 完全没产生任何 ProTrek 分数 → 小罚
#         R_protrek = -0.2

#     # ---------- 4) 效率奖励：强力惩罚“很多轮” ----------
#     R_eff = 0.0
#     num_turns = None
#     if isinstance(extra_info, dict):
#         num_turns = extra_info.get("num_turns", None)

#     try:
#         if num_turns is not None:
#             num_turns = int(num_turns)
#             TARGET_TURNS = 4  # 期望在 ~4 个 turn 内解决

#             if num_turns <= TARGET_TURNS:
#                 # 回合数少一点给点小奖励（最多 +0.15 左右）
#                 R_eff = 0.05 * (TARGET_TURNS - num_turns)
#             else:
#                 # 超过 TARGET_TURNS 每多 1 回合罚 0.1
#                 # 例如 8 回合：R_eff = -0.4；16 回合：R_eff = -1.2
#                 R_eff = -0.1 * (num_turns - TARGET_TURNS)
#     except Exception:
#         pass

#     # ---------- 5) 合成总分 ----------
#     w_format = 0.5
#     w_tools = 1.0
#     w_protrek = 0.5
#     w_eff = 1.0

#     total_reward = (
#         w_format * R_format
#         + w_tools * R_tools
#         + w_protrek * R_protrek
#         + w_eff * R_eff
#     )

#     # ---------- 6) 关键改动：没有 <answer> 一律视为失败 ----------
#     # 必须要有 <answer> 才能拿到上面的奖励；否则直接给一个负分
#     # 这个常数可以自己调，比如 -1.0 / -0.5 / 0.0
#     if not has_answer:
#         total_reward = -0.5

#     return {
#         "score": float(total_reward),

#         # 这些字段仍然记录下来，方便在 log 里分析
#         "R_format": float(R_format),
#         "R_tools": float(R_tools),
#         "R_protrek": float(R_protrek),
#         "R_eff": float(R_eff),
#         "best_global_protrek": float(best_global_protrek or 0.0),
#         "num_tool_events": len(events),
#         "num_turns": num_turns,
#         "has_answer": has_answer,
#     }

def compute_score_has_answer(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    if solution_str is None:
        solution_str = ""

    # ---------- 0) 一些超参数 ----------
    # 回合数相关：希望尽量在 4 回合内解决，超过 8 / 12 回合强烈惩罚
    TARGET_TURNS = 4
    SOFT_MAX_TURNS = 8
    HARD_MAX_TURNS = 12

    # 每多一回合的线性惩罚（> TARGET_TURNS 部分）
    PER_EXTRA_TURN_COST = 0.25
    # 超过 SOFT_MAX_TURNS 的额外大罚
    SOFT_OVER_PENALTY = 0.5
    # 超过 HARD_MAX_TURNS 的额外大罚
    HARD_OVER_PENALTY = 1.0

    # 没有 <answer> 的统一惩罚
    NO_ANSWER_PENALTY = -0.5

    # 各项权重（加重效率这一项）
    w_format = 0.5
    w_tools = 1.0
    w_protrek = 0.5
    w_eff = 1.5

    # ---------- 1) 协议检测 ----------
    has_think = "<think>" in solution_str and "</think>" in solution_str
    has_plan = "<plan>" in solution_str and "</plan>" in solution_str
    has_answer = "<answer>" in solution_str and "</answer>" in solution_str

    R_format = 0.0
    if has_think and has_plan:
        R_format += 0.5    # 遵守 <think>/<plan> 协议
    if has_answer:
        R_format += 0.5    # 有最终 <answer>

    # ---------- 2) 工具调用奖励（保留你原来的 Stage-1 逻辑） ----------
    events = _extract_tool_events(solution_str)
    R_tools = 0.0
    best_global_protrek = None

    # 限制：只对前 N 次成功的 Stage-1 调用给奖励，避免“刷工具”
    MAX_STAGE1_REWARDED_CALLS = 3
    stage1_rewarded_calls = 0

    for ev in events:
        tool = ev.get("tool_name", "")
        success = ev.get("success", False)
        num_scored = ev.get("num_scored", 0)
        best_round = ev.get("best_protrek_round")
        best_global = ev.get("global_best_protrek")

        if best_global is not None:
            best_global_protrek = (
                best_global
                if best_global_protrek is None
                else max(best_global_protrek, best_global)
            )

        # 只看几类 Stage-1 工具
        if tool in {"function2seq", "go2seq", "pathway2seq", "domain2seq", "dna_binding2seq"}:
            if success and num_scored > 0:
                # 只对前 MAX_STAGE1_REWARDED_CALLS 次成功调用计奖励
                if stage1_rewarded_calls < MAX_STAGE1_REWARDED_CALLS:
                    R_tools += 0.3
                    if best_round is not None:
                        if best_round >= 15:
                            R_tools += 0.3
                        elif best_round >= 10:
                            R_tools += 0.15
                    stage1_rewarded_calls += 1
                # 超过的成功调用不再加分，只是为了探索，不鼓励刷分
            else:
                # 白跑一轮，没有拿到任何序列 → 小罚
                R_tools -= 0.1

        # get_score 成功跑完也奖励一点，鼓励 pipeline 收尾
        if tool == "get_score" and success:
            R_tools += 0.3

    # ---------- 3) ProTrek 作为弱 outcome 信号 ----------
    R_protrek = 0.0
    if best_global_protrek is not None:
        # 把 [0, 30+] 映射到大概 [0, 1]，再裁剪
        R_protrek = max(0.0, min(best_global_protrek / 20.0, 1.0))
    else:
        # 完全没产生任何 ProTrek 分数 → 小罚
        R_protrek = -0.2

    # ---------- 4) 效率奖励：强烈惩罚“很多轮” ----------
    R_eff = 0.0
    num_turns = None
    if isinstance(extra_info, dict):
        num_turns = extra_info.get("num_turns", None)

    try:
        if num_turns is not None:
            num_turns = int(num_turns)

            if num_turns <= TARGET_TURNS:
                # 回合数少一点给点小奖励（最多 +0.3 左右）
                R_eff = 0.1 * (TARGET_TURNS - num_turns)
            else:
                # 超过 TARGET_TURNS 每多 1 回合罚 PER_EXTRA_TURN_COST
                over = num_turns - TARGET_TURNS
                R_eff = -PER_EXTRA_TURN_COST * over

                # 再加两级台阶的额外惩罚
                if num_turns > SOFT_MAX_TURNS:
                    R_eff -= SOFT_OVER_PENALTY
                if num_turns > HARD_MAX_TURNS:
                    R_eff -= HARD_OVER_PENALTY
    except Exception:
        pass

    # ---------- 5) 合成总分 ----------
    total_reward = (
        w_format * R_format
        + w_tools * R_tools
        + w_protrek * R_protrek
        + w_eff * R_eff
    )

    # ---------- 6) 关键改动：没有 <answer> 一律视为失败 ----------
    if not has_answer:
        total_reward = NO_ANSWER_PENALTY

    return {
        "score": float(total_reward),

        # 这些字段仍然记录下来，方便在 log 里分析
        "R_format": float(R_format),
        "R_tools": float(R_tools),
        "R_protrek": float(R_protrek),
        "R_eff": float(R_eff),
        "best_global_protrek": float(best_global_protrek or 0.0),
        "num_tool_events": len(events),
        "num_turns": num_turns,
        "has_answer": has_answer,
    }
