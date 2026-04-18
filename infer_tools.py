#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_tools.py
- 从 YAML/JSON 配置文件加载并注册工具（ProteinDesignTool）
- 用 SFT 后模型跑推理 → 解析 <tool_call> → 派发执行 → 记录最终对话
- 优先使用 ProteinToolPromptDataset(data_files, tokenizer, config, processor=None) 导入数据
"""

import os
import re
import json
import argparse
from typing import Any, Dict, Tuple, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from omegaconf import OmegaConf, DictConfig
import asyncio
import uuid


# ========= Verl / 工具运行时 =========
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.tools.protein_tools import ProteinDesignSessionManager, ProteinDesignTool

import datasets
from typing import List
from verl.utils.dataset import RLHFDataset
import json
import datasets
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import shutil



# ===================== 全局参数（可被命令行覆盖） =====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PARQUET_PATH_DEFAULT = os.environ.get(
    "PARQUET_PATH",
    os.path.join(PROJECT_ROOT, "data/proteinllm/desc2seq_agent_eval.parquet"),
)
MODEL_DIR_DEFAULT    = os.environ.get("MODEL_DIR", "/path/to/model_checkpoint")
OUT_JSONL_DEFAULT    = os.environ.get(
    "OUT_JSONL",
    os.path.join(PROJECT_ROOT, "baseline_results/infer_results.jsonl"),
)
TOOLS_CFG_DEFAULT    = os.path.join(PROJECT_ROOT, "recipe/protein/tool_config.yaml")

GEN_CFG: Dict[str, Any] = dict(
    max_new_tokens=4096,
    do_sample=True,
    temperature=0.2,
    top_p=1.0,
    repetition_penalty=1.02,
)

MAX_STEPS_DEFAULT = 8
MAX_SAMPLES_DEFAULT = 100

_RUN_ID = f"ep-{uuid.uuid4().hex[:8]}"

# 你提供的 tool_prompt（保持逐字一致）
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

# 你给的 base_prompt（保持逐字一致）
BASE_PROMPT = """Agent protocol for each sample:
<think> analyze the design requirements and extract keywords by category.
<plan> select which Stage-1 tools to call (you may call multiple), decide whether to call any Stage-2 tools, and finally run Stage-3.

Strict ordering rule:
- Your reply MUST START with <think>...</think> followed immediately by <plan>...</plan>.
- Stage-1 (scaffold generation) FIRST. After finishing all intended Stage-1 tool calls, you MUST call get_score once to rank/filter the Stage-1 scaffolds before moving on.
- Stage-2 (optional constraint construction) MAY run after the Stage-1 + get_score checkpoint.
- Stage-3 (inpainting / final scoring) runs LAST.

<tool_call> whenever you decide to call a tool, output exactly one tool call with concrete arguments.
<answer> after all stages complete, output the final designed protein sequence.""".strip()


def cleanup_episode_tmp(episode_id: str, *, silent: bool = True):
    """
    删除本次 episode 的临时目录：<base_tmp_root>/<episode_id>/
    注意：仅在你确定不再需要调试痕迹时再调用。
    """
    mgr = _get_shared_manager()
    base = getattr(mgr, "base_tmp_root", None) or os.getenv("PROTEIN_TMP_ROOT", "/tmp/protein_design_tasks")
    work_dir = os.path.join(base, episode_id)
    try:
        shutil.rmtree(work_dir)
        if not silent:
            print(f"[CLEANUP] removed {work_dir}")
    except Exception as e:
        if not silent:
            print(f"[CLEANUP] failed to remove {work_dir}: {e}")
class ProteinToolPromptDataset(RLHFDataset):
    """
    只做一件事：把固定的 TOOL_PROMPT 加到现有 prompt[0].content 的最前面。
    其他字段（data_source/ability/reward_model/agent_name 等）保持不变。
    """

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # 读取 parquet
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]

            # 简单模仿你的分支结构：这里不区分 data_source，统一用 map_fn 做“前置拼接”
            dataframe = dataframe.map(self.map_fn, num_proc=16)

            dataframes.append(dataframe)

        # 拼接
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        前置追加 TOOL_PROMPT 到 row['prompt'][0]['content']。
        若原始样本没有 prompt/消息体，保持原样（避免抛错）。
        """
        p = row.get("prompt")
        if isinstance(p, list) and p and isinstance(p[0], dict) and "content" in p[0]:
            orig = p[0]["content"] or ""
            # 避免重复添加（可选防抖）
            if not orig.lstrip().startswith("You may call one or more functions to assist"):
                new_content = TOOL_PROMPT + "\n\n" + orig
                row = dict(row)
                row["prompt"] = [{"role": p[0].get("role", "user"), "content": new_content}]
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

def protein_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any,
    step_coef: float = 0.1,
) -> Dict[str, Any]:
    """
    This mirrors the expected dapo-style compute_score signature.

    Inputs:
      data_source   - e.g. "protein_design_stage3" (unused for now but kept for extensibility)
      solution_str  - model's decoded response (string, may include <plan>, <think>, <answer>)
      ground_truth  - reference AA seq (reward_model['ground_truth'])
      extra_info    - misc info (e.g. tool usage stats); we don't require it now
      step_coef     - weight for shaping reward

    Returns:
      dict with:
        "score": total_reward (float)
        "step_score": shaping reward (float)
        "outcome_score": seq identity reward (float)
        "pred_text": entire response (for debug/logging)
        "pred_seq": extracted final designed AA seq (for debug/logging)
    """

    # 1. step shaping reward
    # +1 if BOTH <plan> and <think> appear somewhere in the response
    # +1 if there's an <answer>...</answer>
    step_score = 0.0
    if _has_plan_and_think(solution_str):
        step_score += 1.0
    if _has_answer(solution_str):
        step_score += 1.0

    # 2. outcome reward (sequence similarity)
    pred_seq = _extract_answer_seq(solution_str)
    if pred_seq == "":
        # didn't produce final sequence in <answer> -> penalize hard
        outcome_score = -1.0
    else:
        identity = _seq_identity_ratio(pred_seq, ground_truth)
        # map identity in [0,1] to [-1,1]:
        # identity=0    => -1
        # identity=0.5  => 0
        # identity=1    => +1
        outcome_score = 2.0 * identity - 1.0

    total_reward = outcome_score + step_coef * step_score

    return {
        "score": float(total_reward),
        "step_score": float(step_score),
        "outcome_score": float(outcome_score),
        "pred_text": solution_str if solution_str is not None else "",
        "pred_seq": pred_seq,
    }




# ===================== 常用小工具函数 =====================
def first_user_text(messages):
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return None
    if not isinstance(messages, (list, tuple)):
        return None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else (str(c) if c is not None else None)
    return None

def restore_tools_field(val):
    try:
        import pandas as _pd
        if isinstance(val, float) and _pd.isna(val):
            return None
    except Exception:
        pass
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return _deep_listify(val)
    if hasattr(val, "tolist"):
        return _deep_listify(val.tolist())
    if isinstance(val, str):
        try:
            obj = json.loads(val)
            if isinstance(obj, (list, dict)):
                return _deep_listify(obj)
        except Exception:
            return None
    return None

def _deep_listify(x):
    import numpy as np
    if isinstance(x, dict):
        return {k: _deep_listify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_deep_listify(v) for v in x]
    if isinstance(x, np.ndarray):
        return _deep_listify(x.tolist())
    return x

# ===================== <tool_call> 解析器 =====================
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*({[\s\S]*?})\s*</tool_call>", re.I)

def parse_tool_call(text: str):
    if not isinstance(text, str):
        return None
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        name = obj.get("name")
        raw_args = obj.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) or {}
            except Exception:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        if not isinstance(name, str) or not name:
            return None
        return {"name": name, "arguments": args}
    except Exception:
        return None

# ===================== 基于配置的工具系统（核心） =====================
_CFG = None
_SHARED_MANAGER: ProteinDesignSessionManager | None = None
_TOOL_REGISTRY: Dict[str, ProteinDesignTool] = {}
_CREATED_FLAGS: Dict[Tuple[str, str], bool] = {}

def _ensure_parameters_shape(p: dict) -> dict:
    p = dict(p or {})
    p.setdefault("type", "object")
    p.setdefault("properties", {})
    p.setdefault("required", [])
    if "additionalProperties" not in p:
        p["additionalProperties"] = True
    return p

def init_tools_from_config(cfg_path: str):
    """
    读取 cfg（其结构为 {'tools': [ {class_name, config, tool_schema}, ... ]} ），
    初始化共享 SessionManager，并将每个条目注册为 ProteinDesignTool。
    """
    global _CFG, _SHARED_MANAGER, _TOOL_REGISTRY
    _CFG = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

    tools_list = (_CFG or {}).get("tools", []) or []
    if not isinstance(tools_list, list) or not tools_list:
        raise ValueError("Config must contain a non-empty list under key 'tools'.")

    _TOOL_REGISTRY.clear()
    _SHARED_MANAGER = None

    for idx, entry in enumerate(tools_list):
        if not isinstance(entry, dict):
            continue

        # 1) 解析 class/config/schema
        cls_name = entry.get("class_name", "")
        tool_cfg = dict(entry.get("config", {}) or {})
        schema_in = entry.get("tool_schema", {}) or {}

        # 2) 共享 SessionManager（只创建一次；从第一个条目里的 config.session_manager 提取）
        if _SHARED_MANAGER is None:
            sm = tool_cfg.get("session_manager", {}) or {}
            # 兜底默认
            base_tmp_root = sm.get("base_tmp_root", "./tmp/protein_design_tasks")
            protrek_env_python = sm.get(
                "protrek_env_python",
                os.environ.get(
                    "PROTREK_ENV_PYTHON",
                    "/path/to/miniconda3/envs/protrek/bin/python",
                ),
            )
            default_topk = int(sm.get("default_topk", 5))
            _SHARED_MANAGER = ProteinDesignSessionManager(
                base_tmp_root=base_tmp_root,
                protrek_env_python=protrek_env_python,
                default_topk=default_topk,
            )

        # 3) 标准化 tool_schema → OpenAIFunctionToolSchema
        #    支持两种形态：
        #    A) {"type":"function","function":{...}}
        #    B) {"name": "...", "description": "...", "parameters": {...}}
        if "function" in schema_in:
            fn = dict(schema_in["function"] or {})
        else:
            fn = dict(schema_in)

        name = fn.get("name")
        if not name:
            raise ValueError(f"[tools[{idx}]] tool_schema must provide a function name.")

        desc = fn.get("description", f"Protein design subtool '{name}'.")
        params = _ensure_parameters_shape(fn.get("parameters", {}))

        tool_schema = OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": name,
                "description": desc,
                "parameters": params,
            },
        )

        # 4) 组装 ProteinDesignTool 配置（以条目自带 config 为主，兜底常用字段）
        cfg_for_tool = dict(tool_cfg)
        # 替换为共享 SessionManager 对象
        cfg_for_tool["session_manager"] = _SHARED_MANAGER
        # 兜底项
        cfg_for_tool.setdefault("single_dir_per_instance", True)
        cfg_for_tool.setdefault("auto_round_subdir", False)
        cfg_for_tool.setdefault("round_dir_fmt", "round_{:03d}")
        # 若条目里没有 default_topk，则用共享 SM 的默认
        cfg_for_tool.setdefault("default_topk", getattr(_SHARED_MANAGER, "default_topk", 5))

        tool = ProteinDesignTool(config=cfg_for_tool, tool_schema=tool_schema)

        # 5) 去重处理：若名字重复，自动追加后缀
        reg_name = name
        if reg_name in _TOOL_REGISTRY:
            alt = f"{name}_{idx}"
            print(f"[WARN] Duplicate tool name '{name}' at index {idx}; register as '{alt}'.", flush=True)
            reg_name = alt

        _TOOL_REGISTRY[reg_name] = tool

    print(f"[INFO] Registered {len(_TOOL_REGISTRY)} tools from config.", flush=True)


def get_tools_schema_list() -> list[dict]:
    return [tool.tool_schema.model_dump() for tool in _TOOL_REGISTRY.values()]

def _get_shared_manager() -> ProteinDesignSessionManager:
    assert _SHARED_MANAGER is not None, "Call init_tools_from_config(cfg_path) first."
    return _SHARED_MANAGER

def _get_tool(name: str) -> ProteinDesignTool:
    if name not in _TOOL_REGISTRY:
        raise ValueError(f"Tool '{name}' not found. Available tools: {list(_TOOL_REGISTRY.keys())}")
    return _TOOL_REGISTRY[name]

async def _a_dispatch_tool(name: str, arguments: dict, raw_prompt=None) -> str:
    # 这里的 _episode_id 是我们自己传进来的
    episode_id = (arguments.pop("_episode_id", None)
                  or os.getenv("VERL_EPISODE_ID")
                  or _RUN_ID)

    round_index = arguments.pop("_round_index", None)

    try:
        tool = _get_tool(name)
    except ValueError as e:
        return f"Error: {str(e)}. Please use a valid tool name."
    
    key = (name, episode_id)

    # 第一次用这个 tool + episode_id：创建 session，并把 raw_prompt 带进去
    if not _CREATED_FLAGS.get(key):
        create_kwargs = {
            "episode_id": episode_id,
            "raw_prompt": raw_prompt,
        }
        _, _ = await tool.create(
            instance_id=episode_id,
            create_kwargs=create_kwargs,
        )
        _CREATED_FLAGS[key] = True

    # 后面只执行，不再 create；但仍然把 episode_id 往下传，好让 _resolve_session_key 一致
    if round_index is None:
        resp, step_reward, metrics = await tool.execute(
            instance_id=episode_id,
            parameters=arguments,
            episode_id=episode_id,
        )
    else:
        resp, step_reward, metrics = await tool.execute(
            instance_id=episode_id,
            parameters=arguments,
            episode_id=episode_id,
            round_index=int(round_index),
        )

    return resp.text or ""



def dispatch_tool(name: str, arguments: dict, raw_prompt=None) -> str:
    try:
        return asyncio.run(_a_dispatch_tool(name, dict(arguments or {}), raw_prompt=raw_prompt))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_a_dispatch_tool(name, dict(arguments or {}), raw_prompt=raw_prompt))
        raise



# ===================== 对话循环 =====================
def run_dialog_with_tools(model, tok, user0: str, max_steps: int, episode_id: str):
    messages = [{"role": "user", "content": user0}]
    gen_cfg = GenerationConfig(**GEN_CFG)
    tools_schema = get_tools_schema_list()

    for step in range(max_steps):
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools_schema,
            enable_thinking=True,
        )
        inputs = tok([prompt], return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_cfg)
        gen_text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False).strip()
        messages.append({"role": "assistant", "content": gen_text})

        if "</answer>" in gen_text.lower():
            break

        call = parse_tool_call(gen_text)
        if not call:
            break

        # ★★★ 关键：把本样本的 episode_id 注入到这一次工具调用的参数里
        raw_args = call.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) or {}
            except Exception:
                raw_args = {}
        args = dict(raw_args)
        args["_episode_id"] = episode_id       # 保证同一 sample 共享一个目录
        args["_round_index"] = step            # 如已开启 auto_round_subdir，可用

        obs_text = dispatch_tool(
            call["name"],
            args,
            raw_prompt=list(messages),  # 或 messages，推荐浅拷贝一下更稳
        )
        messages.append({"role": "tool", "content": obs_text})

    return messages


# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools_cfg", type=str, default=TOOLS_CFG_DEFAULT)
    parser.add_argument("--parquet", type=str, default=PARQUET_PATH_DEFAULT)
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR_DEFAULT)
    parser.add_argument("--out", type=str, default=OUT_JSONL_DEFAULT)
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES_DEFAULT)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--temperature", type=float, default=GEN_CFG["temperature"])
    parser.add_argument("--top_p", type=float, default=GEN_CFG["top_p"])
    parser.add_argument("--max_new_tokens", type=int, default=GEN_CFG["max_new_tokens"])
    parser.add_argument("--repetition_penalty", type=float, default=GEN_CFG["repetition_penalty"])
    args = parser.parse_args()

    GEN_CFG["temperature"] = args.temperature
    GEN_CFG["top_p"] = args.top_p
    GEN_CFG["max_new_tokens"] = args.max_new_tokens
    GEN_CFG["repetition_penalty"] = args.repetition_penalty

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    init_tools_from_config(args.tools_cfg)

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()

    # ========= 使用你给的签名来实例化数据集 =========
    rows: List[Dict[str, Any]] = []
    used_dataset = False
    try:
        # from recipe.protein.reward import ProteinToolPromptDataset  # 提供的类

        # 准备 DictConfig（尽量贴近你eval脚本里的 data.* 设置）
        ds_cfg: DictConfig = OmegaConf.create({
            "return_raw_chat": True,
            "prompt_key": "prompt",
            "filter_overlong_prompts": True,
            "truncation": "error",
            "max_prompt_length": 4096,
        })

        dataset = ProteinToolPromptDataset(
            data_files=args.parquet,      # ← 符合你给的签名
            tokenizer=tok,
            config=ds_cfg,
            processor=None,               # 可按需替换成你的 Processor
        )
        used_dataset = True



        total = len(dataset)
        if args.max_samples is not None:
            total = min(total, args.max_samples)

        for idx in range(total):
            item = dataset[idx]
            if isinstance(item, dict):
                if "messages" in item and item["messages"]:
                    rows.append({"messages": item["messages"]})
                elif "raw_prompt" in item and item["raw_prompt"]:
                    rows.append({"messages": [{"role": "user", "content": item["raw_prompt"]}]})
                elif "input" in item and item["input"]:
                    rows.append({"messages": [{"role": "user", "content": item["input"]}]})
    except Exception as e:
        print(f"[WARN] ProteinToolPromptDataset not used ({type(e).__name__}: {e}). Fallback to pandas parquet.", flush=True)
    
    print(rows)
    # ========= 回退：直接读 parquet =========
    if not used_dataset or not rows:
        df = pd.read_parquet(args.parquet)
        records = df.to_dict(orient="records")
        if args.max_samples is not None:
            records = records[:args.max_samples]
        rows = records

    # ========= 推理并保存 =========
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            messages = row.get("messages")
            if hasattr(messages, "tolist"):
                messages = messages.tolist()

            user0 = first_user_text(messages)
            if not user0 and isinstance(row.get("messages"), str):
                user0 = row["messages"]

            if not user0:
                print(f"[skip {i}] no first user content")
                continue
            
            sample_episode_id = f"ep-{uuid.uuid4().hex[:8]}-s{i}"
            final_dialog = run_dialog_with_tools(model, tok, user0, max_steps=args.max_steps, episode_id=sample_episode_id)

            print(f"\n===== SAMPLE {i} =====")
            for m in final_dialog:
                print(f"\n[{m['role'].upper()}]\n{m['content']}")

            rec = {"index": i, "first_user": user0, "final_dialog": final_dialog}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cleanup_episode_tmp(sample_episode_id)

    print(f"\nSaved results -> {args.out}")

if __name__ == "__main__":
    main()
