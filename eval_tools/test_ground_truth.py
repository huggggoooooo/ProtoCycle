#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
find_seq_in_parquet_and_molinstruction.py

功能：
1. 在 desc2seq parquet 里检查：
   - 是否有 ground_truth_seq 精确等于 target_seq 的样本
   - 一共有多少 ground_truth_seq 含有 'X'

2. 在 Mol-Instructions 的 protein_design.json 里检查：
   - 是否有任何一条 output 中的蛋白序列等于 target_seq
   - 一共有多少条样本的输出序列包含 'X'

其中 Mol-Instructions 的数据位于：
  /path/to/external_data/datasets--zjunlp--Mol-Instructions/Protein-oriented_Instructions/protein_design.json

output 的格式大致为：
  "output": "Your protein design is complete, and the amino acid sequence is\n```\nMSA...\n```"
"""

import ast
import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd


# ----------------- 路径配置 ----------------- #

PARQUET_PATH = "/path/to/ProtoCycle/data/proteinllm/desc2seq_agent_eval_clever_100.parquet"
MOLINSTR_JSON = "/path/to/external_data/datasets--zjunlp--Mol-Instructions/Protein-oriented_Instructions/protein_design.json"

# 你要检查的那条序列（带 X 的）
TARGET_SEQ = (
    "MKIILASKNQDKIREIGKILESSKRTLVTCNDIDIPEVEETGSTFVENAILKARSASLITGLAAIADDSGIEVDYLNAQPGI"
    "KSARYSGXNATNESNNFKLLKALDGVPYEKRKACYRCVIVYMRFPDDPFPVITSGSWEGYITEKLIGANGFGYDPLFYLPEY"
    "DKTSAQISSSXKNKISHRAKALKKLEDYFNK"
)


# ----------------- parquet 部分：ground_truth_seq ----------------- #

def extract_ground_truth_from_reward_model(val: Any) -> str:
    """
    从 parquet.reward_model 的这一格中抽 ground_truth 序列。
    可能的情况：
    - 已经是 dict
    - 是 JSON 字符串
    - 是 Python dict 的字符串表示（单引号），用 ast.literal_eval 解析
    """
    rm: Dict[str, Any] = {}

    if isinstance(val, dict):
        rm = val
    else:
        text = str(val)
        # 先尝试 JSON
        try:
            rm = json.loads(text)
        except json.JSONDecodeError:
            # 再尝试 literal_eval（处理单引号等）
            try:
                rm = ast.literal_eval(text)
            except Exception:
                rm = {}

    if not isinstance(rm, dict):
        rm = {}

    gt = (
        rm.get("ground_truth")
        or rm.get("ground_truth_seq")
        or rm.get("gt")
        or rm.get("sequence")
        or ""
    )
    return str(gt)


def check_parquet_for_target_and_X() -> None:
    print("======== [1] 检查 parquet 里的 ground_truth_seq ========")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[parquet] Loaded: {PARQUET_PATH}, rows = {len(df)}")

    # 加 ground_truth_seq 列
    df["ground_truth_seq"] = df["reward_model"].apply(extract_ground_truth_from_reward_model)

    # 精确匹配 target_seq
    mask_exact = df["ground_truth_seq"] == TARGET_SEQ
    n_exact = int(mask_exact.sum())
    print(f"[parquet] Exact match (ground_truth_seq == target_seq): {n_exact}")
    if n_exact > 0:
        print(df.loc[mask_exact, ["requirement_id", "requirement", "ground_truth_seq"]].to_string())

    # 统计含 'X' 的 ground_truth_seq
    mask_has_X = df["ground_truth_seq"].str.contains("X", na=False)
    n_has_X = int(mask_has_X.sum())
    print(f"[parquet] #rows with 'X' in ground_truth_seq: {n_has_X}")
    if n_has_X > 0:
        print("[parquet] 示例（最多前 10 条含 X 的 ground_truth_seq）：")
        print(df.loc[mask_has_X, ["requirement_id", "requirement", "ground_truth_seq"]].head(10).to_string())


# ----------------- Mol-Instructions 部分：output 里的序列 ----------------- #

def extract_seq_from_output(output_text: str) -> str:
    """
    从 Mol-Instructions 的 output 字符串中尽量抽出蛋白序列。

    规则优先级：
    1) 如果存在 ```...``` 代码块：
       - 提取所有代码块，用正则去掉非字母字符，只保留 A-Z
       - 取其中最长的那一个作为序列
    2) 如果没有代码块，就在全文中找大写连续字母串（长度>=20），取最长的那一个
    """
    if not isinstance(output_text, str):
        return ""

    text = output_text

    # 1) triple-backtick code blocks
    blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    candidates: List[str] = []
    for blk in blocks:
        s = re.sub(r"[^A-Za-z]", "", blk).upper()
        if s:
            candidates.append(s)

    if candidates:
        return max(candidates, key=len)

    # 2) fallback：全文找大写字母串
    upper_text = text.upper()
    aa_runs = re.findall(r"[A-Z]{20,}", upper_text)
    if not aa_runs:
        return ""
    return max(aa_runs, key=len)


def load_molinstruction_data(json_path: str) -> List[Dict[str, Any]]:
    """
    加载 Mol-Instructions 的 protein_design.json。
    通常是一个 list[dict]，但也兼容可能的 {'data': [...]} 或 {'examples': [...]} 结构。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # 尝试一些常见 key
        for key in ["data", "examples", "instances"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    # 其它奇怪格式，就包一层返回
    return [data]


def check_molinstruction_for_target_and_X() -> None:
    print("\n======== [2] 检查 Mol-Instructions protein_design.json ========")
    data = load_molinstruction_data(MOLINSTR_JSON)
    print(f"[MolInstr] Loaded: {MOLINSTR_JSON}, items = {len(data)}")

    n_with_seq = 0
    n_with_X = 0
    exact_match_indices: List[int] = []
    # 这里多存一个 seq，方便后面打印
    examples_with_X: List[Tuple[int, str, str]] = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        output = item.get("output", "")
        seq = extract_seq_from_output(output).strip()

        if not seq:
            continue

        n_with_seq += 1

        # 精确匹配
        if seq == TARGET_SEQ:
            exact_match_indices.append(idx)

        # 是否含 X
        if "X" in seq:
            n_with_X += 1
            if len(examples_with_X) < 10:
                # 记录一些例子，方便你看
                instr = str(item.get("instruction", ""))[:80].replace("\n", " ")
                examples_with_X.append((idx, instr, seq))

    print(f"[MolInstr] #items with extracted sequence: {n_with_seq}")
    print(f"[MolInstr] #items whose sequence contains 'X': {n_with_X}")

    if exact_match_indices:
        print(f"[MolInstr] Found exact TARGET_SEQ in {len(exact_match_indices)} items, indices: {exact_match_indices}")
    else:
        print("[MolInstr] No exact TARGET_SEQ found in extracted sequences.")

    if examples_with_X:
        print("\n[MolInstr] 示例：含 'X' 序列的前几条样本：")
        for idx, instr, seq in examples_with_X:
            print(f"  - idx={idx}, instruction preview: {instr!r}")
            print(f"    sequence_with_X: {seq}")



# ----------------- main ----------------- #

def main():
    check_parquet_for_target_and_X()
    check_molinstruction_for_target_and_X()


if __name__ == "__main__":
    main()
