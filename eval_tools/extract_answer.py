#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_eval_results.py

功能：
- 从 desc2seq 的原始 eval parquet 和 对应结果 jsonl 中抽取：
  * requirement（以 parquet 里的为主）
  * ground_truth 序列（来自 reward_model['ground_truth']）
  * 模型预测序列（只从 final_dialog 最后一个 message 的 <answer></answer> 中抽）
- 对 parquet.requirement 和 jsonl 里的 requirement 做相似度对比（已简化，只保留 requirement 文本）
- 把以上信息写入一个 CSV 方便后续算指标

输出列：
  row_idx
  jsonl_index
  requirement_parquet
  requirement_from_jsonl
  ground_truth_seq
  pred_seq
  has_tag        # 1: 有 <answer> 且解析出了非空序列；0: 否则

用法示例：
python extract_eval_results.py \
  --parquet /path/to/ProtoCycle/data/proteinllm/desc2seq_agent_eval_clever_100.parquet \
  --jsonl   /path/to/your/eval_results.jsonl \
  --out_csv /path/to/merged_eval_results.csv
"""

import argparse
import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ----------------- requirement 相关 ----------------- #

def extract_requirement_from_prompt(prompt: str) -> Optional[str]:
    """
    从 first_user 的 content 里抽 requirement：
    规则：找到那句固定前缀后面的内容，再用一些常见的 stop marker 截断掉 BASE_PROMPT。
    如果找不到 marker，返回 None。
    """
    if not isinstance(prompt, str):
        return None

    marker = "The following text is the design requirement you must satisfy for this conversation."
    idx = prompt.find(marker)
    if idx == -1:
        return None

    # marker 之后的部分
    sub = prompt[idx + len(marker):]
    sub = sub.lstrip()

    # 这些短语一般属于 BASE_PROMPT，用来截断 requirement
    stop_markers = [
        "\n\nOverall agent protocol for each sample",
        "\n\nOverall agent protocol",
        "\n\nYou must:",
        "\n\nDesign stages:",
    ]

    end = len(sub)
    for sm in stop_markers:
        j = sub.find(sm)
        if j != -1 and j < end:
            end = j

    req = sub[:end].strip()
    return req or None


# ----------------- 序列抽取相关 ----------------- #

def extract_sequence_from_answer_content(content: str) -> Tuple[str, bool]:
    """
    只从 final_dialog 最后一个 content 中的 <answer>...</answer> 里提取预测序列：

    规则：
      1. 必须存在 <answer>...</answer>（大小写不敏感）
      2. 只看 tag 里面的内容，不再 fallback 到全文
      3. 把 tag 内所有字母字符 [A-Za-z] 提出来并拼成一个串，转成大写
      4. 如果清洗后是非空串，则认为成功提取，返回 (seq, True)
         否则返回 ("", False)

    返回:
      (sequence_str, has_tag_and_non_empty: bool)
    """
    if not isinstance(content, str):
        return "", False

    # 查找 <answer>...</answer>（大小写不敏感）
    m = re.search(r"<answer>([\s\S]*?)</answer>", content, flags=re.IGNORECASE)
    if not m:
        return "", False

    inner = m.group(1)

    # 只保留字母字符，再转成大写
    seq = re.sub(r"[^A-Za-z]", "", inner or "")
    seq = seq.upper().strip()

    if not seq:
        return "", False

    return seq, True


# ----------------- ground truth 抽取 ----------------- #

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

    # 主键 ground_truth，顺便兼容一下可能的变体
    gt = (
        rm.get("ground_truth")
        or rm.get("ground_truth_seq")
        or rm.get("gt")
        or rm.get("sequence")
        or ""
    )
    return str(gt)


# ----------------- 主流程 ----------------- #

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main(parquet_path: str, jsonl_path: str, out_csv: str) -> None:
    # 读 parquet
    df = pd.read_parquet(parquet_path)
    n_parquet = len(df)

    # 读 jsonl
    jsonl_records = load_jsonl(jsonl_path)
    n_jsonl = len(jsonl_records)

    if n_parquet != n_jsonl:
        print(
            f"[警告] parquet 行数({n_parquet}) 与 jsonl 行数({n_jsonl}) 不一致，"
            f"后面会按 min(len) 对齐。"
        )

    n = min(n_parquet, n_jsonl)

    rows = []

    for i in range(n):
        row_parquet = df.iloc[i]
        rec_jsonl = jsonl_records[i]

        # 1) requirement（parquet & jsonl）
        requirement_parquet = str(row_parquet.get("requirement", ""))

        # jsonl 里的 first_user: 通常是一个 list[message]
        first_user_msgs = rec_jsonl.get("first_user", [])
        if isinstance(first_user_msgs, dict):
            # 某些情况下可能直接是 dict
            first_user_msgs = [first_user_msgs]

        first_user_text_parts = []
        for msg in first_user_msgs:
            if isinstance(msg, dict):
                first_user_text_parts.append(str(msg.get("content", "")))
        first_user_text = "\n".join(first_user_text_parts)

        requirement_from_jsonl = (
            extract_requirement_from_prompt(first_user_text) or ""
        )

        # 2) ground truth 序列
        gt_seq = extract_ground_truth_from_reward_model(
            row_parquet.get("reward_model")
        )

        # 3) 预测序列：严格只看 final_dialog 最后一个消息里的 <answer>...</answer>
        final_dialog = rec_jsonl.get("final_dialog", [])
        pred_seq = ""
        has_tag = False
        if isinstance(final_dialog, list) and final_dialog:
            last_msg = final_dialog[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
                pred_seq, has_tag = extract_sequence_from_answer_content(content)

        # 只保留 compute_metrics 会用到的几列（再额外带上 ground_truth_seq 和 has_tag）
        rows.append(
            {
                "row_idx": i,
                "jsonl_index": rec_jsonl.get("index", i),
                "requirement_parquet": requirement_parquet,
                "requirement_from_jsonl": requirement_from_jsonl,
                "ground_truth_seq": gt_seq,
                "pred_seq": pred_seq,
                "has_tag": int(bool(has_tag)),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"[完成] 共写入 {len(out_df)} 条样本到 {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=str,
        default='/path/to/ProtoCycle/data/proteinllm/desc2seq_agent_eval_clever_100.parquet',
        help="原始 eval parquet 路径",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="eval 结果 jsonl 路径",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="输出的 CSV 路径",
    )

    args = parser.parse_args()
    main(args.parquet, args.jsonl, args.out_csv)
