#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parquet_to_eval_csv.py

功能：
- 把原始的 desc2seq parquet 转成和 extract_eval_results 输出兼容的 CSV：
  列为：
    row_idx
    jsonl_index
    requirement_parquet
    requirement_from_jsonl
    ground_truth_seq
    pred_seq

  其中：
    requirement_parquet = parquet.requirement
    requirement_from_jsonl = requirement_parquet  （占位，方便后面用同一套接口）
    ground_truth_seq = reward_model['ground_truth'] 解析出来
    pred_seq = ground_truth_seq  （baseline：预测=真值）

用法示例：
python parquet_to_eval_csv.py \
  --parquet /path/to/ProtoCycle/data/proteinllm/desc2seq_agent_eval_clever_100.parquet \
  --out_csv /path/to/ProtoCycle/baseline_results/gt_as_pred.csv
"""

import argparse
import ast
import json
from typing import Any, Dict, List

import pandas as pd


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


def main(parquet_path: str, out_csv: str) -> None:
    df = pd.read_parquet(parquet_path)
    n = len(df)
    print(f"[INFO] 读取 parquet：{parquet_path}，共 {n} 条样本。")

    rows: List[Dict[str, Any]] = []

    for i in range(n):
        row = df.iloc[i]

        requirement = str(row.get("requirement", ""))
        # jsonl_index：优先用 requirement_id，没有就用行号
        jsonl_index = row.get("requirement_id", i)

        gt_seq = extract_ground_truth_from_reward_model(row.get("reward_model"))
        pred_seq = gt_seq  # baseline: 预测 = ground truth

        rows.append(
            {
                "row_idx": i,
                "jsonl_index": int(jsonl_index),
                "requirement_parquet": requirement,
                "requirement_from_jsonl": requirement,  # 占位，保持列名一致
                "ground_truth_seq": gt_seq,
                "pred_seq": pred_seq,
                "has_tag": 1
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"[完成] 写入 {len(out_df)} 条到 {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="原始 desc2seq parquet 路径",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="输出 CSV 路径（列结构兼容 extract_eval_results 的输出）",
    )
    args = parser.parse_args()
    main(args.parquet, args.out_csv)
