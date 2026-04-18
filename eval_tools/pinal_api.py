#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
调用 Pinal API 对 desc2seq 的 instruction 做设计，并导出为 CSV。

输入:
    /path/to/ProDVa/desc2seq_agent_eval_clever_100.json
    其中每条记录包含:
        - "instruction": 文本描述
        - "sequence":    对应的 ground truth 序列

输出:
    /path/to/ProDVa/desc2seq_agent_eval_clever_100_pinal.csv

CSV 列:
    row_idx, jsonl_index, requirement_parquet, requirement_from_jsonl,
    ground_truth_seq, pred_seq, has_tag
"""

import json
import csv
import time
from typing import List, Dict, Tuple

from gradio_client import Client

PINAL_URL = "http://www.denovo-pinal.com/"
API_NAME = "/design_and_protrek_score"
INPUT_JSON = "/path/to/ProDVa/desc2seq_agent_eval_clever_100.json"
OUTPUT_CSV = "/path/to/ProtoCycle/baseline_results/pinal.csv"

INPUT_JSON = "/path/to/ProDVa/desc2seq_agent_eval_clever_CAMEO_100.json"
OUTPUT_CSV = "/path/to/ProtoCycle/baseline_results/pinal_CAMEO.csv"


def load_json_or_jsonl(path: str) -> List[Dict]:
    """兼容 JSON(list) 和 JSONL 两种格式."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"Unsupported JSON top-level type: {type(data)}")
    except json.JSONDecodeError:
        # 当成 JSON Lines
        records = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


def parse_best_seq_from_md(table_md: str) -> Tuple[str, float]:
    """
    从 Pinal 返回的 markdown 表格字符串中，解析出 Protrek Score 最高的那条序列。

    table_md 例子见你贴的输出:
        |    |   <div style="width:150px">Log(p) Per Token...</div> | ...
        |---:|-------------------:|----------------:|:---...
        |  0 | ... | 10.7696 | MAGGR...
        |  1 | ... | 10.7604 | MASGK...
        |  2 | ... | 15.5106 | MELLE...

    返回: (best_sequence, best_protrek_score)
    """
    best_seq = None
    best_score = None

    for line in table_md.splitlines():
        line = line.rstrip()
        if not line.startswith("|"):
            continue
        if line.startswith("|---"):
            # 分隔线
            continue
        # 跳过带 <div> header 的那一行
        if "Log(p)" in line or "Protrek Score" in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        # 期望格式: ['', idx, logp, protrek, seq, '']
        if len(parts) < 5:
            continue

        try:
            protrek = float(parts[3])
        except ValueError:
            continue

        seq = parts[4].strip()
        if not seq:
            continue

        if best_seq is None or protrek > best_score:
            best_seq = seq
            best_score = protrek

    if best_seq is None:
        raise RuntimeError("无法从 Pinal 的 markdown 表格中解析出任何序列。")

    return best_seq, best_score


def call_pinal(client: Client, text: str, designed_num: int = 3,
               max_retries: int = 3, sleep_sec: float = 1.0) -> str:
    """
    调 Pinal 的 gradio API，返回 Protrek Score 最高的序列字符串。
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            result = client.predict(
                input=text,
                designed_num=designed_num,
                api_name=API_NAME,
            )
            # result 通常是 (markdown_table_str, file_update_dict)
            if isinstance(result, (list, tuple)) and len(result) >= 1:
                table_md = result[0]
            else:
                # 兜底：如果直接就是字符串
                table_md = str(result)

            best_seq, best_score = parse_best_seq_from_md(table_md)
            return best_seq
        except Exception as e:
            last_err = e
            print(f"[WARN] Pinal 调用失败 (attempt {attempt}/{max_retries}): {e}")
            time.sleep(sleep_sec)

    raise RuntimeError(f"Pinal 调用多次失败，最后错误: {last_err}")


def main():
    records = load_json_or_jsonl(INPUT_JSON)
    print(f"Loaded {len(records)} records from {INPUT_JSON}")

    client = Client(PINAL_URL)

    # 写 CSV
    fieldnames = [
        "row_idx",
        "jsonl_index",
        "requirement_parquet",
        "requirement_from_jsonl",
        "ground_truth_seq",
        "pred_seq",
        "has_tag",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for i, rec in enumerate(records):
            instruction = rec.get("instruction", "")
            gt_seq = rec.get("sequence", "")

            if not instruction:
                print(f"[WARN] index {i} 没有 instruction，跳过")
                continue

            print(f"Processing index {i} ...")
            pred_seq = call_pinal(client, instruction, designed_num=5)
            print(pred_seq)

            row = {
                "row_idx": i,                    # 按输入顺序
                "jsonl_index": i,                # 同样用输入顺序
                "requirement_parquet": instruction,
                "requirement_from_jsonl": instruction,
                "ground_truth_seq": gt_seq,
                "pred_seq": pred_seq,
                "has_tag": 1,                    # 按你的要求写 1
            }
            writer.writerow(row)

    print(f"Done. CSV written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
