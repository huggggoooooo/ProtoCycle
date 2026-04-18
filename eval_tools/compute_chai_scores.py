#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对 CSV 里的每条 seq 用本地 Chai-1 打分，并把 (ptm, plddt, pae) 三列追加回 CSV。

用法示例：
    python run_chai1_on_csv.py \
        --input_csv  input.csv \
        --output_csv output_with_chai.csv \
        --output_root /tmp/chai1_tmp \
        --device cuda:0

说明：
- 需要事先准备好 run_chai1_for_sequence(seq, name, output_root, device) 函数
  （可以直接把你那段函数 copy 到本文件前面，或者从你自己的模块里 import）。
- 如果不传 --output_csv，则会“就地覆盖”，直接写回 input_csv。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

# -----------------------------------------
# 如果 run_chai1_for_sequence 已经在别的文件里，就改成 from xxx import ...
# 这里假设你已经把你给的那个函数粘贴在本文件上方或同一文件中。
# -----------------------------------------
from compute_metrics import run_chai1_for_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="输入的 CSV 路径，要求至少包含一列 `seq`",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="输出 CSV 路径；默认 None 表示覆盖写回 input_csv",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Chai-1 的临时输出根目录（每条序列会建一个子目录，然后自动删掉）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="运行 Chai-1 的 device，比如 cuda:0 / cuda:1 / cpu，默认 cuda:0",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv) if args.output_csv is not None else input_csv
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 设备
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        print("[WARN] CUDA 不可用，将在 CPU 上运行 Chai-1", file=sys.stderr)
        device = torch.device("cpu")

    # 读表
    df = pd.read_csv(input_csv)
    if "seq" not in df.columns:
        raise ValueError(f"CSV 里找不到 'seq' 这一列，实际列为：{list(df.columns)}")

    n = len(df)
    print(f"[INFO] 读取 {input_csv}, 共 {n} 行")

    ptm_list = []
    plddt_list = []
    pae_list = []

    for idx, row in df.iterrows():
        raw_seq = row["seq"]

        # 为空 / NaN 的直接跳过，写 NaN
        if pd.isna(raw_seq) or str(raw_seq).strip() == "":
            print(f"[SKIP] idx={idx}: 空序列，写 NaN")
            ptm_list.append(float("nan"))
            plddt_list.append(float("nan"))
            pae_list.append(float("nan"))
            continue

        seq = str(raw_seq)
        # 如果有 case 这列，就用它做名字；否则用 seq_<idx>
        name = str(row["case"]) if "case" in df.columns else f"seq_{idx}"

        print(f"[RUN] idx={idx}, name={name}, len(seq)={len(seq)}")
        try:
            ptm_mean, plddt_mean, pae_mean = run_chai1_for_sequence(
                seq=seq,
                name=name,
                output_root=output_root,
                device=device,
            )
        except Exception as e:
            print(
                f"[ERROR] idx={idx}, name={name} 跑 Chai-1 失败：{e}",
                file=sys.stderr,
            )
            ptm_mean, plddt_mean, pae_mean = float("nan"), float("nan"), float("nan")

        ptm_list.append(ptm_mean)
        plddt_list.append(plddt_mean)
        pae_list.append(pae_mean)

    # 追加三列
    df["chai_ptm"] = ptm_list
    df["chai_plddt"] = plddt_list
    df["chai_pae"] = pae_list

    # 写回
    df.to_csv(output_csv, index=False)
    print(f"[OK] 已写入：{output_csv}")
    print("    新增列：chai_ptm, chai_plddt, chai_pae")


if __name__ == "__main__":
    main()
