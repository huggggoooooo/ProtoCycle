#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
summarize_metrics.py

功能：
- 输入一个 metrics.csv（单次评测结果）

两种模式：

1）默认模式（不加 --strict_valid_only）：
    - 对每个数值指标列做如下处理：
        * 先在「valid 样本」（has_tag==1 且 is_valid==1）上找该列的最差值：
            - 对于越小越好的指标（esm_ppl, repeat_pct, pae_mean）：最差 = valid 中的最大值
            - 对于其它数值指标：最差 = valid 中的最小值
        * 对所有 **invalid 样本**（以及该列为 NaN 的行），用“最差值”填充
          - invalid 的定义：
              - has_tag==1 且 is_valid!=1 的样本：永远视为 invalid，会被惩罚（填 worst）
              - has_tag==0 的样本：
                  · 默认：不参与均值计算（被从平均里跳过）
                  · 若加 --penalize_no_tag：也视为 invalid，用“最差值”填充后参与平均
        * 最后在「参与平均的样本」上求平均：
            - 默认：只在 has_tag==1 的样本上平均（但其中 is_valid!=1 的会被填 worst）
            - 加 --penalize_no_tag：在全部样本上平均（无 tag 的也会被填 worst）
    - 额外统计（在全部样本上）：
        * has_tag_ratio  = has_tag==1 的比例
        * is_valid_ratio = is_valid==1 的比例

2）严格模式（加 --strict_valid_only）：
    - 若存在 ground_truth_seq 列，则先过滤掉其中包含 'X' 的样本
    - 只保留 has_tag==1 且 is_valid==1 的样本
    - 对这些样本的数值指标列直接做简单均值（不再填最差值）
    - 输出的 has_tag_ratio / is_valid_ratio 都为 1.0（因为只剩真正 valid）

最后：
- 把结果写入 / 追加到一个 metrics_all.csv 中：
    * 第一列：source_file （输入 metrics.csv 的文件名，不含路径）
    * 后续列：各个指标列的整体均值 + has_tag_ratio / is_valid_ratio
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def compute_summary(
    csv_path: Path,
    exclude_cols: List[str] | None = None,
    strict_valid_only: bool = False,
    penalize_no_tag: bool = True,
) -> Dict[str, Any]:
    """
    读取一个 metrics.csv，返回 {列名: 统计值}。

    参数：
    - strict_valid_only:
        True  → 严格模式，只看 has_tag==1 & is_valid==1
        False → 默认模式，见文件头说明
    - penalize_no_tag:
        仅在 strict_valid_only=False 时生效：
        True  → has_tag==0 也按 invalid 处理（填最差值参与平均）——旧行为
        False → has_tag==0 的样本完全跳过，不参与平均
    """
    if exclude_cols is None:
        exclude_cols = []

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"文件 {csv_path} 为空。")

    # 哪些指标是“越小越好”
    smaller_better = {"esm_ppl", "repeat_pct", "pae_mean"}

    # =========== 严格模式：只看真正 valid 的样本 ===========
    if strict_valid_only:
        # 1) 若存在 ground_truth_seq，先去掉含 X 的样本
        if "ground_truth_seq" in df.columns:
            before = len(df)
            mask_no_x = ~df["ground_truth_seq"].astype(str).str.contains("X")
            df = df[mask_no_x].reset_index(drop=True)
            after = len(df)
            if after == 0:
                raise ValueError(
                    f"文件 {csv_path} 中，过滤掉 ground_truth_seq 含 'X' 的样本后已无可用样本。"
                )
            print(
                f"[INFO] {csv_path.name}: 严格模式下过滤 ground_truth_seq 含 'X' 的样本 "
                f"{before - after}/{before} 条，剩余 {after} 条。"
            )

        # 2) 只保留 has_tag == 1 且 is_valid == 1 的样本（如果这两列存在）
        if "has_tag" in df.columns:
            df = df[df["has_tag"] == 1]
        if "is_valid" in df.columns:
            df = df[df["is_valid"] == 1]

        if len(df) == 0:
            raise ValueError(
                f"文件 {csv_path} 在严格模式下（过滤 X + 只保留 has_tag==1 & is_valid==1）"
                f"已无可用样本。"
            )

        # has_tag_ratio / is_valid_ratio 在严格模式下固定为 1.0（如果列存在）
        has_tag_ratio = 1.0 if "has_tag" in df.columns else None
        is_valid_ratio = 1.0 if "is_valid" in df.columns else None

        # 数值指标列：排除 idx / has_tag / is_valid
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        to_exclude = set(exclude_cols) | {"idx", "has_tag", "is_valid"}
        metric_cols = [c for c in numeric_cols if c not in to_exclude]

        if not metric_cols:
            raise ValueError(
                f"文件 {csv_path} 中没有可用于求平均的数值指标列。"
            )

        means: Dict[str, Any] = {}
        for col in metric_cols:
            means[col] = float(df[col].mean())

        if has_tag_ratio is not None:
            means["has_tag_ratio"] = has_tag_ratio
        if is_valid_ratio is not None:
            means["is_valid_ratio"] = is_valid_ratio

        return means

    # =========== 默认模式：惩罚 invalid 的样本 ===========
    n_total = len(df)

    # 比例：在所有样本上统计（0/1）
    has_tag_ratio = None
    is_valid_ratio = None
    if "has_tag" in df.columns:
        has_tag_ratio = float(df["has_tag"].mean())
    if "is_valid" in df.columns:
        is_valid_ratio = float(df["is_valid"].mean())

    # valid 定义：has_tag==1 & is_valid==1（如果两列都存在）
    #   - 用于：从 valid 中提取“最差值”（worst）
    # include_mask：
    #   - 决定哪些样本参与均值：
    #       · penalize_no_tag=True  → 所有样本都参与（无 tag 也参与）
    #       · penalize_no_tag=False → 仅 has_tag==1 的样本参与（无 tag 被跳过）
    if ("has_tag" in df.columns) and ("is_valid" in df.columns):
        has_tag_col = df["has_tag"]
        is_valid_col = df["is_valid"]

        base_valid_mask = (has_tag_col == 1) & (is_valid_col == 1)

        if penalize_no_tag:
            # 旧行为：所有样本参与平均
            include_mask = pd.Series(True, index=df.index)
        else:
            # 新行为：无 tag 的样本完全不参与平均
            include_mask = (has_tag_col == 1)

        # 需要“惩罚”的样本 = 会参与平均，但不是 valid 的
        invalid_mask_for_penalty = include_mask & (~base_valid_mask)
    else:
        # 没有 has_tag / is_valid 的时候，退化为原来逻辑：所有都参加 & 都算 valid
        base_valid_mask = pd.Series(True, index=df.index)
        include_mask = pd.Series(True, index=df.index)
        invalid_mask_for_penalty = pd.Series(False, index=df.index)

    df_valid = df[base_valid_mask]

    # 数值指标列
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    to_exclude = set(exclude_cols) | {"idx", "has_tag", "is_valid"}
    metric_cols = [c for c in numeric_cols if c not in to_exclude]

    if not metric_cols:
        raise ValueError(f"文件 {csv_path} 中没有可用于求平均的数值指标列。")

    means: Dict[str, Any] = {}

    for col in metric_cols:
        # 1) 在 valid 样本中取该列的值
        col_valid = df_valid[col].dropna()

        # 如果 valid 中没有非 NaN 值，就在全体里找
        if col_valid.empty:
            col_all_non_nan = df[col].dropna()
            if col_all_non_nan.empty:
                # 整列都是 NaN
                means[col] = float("nan")
                continue
            col_valid = col_all_non_nan

        # 2) 决定“最差值”
        if col in smaller_better:
            worst = float(col_valid.max())  # 越小越好 → 最大值最差
        else:
            worst = float(col_valid.min())  # 越大越好 → 最小值最差

        # 3) 构造一个新列：
        #    - 对于「会参与平均」的样本：
        #        · invalid 或 NaN → worst
        #    - 不参与平均的样本（include_mask==False）随便，反正后面会被丢掉
        col_all = df[col].copy()

        invalid_or_nan = (invalid_mask_for_penalty | col_all.isna()) & include_mask
        col_all[invalid_or_nan] = worst

        # 4) 在“参与平均”的样本上取平均
        if include_mask.any():
            means[col] = float(col_all[include_mask].mean())
        else:
            # 极端情况：没有任何样本参与平均（例如所有 has_tag==0 且没开 penalize_no_tag）
            # 退化为原来的“全体平均”
            means[col] = float(col_all.mean())

    # 把比例也加进去（仍在所有样本上统计）
    if has_tag_ratio is not None:
        means["has_tag_ratio"] = has_tag_ratio
    if is_valid_ratio is not None:
        means["is_valid_ratio"] = is_valid_ratio

    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_csv",
        type=str,
        required=True,
        help="单次评测的 metrics.csv 路径",
    )
    parser.add_argument(
        "--metrics_all",
        type=str,
        default="/path/to/ProtoCycle/baseline_results/metrics/metrics_all.csv",
        help=(
            "汇总文件 metrics_all.csv 路径；"
            "若不指定，默认写在 metrics_csv 所在目录下的 metrics_all.csv"
        ),
    )
    parser.add_argument(
        "--strict_valid_only",
        action="store_true",
        help=(
            "严格模式：过滤掉 ground_truth_seq 含 'X' 的样本，"
            "只保留 has_tag==1 且 is_valid==1 的样本做简单均值；"
            "输出的 has_tag_ratio / is_valid_ratio 将为 1.0。"
        ),
    )
    parser.add_argument(
        "--penalize_no_tag",
        action="store_true",
        help=(
            "是否惩罚 has_tag==0 的样本：\n"
            "  - 默认（不加本参数）：has_tag==0 的样本在均值计算中被忽略（不参与平均，分母只算有 tag 的样本）；\n"
            "  - 加上本参数：has_tag==0 的样本也视为 invalid，用该列最差值填充后参与平均（旧行为）。"
        ),
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv).resolve()
    if not metrics_path.exists():
        raise FileNotFoundError(f"找不到 metrics_csv: {metrics_path}")

    if args.metrics_all is None:
        metrics_all_path = metrics_path.parent / "metrics_all.csv"
    else:
        metrics_all_path = Path(args.metrics_all).resolve()

    metrics_all_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 计算整体均值 + 比例（排除 idx 列）
    stats = compute_summary(
        metrics_path,
        exclude_cols=["idx"],
        strict_valid_only=args.strict_valid_only,
        penalize_no_tag=args.penalize_no_tag,
    )

    # 2. 组装一行：第一列是文件名
    row: Dict[str, Any] = {"source_file": metrics_path.name}
    row.update(stats)

    # 3. 写入 / 追加到 metrics_all.csv
    if metrics_all_path.exists():
        df_all = pd.read_csv(metrics_all_path)
        df_new = pd.DataFrame([row])
        df_all = pd.concat([df_all, df_new], ignore_index=True)
        df_all.to_csv(metrics_all_path, index=False)
        print(f"[INFO] 已在 {metrics_all_path} 末尾追加一行。")
    else:
        df_all = pd.DataFrame([row])
        df_all.to_csv(metrics_all_path, index=False)
        print(f"[INFO] 已创建新的 {metrics_all_path}。")

    print(f"[INFO] 当前加入的行对应文件: {metrics_path.name}")


if __name__ == "__main__":
    main()
