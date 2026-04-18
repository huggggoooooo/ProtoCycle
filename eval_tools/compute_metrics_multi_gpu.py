#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compute_metrics_multi_gpu.py

功能：
- 读取一个 merged_results.csv
- 按行切成若干 shard
- 每个 shard 在一张 GPU 上跑一份 compute_metrics.py
- 最后把各 shard 的 metrics_* 合并为一个总的 metrics.csv

用法示例：

# 自动检测卡数（比如 4 张），全部用上：
python compute_metrics_multi_gpu.py \
  --input_csv merged_results.csv \
  --output_csv metrics_all.csv

# 手动指定用第 0 和第 2 张卡，并给 compute_metrics.py 传一些额外参数：
python compute_metrics_multi_gpu.py \
  --input_csv merged_results.csv \
  --output_csv metrics_all.csv \
  --gpus 0,2 \
  --pass_through --skip_chai --skip_protrek --skip_retrieval
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import uuid  # 新增：用于生成唯一 tmp 目录名

try:
    import torch
except ImportError:
    torch = None


def parse_gpus(gpus_str: Optional[str], num_gpus: Optional[int]) -> List[int]:
    """解析 GPU 列表：优先用 --gpus（逗号分隔），否则用 --num_gpus，否则自动检测。"""
    if gpus_str:
        gpus = []
        for part in gpus_str.split(","):
            part = part.strip()
            if not part:
                continue
            gpus.append(int(part))
        if not gpus:
            raise ValueError(f"Parsed empty GPU list from --gpus={gpus_str}")
        return gpus

    if num_gpus is not None:
        if num_gpus <= 0:
            raise ValueError("--num_gpus must be positive")
        return list(range(num_gpus))

    # 自动检测
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch CUDA 不可用，且未指定 --gpus / --num_gpus，无法自动检测 GPU。"
        )
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("没有可用 GPU。")
    return list(range(n))


def make_shards(df: pd.DataFrame, num_shards: int) -> List[pd.DataFrame]:
    """把 df 按行平均切成 num_shards 份（最后一份稍微短一点没关系）。"""
    n = len(df)
    if num_shards > n:
        num_shards = n
    shards: List[pd.DataFrame] = []
    if num_shards <= 0:
        return shards

    shard_size = math.ceil(n / num_shards)
    for i in range(num_shards):
        start = i * shard_size
        end = min(n, (i + 1) * shard_size)
        if start >= end:
            break
        shards.append(df.iloc[start:end].copy())
    return shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="原始 merged_results.csv（extract_eval_results 的输出）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="最终合并后的 metrics.csv 路径",
    )
    parser.add_argument(
        "--compute_metrics_script",
        type=str,
        default=None,
        help="compute_metrics.py 的路径（默认是当前目录下的 compute_metrics.py）",
    )
    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="用于运行 compute_metrics.py 的 python 可执行文件（默认当前解释器）",
    )

    # GPU 相关
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="逗号分隔的 GPU 序号，比如 '0,1,3'；优先级高于 --num_gpus",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="使用前 num_gpus 张卡（从 0 开始编号），如果未指定 --gpus",
    )

    # 传给 compute_metrics.py 的其它参数：写在 --pass_through 后面，会原样附加到命令行
    parser.add_argument(
        "--pass_through",
        nargs=argparse.REMAINDER,
        help="额外传给 compute_metrics.py 的参数，写在 '--pass_through' 后面",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    out_dir = output_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compute_metrics_script is None:
        script_path = Path(__file__).resolve().parent / "compute_metrics.py"
    else:
        script_path = Path(args.compute_metrics_script).resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"找不到 compute_metrics.py: {script_path}")

    # 读入总的 CSV
    df = pd.read_csv(input_csv)
    n_rows = len(df)
    print(f"[multi-gpu] 总共有 {n_rows} 条样本。")

    # 解析 GPU 列表
    gpus = parse_gpus(args.gpus, args.num_gpus)
    if not gpus:
        raise RuntimeError("GPU 列表为空。")
    print(f"[multi-gpu] 使用 GPU: {gpus}")

    # 按 GPU 数量切 shard
    shards = make_shards(df, len(gpus))
    num_shards = len(shards)
    print(f"[multi-gpu] 实际生成 {num_shards} 个 shard。")

    if num_shards == 0:
        print("[multi-gpu] 没有数据可处理，直接退出。")
        return

    # 临时文件路径：每次 run 一个唯一目录，避免并发时冲突
    run_tag = f"multi_gpu_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = out_dir / run_tag
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[multi-gpu] 本次临时目录: {tmp_dir}")

    # 启动子进程
    procs = []
    shard_out_files = []

    extra_args = args.pass_through or []

    for shard_idx, (gpu, shard_df) in enumerate(zip(gpus, shards)):
        shard_in = tmp_dir / f"input_shard_{shard_idx}.csv"
        shard_out = tmp_dir / f"metrics_shard_{shard_idx}.csv"
        shard_df.to_csv(shard_in, index=False)
        shard_out_files.append(shard_out)

        # 每个子进程里，我们通过 CUDA_VISIBLE_DEVICES 把这张卡映射为 0，
        # 然后 compute_metrics.py 里统一用 cuda:0 即可。
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            args.python_exe,
            str(script_path),
            "--input_csv",
            str(shard_in),
            "--output_csv",
            str(shard_out),
            # 统一在视角内用 cuda:0
            "--esm_device",
            "cuda:0",
            "--chai_device",
            "cuda:0",
        ]
        # 如果你的 compute_metrics 里有 --evollama_device、--retrieval_device 等，也可以在这里统一设成 cuda:0 / cuda
        # 不过由于我们限制了 CUDA_VISIBLE_DEVICES，这些 "cuda:0"/"cuda" 实际都会指向这张卡。

        cmd.extend(extra_args)

        print(f"[multi-gpu] 启动子进程 shard {shard_idx} 用 GPU {gpu}：")
        print("          ", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        procs.append(proc)

    # 等待所有子进程结束
    exit_codes = []
    for idx, proc in enumerate(procs):
        ret = proc.wait()
        exit_codes.append(ret)
        print(f"[multi-gpu] shard {idx} 进程退出码：{ret}")

    if any(code != 0 for code in exit_codes):
        print(
            "[multi-gpu] 警告：至少有一个 shard 的 compute_metrics.py 退出码非 0，"
            "请检查上面的日志。",
            file=sys.stderr,
        )

    # 合并 shard 输出
    merged_list = []
    for shard_out in shard_out_files:
        if not shard_out.exists():
            print(
                f"[multi-gpu] 警告：输出文件缺失 {shard_out}，跳过该 shard。",
                file=sys.stderr,
            )
            continue
        df_metrics = pd.read_csv(shard_out)
        merged_list.append(df_metrics)

    if not merged_list:
        print("[multi-gpu] 没有任何 shard 输出成功生成 metrics，退出。", file=sys.stderr)
        return

    merged = pd.concat(merged_list, ignore_index=True)

    # 按 idx 排序（compute_metrics 的输出里有 'idx' 列）
    if "idx" in merged.columns:
        merged.sort_values("idx", inplace=True)

    merged.to_csv(output_csv, index=False)
    print(f"[multi-gpu] 已将所有 shard 合并写入 {output_csv}")

    # 清理本次运行的临时目录
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[multi-gpu] 已删除临时目录 {tmp_dir}")


if __name__ == "__main__":
    main()
