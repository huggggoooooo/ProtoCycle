#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_constraints_from_uniprot.py

用法：
  python build_constraints_from_uniprot.py \
      --accession A0A010S514 \
      --fetch-script ./fetch_uniprot_features.py \
      --out out

流程：
  1) 调用：python fetch_uniprot_features.py --accession <ACC> --out <OUT>
  2) 读取：<OUT>/<ACC>.json
  3) 生成：<OUT>/<ACC>_constraints.json
     只写：
       - sequence
       - locked（来自 binding_sites / active_sites / motifs / domains / regions 的区间逐位锁定）
       - decode（默认块）
"""

import os
import json
import argparse
import subprocess
import sys
from typing import Dict, Any, List, Optional

def run_fetch(fetch_script: str, accession: str, out_dir: str):
    cmd = [sys.executable, fetch_script, "--accession", accession, "--out", out_dir]
    print(f"[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"fetch_uniprot_features.py failed with code {res.returncode}")
    if res.stdout.strip():
        print(f"[fetch] stdout (tail): {res.stdout.strip().splitlines()[-1][:200]}")
    if res.stderr.strip():
        print(f"[fetch] stderr (tail): {res.stderr.strip().splitlines()[-1][:200]}", file=sys.stderr)

def load_uniprot_json(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"UniProt feature JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("accession", "sequence", "length"):
        if key not in data:
            raise ValueError(f"Missing '{key}' in UniProt JSON: {json_path}")
    seq = data["sequence"]
    if not isinstance(seq, str) or not seq:
        raise ValueError("Invalid 'sequence' in UniProt JSON.")
    return data

def _lock_range(seq: str, locked: Dict[str, str], start: int, end: int):
    """将 1-based 区间 [start,end] 内每个位置锁定为原序列氨基酸。"""
    L = len(seq)
    if start <= 0 or end <= 0:
        return
    if start > end:
        start, end = end, start
    start = max(1, min(L, start))
    end   = max(1, min(L, end))
    for pos in range(start, end + 1):
        locked[str(pos)] = seq[pos - 1]

def _lock_from_feature_list(seq: str, locked: Dict[str, str], items: Any, label: str) -> int:
    """通用：从一个列表型特征（元素含 start/end）逐位锁定。"""
    if not items:
        return 0
    count = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            s = int(it.get("start", 0))
            e = int(it.get("end", 0))
        except Exception:
            continue
        if not (s and e):
            continue
        _lock_range(seq, locked, s, e)
        count += (abs(e - s) + 1)
    if count > 0:
        print(f"[lock] {label:14s}: {count:5d} residues")
    return count

def gather_feature_lists(data: dict) -> Dict[str, List[dict]]:
    """
    同时支持两种放置方式：
      1) 顶层：active_sites/motifs/domains/regions
      2) key_sites.*：binding_sites/active_sites/motifs/domains/regions
    返回统一的 dict，其中每个键都是一个列表（可能为空）。
    """
    # 顶层
    top_active  = data.get("active_sites") or []
    top_motifs  = data.get("motifs") or []
    top_domains = data.get("domains") or []
    top_regions = data.get("regions") or []

    # key_sites.*
    ks = data.get("key_sites") or {}
    ks_binding = ks.get("binding_sites") or []
    ks_active  = ks.get("active_sites") or []
    ks_motifs  = ks.get("motifs") or []
    ks_domains = ks.get("domains") or []
    ks_regions = ks.get("regions") or []

    # 合并（binding_sites 目前只在 key_sites 中常见，但也留接口）
    binding_sites = ks_binding  # 若未来顶层也有，可改为：(data.get("binding_sites") or []) + ks_binding

    return {
        "binding_sites": binding_sites,
        "active_sites": top_active + ks_active,
        "motifs":       top_motifs + ks_motifs,
        "domains":      top_domains + ks_domains,
        "regions":      top_regions + ks_regions,
    }

def build_locked_from_features(seq: str, data: dict) -> Dict[str, str]:
    """
    汇总锁定来源：
      - binding_sites（通常在 key_sites 内）
      - active_sites / motifs / domains / regions（顶层或 key_sites 都支持）
    """
    locked: Dict[str, str] = {}
    feat_lists = gather_feature_lists(data)

    _lock_from_feature_list(seq, locked, feat_lists["binding_sites"], "binding_sites")
    _lock_from_feature_list(seq, locked, feat_lists["active_sites"],  "active_sites")
    _lock_from_feature_list(seq, locked, feat_lists["motifs"],        "motifs")
    _lock_from_feature_list(seq, locked, feat_lists["domains"],       "domains")
    _lock_from_feature_list(seq, locked, feat_lists["regions"],       "regions")

    return locked

def main():
    ap = argparse.ArgumentParser(description="Build constraints.json from UniProt accession via fetch_uniprot_features.py")
    ap.add_argument("--accession", required=True, help="UniProt accession, e.g., A0A010S514")
    ap.add_argument(
        "--fetch-script",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_uniprot_features.py"),
        help="Path to fetch_uniprot_features.py",
    )
    ap.add_argument("--out", default='./out', help="Output directory (will contain <ACC>.json and <ACC>_constraints.json)")
    args = ap.parse_args()

    acc = args.accession.strip()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # 1) 调用抓取脚本
    run_fetch(args.fetch_script, acc, out_dir)

    # 2) 读取抓取结果
    uni_json_path = os.path.join(out_dir, f"{acc}.json")
    data = load_uniprot_json(uni_json_path)
    seq = data["sequence"]

    # 3) 构造 locked（binding_sites + active_sites + motifs + domains + regions）
    locked = build_locked_from_features(seq, data)

    # 4) 输出最小 constraints：sequence / locked / decode 默认
    constraints = {
        "sequence": seq,
        "locked": locked,
        "decode": {
            "temperature": 0.9,
            "top_k": 12,
            "num_candidates": 24,
            "max_retries": 64
        }
    }

    out_json_path = os.path.join(out_dir, f"{acc}_constraints.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(constraints, f, ensure_ascii=False, indent=2)

    print(f"[OK] constraints written: {out_json_path}")
    print(f"[SUMMARY] sequence_len={len(seq)}  locked_sites={len(locked)}")

if __name__ == "__main__":
    main()
