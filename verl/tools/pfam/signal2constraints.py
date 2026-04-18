#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal2constraints.py
从 constraints.json 中读取序列，自动选择一个合适的信号肽并写回:
  "signal_pep": {
      "sequence": "...",
      "fix_cleavage": true,
      "post_inpaint": false,
      "post_window_left": 6,
      "post_window_right": 2
  }

用法:
  python signal2constraints.py --json constraints.json
  python signal2constraints.py --json constraints.json --host bacteria
  python signal2constraints.py --json constraints.json --host mammal
  python signal2constraints.py --json constraints.json --prefer pelb
"""

import json, os, argparse, sys
from typing import Dict, Tuple, List

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
SMALL_AA = set(list("ASGTV"))
POS = set("KRH")
NEG = set("DE")

# 一个精简但常用的信号肽库（名字: (序列, 推荐宿主)）
SP_LIBRARY: Dict[str, Tuple[str, str]] = {
    # 细菌（E. coli）常用
    "pelb": ("MKYLLPTAAAGLLLLAAQPAMA", "bacteria"),   # PelB
    "ompa": ("MKKTAIAIAVALAGFATVAQA",  "bacteria"),   # OmpA
    "dsba": ("MKKIWLALAGLVLAFSASA",    "bacteria"),   # DsbA（偏强疏水，C区短）
    "phoa": ("MKQSTIALALLPLLFTPVTKA",  "bacteria"),   # PhoA

    # 哺乳动物分泌
    "igk":  ("METDTLLLWVLLLWVPGSTG",   "mammal"),     # Ig kappa leader
    "il2":  ("MYRMQLLSCIALSLALVTNS",   "mammal"),     # IL-2 signal peptide
    "tpa":  ("MDAMKRGLCCVLLLCGAVFVSPS","mammal"),     # tPA signal peptide

    # 酵母（通用真核也常可用）
    "mfalpha": ("MKFFSSGLVAGAAAATVAFVATS", "yeast"),  # α-factor prepro（只取其前导信号段的常见等价片段）
}

def valid_aa_seq(s: str) -> bool:
    return bool(s) and set(s).issubset(AA20)

def cleavage_ok(sp: str) -> bool:
    """检查(-3,-1)是否是“小残基且非P”，符合常见 SignalP 统计偏好"""
    if len(sp) < 3: return False
    return (sp[-3] in SMALL_AA) and (sp[-1] in SMALL_AA) and (sp[-3] != 'P') and (sp[-1] != 'P')

def c_region_penalty(sp: str, c_len: int = 6) -> int:
    """在 C 区末端惩罚带电和 P，越少越好"""
    if not sp: return 999
    start = max(0, len(sp) - c_len)
    c = sp[start:]
    return sum(1 for a in c if (a in POS or a in NEG or a == 'P'))

def downstream_compat_penalty(first_target_aa: str) -> int:
    """
    对下游第一个氨基酸做一点点“兼容性”惩罚：
      - 若为酸性( D/E )，给一点惩罚（传统经验：+1酸性不利于切割）。
    """
    if first_target_aa in NEG:
        return 1
    return 0

def score_signal_peptide(name: str, sp: str, target_seq: str, host_pref: str) -> Tuple[int, Dict]:
    """
    返回一个越小越好的综合分（多指标相加），附带诊断信息
    """
    diag = {}
    score = 0

    # 1) 宿主偏好（不匹配加小罚分）
    lib_host = SP_LIBRARY[name][1]
    if host_pref and lib_host != host_pref:
        score += 1
        diag["host_mismatch"] = True
    else:
        diag["host_mismatch"] = False

    # 2) (-3, -1) 小残基与非P
    if not cleavage_ok(sp):
        score += 2
        diag["cleavage_pref_bad"] = True
    else:
        diag["cleavage_pref_bad"] = False

    # 3) C 区末端“坏氨基酸”个数（带电/P）
    c_pen = c_region_penalty(sp, c_len=6)
    score += c_pen
    diag["c_region_penalty"] = c_pen

    # 4) 下游首位兼容性（酸性罚分）
    first_aa = target_seq[0] if target_seq else "A"
    d_pen = downstream_compat_penalty(first_aa)
    score += d_pen
    diag["downstream_penalty"] = d_pen
    diag["first_aa"] = first_aa

    # 5) 轻度长度约束（过短/过长加一点点罚分）
    if len(sp) < 15: score += 1
    if len(sp) > 30: score += 1

    return score, diag

def choose_sp(seq: str, host: str, prefer: str = None) -> Tuple[str, str, Dict]:
    """
    从库中选择一个评分最优的 SP。优先返回 prefer（若提供且可用），否则综合打分选最优。
    返回: (name, sequence, diagnostics)
    """
    if not valid_aa_seq(seq):
        raise ValueError("Input 'sequence' must be 20-AA 字母组成且非空。")

    host = (host or "bacteria").lower()
    if prefer and prefer.lower() in SP_LIBRARY:
        name = prefer.lower()
        sp, _ = SP_LIBRARY[name]
        sc, diag = score_signal_peptide(name, sp, seq, host)
        diag["chosen_by"] = "prefer"
        return name, sp, diag

    # 评分表
    candidates = []
    for name, (sp, sp_host) in SP_LIBRARY.items():
        sc, diag = score_signal_peptide(name, sp, seq, host)
        candidates.append((sc, name, sp, diag))

    # 先按得分排序；同分按宿主匹配、再按惯用度排序（手工优先级：pelb > ompa > dsba > phoa > igk > il2 > tpa > mfalpha）
    prefer_order = {"pelb":0, "ompa":1, "dsba":2, "phoa":3, "igk":4, "il2":5, "tpa":6, "mfalpha":7}
    candidates.sort(key=lambda x: (x[0], SP_LIBRARY[x[1]][1] != host, prefer_order.get(x[1], 999)))
    best_sc, best_name, best_sp, best_diag = candidates[0]
    best_diag["chosen_by"] = "score"
    best_diag["score"] = best_sc
    return best_name, best_sp, best_diag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="path to constraints.json")
    ap.add_argument("--host", default="bacteria", choices=["bacteria","mammal","yeast"],
                    help="宿主偏好（仅作为选择时的软偏置）")
    ap.add_argument("--prefer", default=None, help="优先使用某个库内名称（如: pelb, ompa, dsba, igk, ...）")
    args = ap.parse_args()

    if not os.path.exists(args.json):
        print(f"ERROR: file not found: {args.json}", file=sys.stderr)
        sys.exit(1)

    with open(args.json, "r", encoding="utf-8") as f:
        try:
            cfg = json.load(f)
        except Exception as e:
            print(f"ERROR: invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    seq = cfg.get("sequence", "")
    if not valid_aa_seq(seq):
        print("ERROR: JSON['sequence'] 必须为20种氨基酸字母且非空。", file=sys.stderr)
        sys.exit(1)

    name, sp, diag = choose_sp(seq, host=args.host, prefer=args.prefer)

    # 写回 signal_pep 字段（不动其他内容）
    cfg["signal_pep"] = {
        "sequence": sp if sp.startswith("M") else ("M"+sp),
        "fix_cleavage": True,
        "post_inpaint": False,
        "post_window_left": 6,
        "post_window_right": 2
    }

    # 原地覆盖原文件
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # 控制台友好提示
    print(f"[OK] Added signal_pep using '{name}': {cfg['signal_pep']['sequence']}")
    print("[Diag]", json.dumps(diag, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
