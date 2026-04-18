#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cofactor2constraints.py  (offline-only, motif-list aware)

特性：
- 完全离线：不走 UniProt / 不走网络
- 输入:
    --in_json <constraints.json>  (可选，里面有 sequence / locked /已有约束)
    --sequence <seq or fasta-like file>（若 in_json 没有 sequence 时需要）
    --cofactor "Mg2+, acetyl-CoA; UDP-GlcNAc"
    --prosite_index <prosite_index.json>  本地索引
- 输出：
    1) 打印合并后的新 constraints.json (stdout)
    2) 如果提供了 --in_json，会覆盖写回

本脚本做的事情：
1. 从 base constraints 读取已知 locked 位点
2. 对每个 cofactor 描述：
    a. detect_cofactor_atom → 金属/辅基分类 (mg / zn / atp_gtp ...)
    b. choose_span_avoid_locked → 选窗口（尽量避开 locked）
    c. build_constraints_for → 基础手工规则 (regex/soft_preferences/…)
    d. build_constraints_from_prosite → (模式A) 在本序列上跑 PROSITE motif 命中并保护这些区域
    e. build_constraints_from_prosite_textbased → (模式B) 用描述文本在 PROSITE 里检索最相近的 motif（即使还没出现在序列中），把 motif pattern 追加到 motif 列表
    f. build_constraints_auto → KR/G 富集、疏水性、charge等轻量软约束
    g. merge_constraints 按字段合并（不会覆盖 motif；motif 是 list）

最终 motif 字段统一成：
"motif": [
  {
    "sequence": "CXXCH",
    "left_linker": "",
    "right_linker": "",
    "flank_k": 8,
    "alpha": 1.0,
    "beta": 0.25,
    "forbid_overlap_locked": true
  },
  ...
]

依赖文件：
- prosite_index.json: list[ { "RX": <python regex>, "DE": "...", "CC": "...", "ID": "...", "AC": "..."} ]
"""

import re, json, argparse, os, sys
from typing import Dict, List, Tuple, Optional, Set, Any
from copy import deepcopy

# -------------------- 常量与基础工具 --------------------

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
POS = set("KRH")   # 近似正电
NEG = set("DE")    # 近似负电
KD = { # Kyte-Doolittle hydrophobicity
 'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,
 'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5
}

def load_seq(arg: str) -> str:
    """
    允许用户传文件路径或直接传AA序列
    """
    if arg is None:
        return ""
    if os.path.exists(arg):
        with open(arg, "r") as f:
            s = f.read().strip()
    else:
        s = arg.strip()
    # 去掉非字母
    s = "".join([c for c in s if c.isalpha()]).upper()
    if not s or not set(s).issubset(AA20):
        raise ValueError("Input sequence must be 20AA letters only.")
    return s

def default_decode():
    return {"temperature": 0.9, "top_k": 12, "num_candidates": 24, "max_retries": 64}

def default_global_soft():
    return {
        "max_cys": 10,
        "charge_range": [-6, 6],
        "max_homopolymer": 5,
        "forbid_substrings": []
    }

def subseq_charge(s: str) -> int:
    return sum(1 for c in s if c in POS) - sum(1 for c in s if c in NEG)

def mean_kd(s: str) -> float:
    return sum(KD.get(a,0) for a in s)/max(1,len(s))

# -------------------- locked & 窗口选择 --------------------

def parse_locked_positions(d: Dict) -> Set[int]:
    locked = d.get("locked") or {}
    pos = set()
    for k in locked.keys():
        try:
            pos.add(int(k))
        except Exception:
            continue
    return pos

def window_score(seq: str, i0: int, win: int, desire: str) -> float:
    """
    给定窗口起点 i0、长度 win，在序列 seq 上打启发式分数。
    desire 是我们想 enrich 的AA类别，比如 "DE"(负电), "CH"(Cys/His), "FWY"(芳香), "G", "KR", ...
    分数越低越好。
    """
    s = seq[i0:i0+win]
    if len(s) < win:
        return 1e9

    want = set(list(desire)) if desire else set()
    frac = (sum(1 for c in s if c in want)/win) if want else 0.0

    # 惩罚太多P/G（太软）
    pg = (s.count("P") + s.count("G"))/win

    # 最长同聚 run
    best, cur = 1, 1
    for j in range(1,len(s)):
        if s[j] == s[j-1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    homog = best / win

    # charge 偏好
    ch = subseq_charge(s)
    charge_term = 0.0
    # Mg/Mn/Ca 口袋喜欢负电：不想太正
    if desire == "DE":
        charge_term = -min(0, ch)          # 惩罚正电
    # Zn/Heme 口袋常有C/H/FWY，不想太负
    elif desire in ("CH","FWY"):
        charge_term = max(0, -ch)          # 惩罚过负(很负)

    return (0.8 - frac) + 0.3*pg + 0.3*homog + 0.2*charge_term

def span_overlaps_locked(span: Tuple[int,int], locked: Set[int]) -> bool:
    i,j = span
    for p in range(i, j+1):
        if p in locked:
            return True
    return False

def choose_span_avoid_locked(seq: str,
                             win: int,
                             stride: int,
                             desire: str,
                             locked: Set[int]) -> Tuple[Optional[Tuple[int,int]], Optional[int]]:
    """
    目标：
      1) 先找一个完全不覆盖 locked 的窗口
      2) 如果实在找不到，允许最小程度 overlap
    返回 (span, overlap_locked_count)
    """
    L = len(seq)

    # 首先尝试完全不重叠的窗口
    best = (1e9, None)
    for i in range(0, max(1, L-win+1), max(1, stride)):
        span = (i+1, i+win)
        if span_overlaps_locked(span, locked):
            continue
        sc = window_score(seq, i, win, desire)
        if sc < best[0]:
            best = (sc, span)
    if best[1] is not None:
        return best[1], 0

    # 兜底：允许少量 overlap，选 overlap 最小+score最好
    best2 = (1e9, 1e9, None)  # (overlap_cnt, score, span)
    for i in range(0, max(1, L-win+1), max(1, stride)):
        span = (i+1, i+win)
        ov = sum(1 for p in range(span[0], span[1]+1) if p in locked)
        sc = window_score(seq, i, win, desire)
        key = (ov, sc, span)
        if key < best2:
            best2 = key
    if best2[2] is not None:
        return best2[2], int(best2[0])

    return None, None

# -------------------- cofactor 分类 --------------------

def _normtxt(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "").strip()).lower()

def detect_cofactor_atom(text: str) -> str:
    """
    把自然语言的 cofactor/ligand 描述，
    归一到少量关键词 (mg / zn / atp_gtp / heme_c / coa / ...)
    未知则返回 ""，表示自定义小分子等。
    """
    t = _normtxt(text)
    if re.search(r"\bmg(2\+)?\b|magnesium", t): return "mg"
    if re.search(r"\bmn(2\+)?\b|manganese", t): return "mn"
    if re.search(r"\bca(2\+)?\b|calcium",  t): return "ca"
    if re.search(r"\bzn(2\+)?\b|zinc",     t): return "zn"
    if re.search(r"\bni(2\+)?\b|nickel",   t): return "ni"
    if re.search(r"\bcu(2\+)?\b|copper",   t): return "cu"
    if re.search(r"\bfe[-\s]*s\b|\b2fe-2s\b|\b4fe-4s\b", t): return "fe_s"
    if re.search(r"\bheme[-\s_]*c\b", t): return "heme_c"
    if re.search(r"\bheme([- \_]b)?\b|\bhaem\b", t): return "heme_b"
    if re.search(r"\batp|gtp|ntp\b", t): return "atp_gtp"
    if re.search(r"\bnadp?\+?\b", t): return "nad"
    if re.search(r"\bflavin|fad|fmn\b", t): return "flavin"
    if re.search(r"\bplp|pyridoxal\b", t): return "plp"
    if re.search(r"\bsam|s-adenosylmethionine\b", t): return "sam"
    if re.search(r"\bbiotin\b", t): return "biotin"
    if re.search(r"\btpp\b|thiamine pyrophosphate", t): return "tpp"
    if re.search(r"\bcoenzyme\s*a\b|\bcoa\b", t): return "coa"
    return ""

# -------------------- exemplar stats / auto层 --------------------

def fetch_exemplar_stats(auto_key: str) -> Dict[str, Any]:
    """
    极简“家族统计”占位，用关键字给出口袋偏好（KR富集、Gly loop、疏水上限等）
    """
    k = (auto_key or "").lower()

    # 典型 NAD / flavin / nucleotide-binding loops 都富Gly
    if k in ("nadp","nad","flavin","fmn","fad"):
        return {
            "composition": {"G": 0.10, "KR": 0.14, "DE_max_run": 4},
            "kd_max_mean": 0.8,
            "motifs_weak": [r"[TASV][AG].{1,2}G.{1,2}G"],
            "ss": {"helix":0.35,"strand":0.30,"coil":0.35}
        }

    # ATP/GTP/P-loop 风格
    if "ntp" in k or "atp" in k or "gtp" in k:
        return {
            "composition": {"G": 0.11, "KR": 0.10, "DE_max_run": 4},
            "kd_max_mean": 0.9,
            "motifs_weak": [r"GxxxxGKS", r"[STAG].{1,2}G.{1,2}G"]
        }

    # 磷酸/负电配体
    if "phosphate" in k:
        return {
            "composition": {"KR": 0.16, "G": 0.08},
            "kd_max_mean": 1.0,
            "motifs_weak": [r"G.{1,4}G.{0,2}[KR]"]
        }

    return {}

def build_constraints_auto(auto_key: str,
                           seq: str,
                           span: Optional[Tuple[int,int]],
                           win: int) -> Dict:
    """
    自动弱约束层：
    - KR/G富集/charge窗口
    - 避免长DE/KR runs
    - 疏水性上限
    """
    target = fetch_exemplar_stats(auto_key)

    out = {
        "sequence": seq,
        "locked": {},
        "allowed_set": {},
        "banned_set": {},

        "regex_allow": [],
        "regex_forbid": [r"[KR]{6,}", r"[DE]{6,}", r"C.{0,3}C"],  # 一些保守兜底

        "global_soft": default_global_soft(),
        "decode": default_decode(),

        "motif": {
            "sequence":"",
            "left_linker":"",
            "right_linker":"",
            "flank_k":8,
            "alpha":1.0,
            "beta":0.25,
            "forbid_overlap_locked":True
        },

        "_meta": {"auto_key": auto_key, "auto_used": bool(target)}
    }

    comp = target.get("composition", {})
    kd_cap = target.get("kd_max_mean", None)

    # 给指定 span（窗口）加局部偏好
    if span:
        out["window_min_counts"] = []
        out["window_charge"] = []

        if comp.get("KR") is not None:
            out["window_min_counts"].append({
                "span":[span[0], span[1]],
                "allow_set":"KR",
                "min_count": max(1, (span[1]-span[0]+1)//14)
            })
            out["window_charge"].append({
                "span":[span[0], span[1]],
                "min_charge": +1,
                "max_charge": +8
            })

        if comp.get("G") is not None:
            out["window_min_counts"].append({
                "span":[span[0], span[1]],
                "allow_set":"G",
                "min_count": max(1, (span[1]-span[0]+1)//12)
            })

        if kd_cap is not None:
            out.setdefault("hydrophobicity_guard", [])
            out["hydrophobicity_guard"].append({
                "span":[span[0], span[1]],
                "max_mean_KD": kd_cap,
                "weight": 0.8
            })

    # 全局疏水性约束
    if kd_cap is not None:
        out.setdefault("hydrophobicity_guard", [])
        out["hydrophobicity_guard"].append({
            "span":[1,len(seq)],
            "max_mean_KD": kd_cap,
            "weight": 0.6
        })

    # 加弱 motif 正则（family-level弱模式）
    if target.get("motifs_weak"):
        out["regex_allow"] = _extend_unique_str_list(
            out["regex_allow"],
            target["motifs_weak"]
        )

    # 根据 DE_max_run 限制，避免太长酸性连串
    if comp.get("DE_max_run") is not None:
        out["regex_forbid"] = _extend_unique_str_list(
            out["regex_forbid"],
            [r"[DE]{%d,}" % int(comp["DE_max_run"])]
        )

    return out

# -------------------- 手写 cofactor 规则 --------------------

def build_constraints_for(cof: str,
                          seq: str,
                          span: Optional[Tuple[int,int]],
                          win: int) -> Dict:
    """
    针对常见金属 & 辅基，手写启发式规则：
    - Mg/Mn/Ca: 富DE、负电口袋
    - Zn/Ni/Cu: Cys/His配位
    - heme_c: CXXCH motif
    - ATP/GTP: Gly-rich P-loop
    - NAD/flavin: Gly-rich Rossmann-like
    - 等等
    """

    out = {
        "sequence": seq,
        "locked": {},
        "allowed_set": {},
        "banned_set": {},

        "regex_allow": [],
        "regex_forbid": [],

        "global_soft": default_global_soft(),
        "decode": default_decode(),

        "motif": {
            "sequence": "",
            "left_linker": "",
            "right_linker": "",
            "flank_k": 8,
            "alpha": 1.0,
            "beta": 0.25,
            "forbid_overlap_locked": True
        }
    }

    # Mg2+ / Mn2+：负电DE口袋
    if cof in ("mg", "mn"):
        out["regex_allow"].append(r"^(?!.*C.{0,3}C)(?!.*H.{0,2}H).*$")
        out["regex_forbid"] += [r"[KR]{5,}", r"H{5,}"]
        if span:
            out["soft_preferences"] = [
                {"span": [span[0], span[1]], "prefer_set": "DE",  "weight": 0.35},
                {"span": [span[0], span[1]], "prefer_set": "AST", "weight": 0.10}
            ]
            out["window_min_counts"] = [
                {"span": [span[0], span[1]], "allow_set": "DE", "min_count": 3}
            ]
            out["window_charge"] = [
                {"span": [span[0], span[1]], "max_charge": -2}
            ]
        return out

    # Ca2+：酸性+柔软G/S/T/A
    if cof == "ca":
        out["regex_allow"].append(r"^(?!.*C.{0,3}C)(?!.*H.{0,2}H).*$")
        if span:
            out["soft_preferences"] = [
                {"span": [span[0], span[1]], "prefer_set": "GSTA", "weight": 0.2}
            ]
            out["window_min_counts"] = [
                {"span": [span[0], span[1]], "allow_set": "DE", "min_count": 2}
            ]
        return out

    # Zn / Ni / Cu：Cys/His 配位
    if cof in ("zn", "ni", "cu"):
        out["regex_allow"] += [r"C.{2,4}C", r"H.{2,5}H", r"C.{2,4}H", r"H.{2,4}C"]
        out["regex_forbid"].append(r"[DE]{4,}")
        if span:
            out["soft_preferences"] = [
                {"span": [span[0], span[1]], "prefer_set": "CH", "weight": 0.35}
            ]
        return out

    # Fe-S 簇
    if cof == "fe_s":
        out["regex_allow"] += [r"C.{2}C", r"C.{3}C", r"C.{2,4}C.{2,4}C"]
        out["global_soft"]["max_cys"] = 12
        return out

    # Heme c (CXXCH)
    if cof == "heme_c":
        out["motif"] = {
            "sequence": "CXXCH",
            "left_linker": "",
            "right_linker": "",
            "flank_k": 8,
            "alpha": 1.0,
            "beta": 0.25,
            "forbid_overlap_locked": True
        }
        out["soft_preferences"] = [
            {"span": [1, len(seq)], "prefer_set": "FWY", "weight": 0.2}
        ]
        return out

    # Heme b：His/Cys配位，芳香环境
    if cof == "heme_b":
        out["regex_allow"] += ["H", "C"]
        out["soft_preferences"] = [
            {"span": [1, len(seq)], "prefer_set": "FWY", "weight": 0.2}
        ]
        return out

    # ATP/GTP：P-loop/Walker A (GxxxxGKS)
    if cof == "atp_gtp":
        out["regex_forbid"].append(r"[DE]{4,}")
        if span:
            out["soft_preferences"] = [
                {"span": [span[0], span[1]], "prefer_set": "G",   "weight": 0.25},
                {"span": [span[0], span[1]], "prefer_set": "KST", "weight": 0.20}
            ]
        return out

    # NAD / Flavin：Rossmann-like Gly-rich loop
    if cof in ("nad", "flavin"):
        out["regex_allow"].append(r"G.{1,2}GG.{1,2}G")
        if span:
            out["soft_preferences"] = [
                {"span": [span[0], span[1]], "prefer_set": "GAS", "weight": 0.2}
            ]
        return out

    # PLP (pyridoxal phosphate)：赖氨酸Schiff base
    if cof == "plp":
        out["regex_allow"].append(r"K")
        out["regex_forbid"].append(r"[DE]{3,}")
        if span:
            out["soft_preferences"] = [
                {"span": [span[0], span[1]], "prefer_set": "ASTG", "weight": 0.15}
            ]
        return out

    # SAM, Biotin, TPP, CoA：辅酶/辅基口袋，通常富 Gly/Ala/Ser + 芳香堆叠
    if cof == "sam":
        if span:
            out["soft_preferences"] = [
                {"span": [span[0], span[1]], "prefer_set": "FWY", "weight": 0.15},
                {"span": [span[0], span[1]], "prefer_set": "GAS", "weight": 0.10}
            ]
        return out

    if cof in ("biotin", "tpp", "coa"):
        if span:
            out["soft_preferences"] = [
                {"span": [span[0], span[1]], "prefer_set": "GAS", "weight": 0.1}
            ]
        return out

    # 不识别的 cofactor，给空壳
    return out

# -------------------- PROSITE 支持 --------------------

def load_prosite_index(json_path: str) -> list:
    if not json_path or not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _simple_text_score(query: str, text: str) -> float:
    """
    非 fancy 的相似度打分：子串命中 + token overlap
    """
    q = query.lower()
    t = text.lower()
    score = 0.0

    if q in t:
        score += 5.0

    q_tokens = re.split(r"[^a-z0-9]+", q)
    t_tokens = re.split(r"[^a-z0-9]+", t)
    t_set = set([tok for tok in t_tokens if tok])
    for tok in q_tokens:
        if tok and tok in t_set:
            score += 1.0

    return score

def _regex_to_consensus(seq_regex: str, max_len: int = 12) -> str:
    """
    把 PROSITE/Python 正则 转成一个近似“代表性序列”：
    - [KRH] -> K (取第一个)
    - '.' / 'x' -> 'X'
    - A{3} -> AAA, .{3} -> XXX
    - 去掉 () | ? + ^ $ 等分支符，保留线性主干
    只是一个 inpaint 引导用的 motif 占位符，不是严格共识logo
    """

    rx = seq_regex

    # 展开 {n} 重复：A{3} -> AAA, .{3} -> XXX, [KR]{2} -> K K
    def _expand_repeat(m):
        unit = m.group(1)
        n = int(m.group(2))
        if unit == ".":
            unit_rep = "X"
        elif len(unit) == 1 and unit.isalpha():
            unit_rep = unit
        else:
            # "[KR]" 等
            if unit.startswith("[") and unit.endswith("]") and len(unit) > 2:
                unit_rep = unit[1]  # first char inside []
            else:
                unit_rep = "X"
        return unit_rep * min(n, max_len)

    rx = re.sub(r"(\[[^\]]+\]|\w|\.)\{(\d+)\}", _expand_repeat, rx)

    # 字符类别 [KRH] -> 'K' (第一个)
    rx = re.sub(r"\[([A-Z]+)\]", lambda m: m.group(1)[0], rx)

    # '.' / 'x' -> 'X'
    rx = rx.replace(".", "X")
    rx = rx.replace("x", "X")

    # 去括号、分支等
    rx = re.sub(r"[\(\)\|\?\+\^\$\{\}]", "", rx)

    rx = rx.strip()
    if len(rx) > max_len:
        rx = rx[:max_len]

    if not rx:
        rx = "XXX"
    return rx

def build_constraints_from_prosite_textbased(
    seq: str,
    prosite_json: str,
    hint_text: str,
    win: int = 16
) -> Dict:
    """
    模式B：基于描述文本 hint_text，从 PROSITE 索引里找最相关 motif，
    即使该 motif 目前不在 seq 里出现，也把它加入到设计约束中。

    返回的约束：
      - regex_allow: 把这些 motif 的正则都允许
      - motif: 用 top motif 的共识序列当作一个 motif dict
      - 这里不生成 window_min_counts/window_charge，因为我们没确定具体坐标
    """

    idx = load_prosite_index(prosite_json)
    if not idx:
        return {}

    query = hint_text.strip()
    scored = []

    for rec in idx:
        rx = (rec.get("RX") or "").strip()
        if not rx:
            continue

        # 拼一个文本 blob 用于匹配
        blob_parts = []
        for key in ("ID","AC","DE","CC"):
            if key in rec and rec[key]:
                blob_parts.append(str(rec[key]))
        blob = " ".join(blob_parts)

        sc = _simple_text_score(query, blob)
        if sc > 0:
            scored.append((sc, rx, rec, blob))

    if not scored:
        return {}

    scored.sort(key=lambda x: -x[0])
    top_hits = scored[:3]

    regexes = []
    consensus_candidates = []
    for _, rx, rec, blob in top_hits:
        if rx not in regexes:
            regexes.append(rx)
        consensus_candidates.append(_regex_to_consensus(rx))

    motif_seq_guess = consensus_candidates[0] if consensus_candidates else ""

    out = {
        "sequence": seq,
        "locked": {},
        "allowed_set": {},
        "banned_set": {},

        "regex_allow": regexes,
        "regex_forbid": [r"[KR]{6,}", r"[DE]{6,}", r"C.{0,3}C"],

        "global_soft": default_global_soft(),
        "decode": default_decode(),

        # 单个 motif（merge_constraints 会把它升级/合并成 motif list）
        "motif": {
            "sequence": motif_seq_guess,
            "left_linker": "",
            "right_linker": "",
            "flank_k": 8,
            "alpha": 1.0,
            "beta": 0.25,
            "forbid_overlap_locked": True
        },

        "soft_preferences": [],
        "window_min_counts": [],
        "window_charge": [],

        "_meta": {
            "prosite_used": True,
            "prosite_text_query": query,
            "prosite_hits_textbased": len(top_hits),
            "prosite_top_blob": top_hits[0][3] if top_hits else ""
        }
    }

    return out

def build_constraints_from_prosite(
    seq: str,
    prosite_json: str,
    base_span: Optional[Tuple[int,int]],
    win: int,
    hint_text: str = ""
) -> Dict:
    """
    模式A：在当前序列 seq 里跑 PROSITE motif 正则。
    命中的片段被视为“已有/天然口袋”，我们围绕这些坐标生成窗口约束、charge、疏水guard等。
    """

    idx = load_prosite_index(prosite_json)
    if not idx:
        return {}

    hits = []
    for rec in idx:
        rx = rec.get("RX")
        if not rx:
            continue
        try:
            for m in re.finditer(rx, seq):
                L = m.start() + 1  # 1-based inclusive
                R = m.end()        # 1-based inclusive end
                hits.append((L, R, rec))
        except re.error:
            # 如果 RX 不是合法的 python 正则，就跳过
            continue

    if not hits:
        return {}

    # 优先保留长的（长度越长越特异）
    hits.sort(key=lambda x: (-(x[1]-x[0]+1)))
    hits = hits[:5]

    # 用 hint_text + PROSITE描述 来推断 auto_key（决定KR/G/疏水性偏好）
    text_pool = " ".join(
        [hint_text] +
        [h[2].get("DE","") + " " + h[2].get("CC","") for h in hits]
    ).lower()

    auto_key = "unknown"
    for pat, key in [
        (r"heme|haem",             "heme_b"),
        (r"cxxch|cytochrome",      "heme_c"),
        (r"\bzn\b|zinc|cys2his2",  "zn"),
        (r"\bmg\b|magnesium",      "mg"),
        (r"\bca2?\+\b|calcium|ef-hand", "ca"),
        (r"\bnadp?\b|nadph",       "nad"),
        (r"\batp\b|\bgtp\b|\bntp\b|p-loop", "atp_gtp"),
        (r"flavin|fad|fmn",        "flavin"),
        (r"plp|pyridoxal",         "plp"),
        (r"biotin",                "biotin"),
        (r"iron[- ]sulfur|fe-s",   "fe_s"),
        (r"sam|s-adenosyl",        "sam"),
        (r"coenzyme a|coa",        "coa"),
    ]:
        if re.search(pat, text_pool):
            auto_key = key
            break

    out = {
        "sequence": seq,
        "locked": {},
        "allowed_set": {},
        "banned_set": {},

        "regex_allow": [],
        "regex_forbid": [r"[KR]{6,}", r"[DE]{6,}", r"C.{0,3}C"],

        "global_soft": default_global_soft(),
        "decode": default_decode(),

        # 我们这里不会直接强推 motif.sequence，
        # 因为这是“已有注释位点”，后面 inpaint 通常是保护它而不是新插它
        "motif": {
            "sequence":"",
            "left_linker":"",
            "right_linker":"",
            "flank_k":8,
            "alpha":1.0,
            "beta":0.25,
            "forbid_overlap_locked":True
        },

        "_meta": {
            "prosite_used": True,
            "prosite_hits": len(hits),
            "auto_key": auto_key
        }
    }

    # 把这些命中的正则pattern加到 regex_allow（弱地告诉 inpaint：别破坏这类模体）
    for _,_,rec in hits:
        rx = rec.get("RX")
        if rx and rx not in out["regex_allow"]:
            out["regex_allow"].append(rx)

    # 生成窗口（以命中段为中心拓成 win）
    spans = []
    for L, R, _rec in hits:
        length = max(win, R-L+1)
        mid = (L+R)//2
        s = max(1, mid - length//2)
        e = min(len(seq), s + length - 1)
        spans.append((s, e))

    # 如果有 base_span（例如我们自己选的口袋窗口），也加进去
    if base_span:
        spans = [base_span] + spans

    exemplar = fetch_exemplar_stats(auto_key)
    comp = exemplar.get("composition", {})
    kd_cap = exemplar.get("kd_max_mean", None)

    out["window_min_counts"] = []
    out["window_charge"] = []

    if spans:
        out.setdefault("hydrophobicity_guard", [])
        for (L,R) in spans[:3]:
            # KR enrichment + 正电
            if comp.get("KR") is not None:
                out["window_min_counts"].append({
                    "span":[L,R],
                    "allow_set":"KR",
                    "min_count": max(1,(R-L+1)//14)
                })
                out["window_charge"].append({
                    "span":[L,R],
                    "min_charge": +1,
                    "max_charge": +8
                })

            # Gly loop enrichment
            if comp.get("G") is not None:
                out["window_min_counts"].append({
                    "span":[L,R],
                    "allow_set":"G",
                    "min_count": max(1,(R-L+1)//12)
                })

            # 疏水性上限
            if kd_cap is not None:
                out["hydrophobicity_guard"].append({
                    "span":[L,R],
                    "max_mean_KD": kd_cap,
                    "weight": 0.8
                })

    # 全局疏水 guard
    if kd_cap is not None:
        out.setdefault("hydrophobicity_guard", [])
        out["hydrophobicity_guard"].append({
            "span":[1,len(seq)],
            "max_mean_KD": kd_cap,
            "weight": 0.6
        })

    return out

# -------------------- 合并逻辑（含 motif list 支持） --------------------

def _extend_unique_str_list(base: List[str], add: List[str]) -> List[str]:
    s = set(base)
    for x in add or []:
        if x not in s:
            base.append(x)
            s.add(x)
    return base

def _norm_span_pair(obj):
    if not obj or not isinstance(obj, (list,tuple)) or len(obj)!=2:
        return None
    i,j = int(obj[0]), int(obj[1])
    if i>j:
        i,j=j,i
    return (i,j)

def _dict_list_key(d: dict) -> Tuple:
    span = _norm_span_pair(d.get("span")) if isinstance(d,dict) else None
    rest = tuple(sorted([
        (k, json.dumps(v, sort_keys=True))
        for k,v in d.items() if k!="span"
    ]))
    return (span, rest)

def _extend_unique_dict_list(base: List[dict], add: List[dict]) -> List[dict]:
    seen = set(_dict_list_key(x) for x in base if isinstance(x,dict))
    for d in add or []:
        if not isinstance(d,dict):
            continue
        key = _dict_list_key(d)
        if key not in seen:
            base.append(d)
            seen.add(key)
    return base

def _normalize_motif_list(obj) -> List[dict]:
    """
    把 motif 字段(可能是None/{} or dict or list[dict])规范成 list[dict]。
    确保每个元素有标准字段。
    """
    if obj is None:
        base_list = []
    elif isinstance(obj, list):
        base_list = obj[:]
    elif isinstance(obj, dict):
        base_list = [obj]
    else:
        base_list = []

    normalized = []
    for m in base_list:
        if not isinstance(m, dict):
            continue
        seq_str = str(m.get("sequence", "") or "")
        norm_entry = {
            "sequence": seq_str,
            "left_linker": m.get("left_linker", ""),
            "right_linker": m.get("right_linker", ""),
            "flank_k": int(m.get("flank_k", 8)),
            "alpha": float(m.get("alpha", 1.0)),
            "beta": float(m.get("beta", 0.25)),
            "forbid_overlap_locked": bool(m.get("forbid_overlap_locked", True)),
        }
        normalized.append(norm_entry)

    return normalized

def _merge_motif_lists(base_list: List[dict], add_list: List[dict]) -> List[dict]:
    """
    合并两个 motif 列表，按 sequence 去重，
    base_list 顺序优先。
    """
    out = []
    seen_seq = set()

    for m in base_list:
        seq = m.get("sequence", "")
        if seq not in seen_seq:
            out.append(m)
            seen_seq.add(seq)

    for m in add_list:
        seq = m.get("sequence", "")
        if seq not in seen_seq:
            out.append(m)
            seen_seq.add(seq)

    return out

def merge_constraints(base: Dict, add: Dict) -> Dict:
    """
    合并两个约束对象（base <- add）。

    关键：
    - motif 字段合并成 list[dict]，不会互相覆盖
    - regex/窗口类字段累积去重
    - global_soft/decode 只补不存在的 key
    - _meta 合并我们关心的元信息
    """

    out = dict(base)

    # sequence：如果 base 还没有，就用 add 的
    if not out.get("sequence"):
        out["sequence"] = add.get("sequence")

    # regex list 合并
    out["regex_allow"] = _extend_unique_str_list(
        list(out.get("regex_allow", [])),
        add.get("regex_allow", [])
    )
    out["regex_forbid"] = _extend_unique_str_list(
        list(out.get("regex_forbid", [])),
        add.get("regex_forbid", [])
    )

    # dict list 合并
    for key in (
        "soft_preferences",
        "soft_penalties",
        "window_min_counts",
        "window_charge",
        "hydrophobicity_guard",
        "secondary_structure_soft",
        "disorder_soft",
    ):
        out[key] = _extend_unique_dict_list(
            list(out.get(key, [])),
            add.get(key, [])
        )

    # motif 合并为列表
    base_motif_list = _normalize_motif_list(out.get("motif"))
    add_motif_list  = _normalize_motif_list(add.get("motif"))
    merged_motif_list = _merge_motif_lists(base_motif_list, add_motif_list)
    out["motif"] = merged_motif_list  # 即使空，也保持 []

    # global_soft：只补没有的 key
    gs = dict(out.get("global_soft") or {})
    for k,v in (add.get("global_soft") or {}).items():
        if k not in gs:
            gs[k] = v
    if gs:
        out["global_soft"] = gs

    # decode：只补没有的 key
    dec = dict(out.get("decode") or {})
    for k,v in (add.get("decode") or {}).items():
        if k not in dec:
            dec[k] = v
    if dec:
        out["decode"] = dec

    # _meta 合并：保留/更新我们关心的 keys
    meta = dict(out.get("_meta") or {})
    add_meta = add.get("_meta") or {}
    for k in (
        "detected_cofactor",
        "span",
        "win",
        "stride",
        "used_model",
        "span_locked_overlap",
        "auto_key",
        "auto_used",
        "prosite_used",
        "prosite_hits",
        "prosite_text_query",
        "prosite_hits_textbased",
        "prosite_top_blob",
    ):
        if k in add_meta:
            meta[k] = add_meta[k]
    if meta:
        out["_meta"] = meta

    return out

# -------------------- 主逻辑 --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence",
                    help="AA 序列或文件路径（如 --in_json 没有 sequence，就必须给这个）")
    ap.add_argument("--cofactor",
                    required=True,
                    help='一个或多个配体/辅因子描述，用逗号或分号分隔，比如 "Mg2+, acetyl-CoA; UDP-GlcNAc"')
    ap.add_argument("--in_json",
                    help="已有 constraints.json，会被合并并回写")
    ap.add_argument("--win", type=int, default=16,
                    help="窗口长度 (默认16)")
    ap.add_argument("--stride", type=int, default=4,
                    help="窗口滑动步长 (默认4)")
    ap.add_argument("--prosite_index",
                    default="./pfam/prosite_index.json",
                    help="PROSITE 索引 JSON 路径")
    args = ap.parse_args()

    # 1. 读取已有 constraints.json（如果有）
    base_constraints = {}
    base_seq = ""
    if args.in_json and os.path.exists(args.in_json):
        with open(args.in_json, "r", encoding="utf-8") as f:
            base_constraints = json.load(f)
        base_seq = (base_constraints.get("sequence") or "").strip().upper()

    # 2. 拿到目标序列
    seq_cli = load_seq(args.sequence) if args.sequence else ""
    if seq_cli:
        if base_seq and base_seq != seq_cli:
            print("[WARN] --sequence 与 --in_json.sequence 不一致，将以 --sequence 为准", file=sys.stderr)
        seq = seq_cli
    else:
        if not base_seq:
            raise ValueError("No sequence provided: either pass --sequence or ensure --in_json has 'sequence'.")
        seq = base_seq

    # 3. 从 base_constraints 提取 locked 位置
    locked_set: Set[int] = parse_locked_positions(base_constraints)

    # 4. 拆解 --cofactor 为多个短描述
    cof_items = [re.sub(r"\s+"," ", s).strip()
                 for s in re.split(r"[;,]", args.cofactor)
                 if s.strip()]
    if not cof_items:
        raise ValueError("Empty --cofactor after splitting by comma/semicolon.")

    # 5. 初始化 merged = base_constraints 或一个空壳
    if base_constraints:
        merged = deepcopy(base_constraints)
    else:
        merged = {
            "sequence": seq,
            "regex_allow": [],
            "regex_forbid": [],
            "soft_preferences": [],
            "soft_penalties": [],
            "window_min_counts": [],
            "window_charge": [],
            "global_soft": default_global_soft(),
            "decode": default_decode(),
            "motif": []
        }

    # 6. 针对每个 cofactor 描述，依次添加约束
    for item in cof_items:
        cof_atom = detect_cofactor_atom(item)

        desire_map = {
            "mg": "DE","mn": "DE","ca": "DE",          # 酸性口袋
            "zn":"CH","ni":"CH","cu":"CH",             # Cys/His
            "fe_s":"C","heme_c":"C","heme_b":"FWY",    # 硫配位/血红素
            "atp_gtp":"G","nad":"G","flavin":"G",      # Gly-rich loop
            "plp":"K","sam":"FWY","biotin":"GAS",
            "tpp":"GAS","coa":"GAS",
            "": ""
        }
        desire = desire_map.get(cof_atom, "")

        span = None
        overlap = None

        # heme_c / fe_s 这种可能是明确motif驱动，就不强制窗口
        if cof_atom not in ("heme_c","fe_s"):
            span, overlap = choose_span_avoid_locked(
                seq=seq,
                win=max(8, args.win),
                stride=max(1, args.stride),
                desire=desire,
                locked=locked_set
            )

        # 6a. 手写 cofactor 规则
        local_frag = build_constraints_for(
            cof=cof_atom if cof_atom else "unknown",
            seq=seq,
            span=span,
            win=args.win
        )
        meta = {
            "detected_cofactor": cof_atom or "unknown",
            "span": span,
            "win": args.win,
            "stride": args.stride,
            "used_model": False
        }
        if overlap:
            meta["span_locked_overlap"] = int(overlap)
        local_frag["_meta"] = meta

        merged = merge_constraints(merged, local_frag)

        # 6b. PROSITE 模式A：扫描当前序列有没有已知 motif
        if args.prosite_index and os.path.exists(args.prosite_index):
            pro_frag_seqscan = build_constraints_from_prosite(
                seq=seq,
                prosite_json=args.prosite_index,
                base_span=span,
                win=args.win,
                hint_text=item
            )
            if pro_frag_seqscan:
                merged = merge_constraints(merged, pro_frag_seqscan)

        # 6c. PROSITE 模式B：根据文字描述去 PROSITE 里找最相近 motif（即使本序列还没有）
        if args.prosite_index and os.path.exists(args.prosite_index):
            pro_frag_text = build_constraints_from_prosite_textbased(
                seq=seq,
                prosite_json=args.prosite_index,
                hint_text=item,
                win=args.win
            )
            if pro_frag_text:
                merged = merge_constraints(merged, pro_frag_text)

        # 6d. 自动统计 / KR/G / 疏水guard
        auto_frag = build_constraints_auto(
            auto_key=(cof_atom or item),
            seq=seq,
            span=span,
            win=args.win
        )
        merged = merge_constraints(merged, auto_frag)

    # 7. 确保 sequence 是最终的
    merged["sequence"] = seq

    # 8. 打印 + 回写
    print(json.dumps(merged, ensure_ascii=False, indent=2))
    if args.in_json:
        with open(args.in_json, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
