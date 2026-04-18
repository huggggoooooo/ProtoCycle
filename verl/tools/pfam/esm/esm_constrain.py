# -*- coding: utf-8 -*-
"""
ESM unified constrained inpaint (+ multi-motif sequential implant with locking)
with Gibbs + Lagrangian biasing + optional N-terminal signal peptide append

本版特性（2025-10-25 修改版）:
- 纯插入，不做替换：fragment = left_linker + motif + right_linker
  直接插入到序列的某个gap里，原序列不删不覆盖，只整体右移。
- 不往连续锁区段(例如15-20整段都锁)内部插；这些区段视为“实心禁区”。
  只能插在这些锁区段之间的缝隙，或N端/C端。
- 插入后，fragment非'X'位点会立刻加锁。
- inpaint阶段只优化 fragment 周围 ± flank_k 的非锁位点；锁住的位点不mask。
- 仍然用分桶采样候选位置，避免永远插在末端。

运行示例:
  python esm_inpaint_unified.py --json constraints.json --model_dir /path/to/esm2_t36_3B_UR50D
"""

import os, re, json, math, random, argparse
from typing import Dict, List, Tuple, Optional, Union

import torch
from transformers import EsmTokenizer, EsmForMaskedLM

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA20_PLUS_X = set(AA20) | set("X")
POS_AA_CHARGE = set(["K","R","H"])
NEG_AA_CHARGE = set(["D","E"])
SMALL_AA     = set(list("ASGTV"))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def compute_charge(seq: str) -> int:
    pos = sum(1 for c in seq if c in POS_AA_CHARGE)
    neg = sum(1 for c in seq if c in NEG_AA_CHARGE)
    return pos - neg

def max_homopolymer_run(seq: str) -> int:
    if not seq: return 0
    best = 1; cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

def _normalize_motif_list(motif_field: Union[dict, list, None]) -> List[dict]:
    if motif_field is None:
        raw_list = []
    elif isinstance(motif_field, list):
        raw_list = motif_field
    elif isinstance(motif_field, dict):
        raw_list = [motif_field]
    else:
        raw_list = []

    normed = []
    for m in raw_list:
        if not isinstance(m, dict):
            continue
        seq = str(m.get("sequence","") or "").strip().upper()
        normed.append({
            "sequence": seq,
            "left_linker": m.get("left_linker",""),
            "right_linker": m.get("right_linker",""),
            "flank_k": int(m.get("flank_k", 8)),
            "alpha": float(m.get("alpha", 1.0)),
            "beta": float(m.get("beta", 0.25)),
            "forbid_overlap_locked": bool(m.get("forbid_overlap_locked", True)),
        })
    return normed

def load_constraints_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        c = json.load(f)

    def _to_int_key(d):
        if not isinstance(d, dict): return {}
        out = {}
        for k, v in d.items():
            try:
                out[int(k)] = v
            except Exception:
                pass
        return out

    c["locked"]            = _to_int_key(c.get("locked", {}))
    c["allowed_set"]       = _to_int_key(c.get("allowed_set", {}))
    c["banned_set"]        = _to_int_key(c.get("banned_set", {}))
    c["regex_allow"]       = c.get("regex_allow", [])
    c["regex_forbid"]      = c.get("regex_forbid", [])
    c["global_soft"]       = c.get("global_soft", {})
    c["decode"]            = c.get("decode", {})
    c["soft_preferences"]  = c.get("soft_preferences", [])
    c["soft_penalties"]    = c.get("soft_penalties", [])
    c["window_min_counts"] = c.get("window_min_counts", [])
    c["window_charge"]     = c.get("window_charge", [])
    c["_meta"]             = c.get("_meta", {})

    c["motif"] = _normalize_motif_list(c.get("motif", None))

    sp = c.get("signal_pep", None)
    if sp is not None:
        if isinstance(sp, str):
            sp = {"sequence": sp}
        sp.setdefault("sequence", "")
        sp.setdefault("fix_cleavage", True)
        sp.setdefault("post_inpaint", False)
        sp.setdefault("post_window_left", 6)
        sp.setdefault("post_window_right", 2)
        if sp["sequence"] and not set(sp["sequence"]).issubset(set(AA20)):
            raise ValueError("signal_pep.sequence must be 20AA only")
        c["signal_pep"] = sp

    if "sequence" not in c or not c["sequence"]:
        raise ValueError("JSON must contain non-empty 'sequence'")
    if not set(c["sequence"]).issubset(set(AA20)):
        raise ValueError("JSON 'sequence' must be 20AA only")

    for m in c["motif"]:
        seq = m.get("sequence")
        if not seq:
            continue

        # 如果存在不在 20AA+X 里的字符，就自动转成 X
        if not set(seq).issubset(AA20_PLUS_X):
            # 按字符替换非法字符为 'X'
            fixed = "".join(ch if ch in AA20_PLUS_X else "X" for ch in seq)
            m["sequence"] = fixed
            # 如果你想顺便记录一下发生了规范化，可以加一行 log：
            print(f"[motif normalize] invalid chars in '{seq}', normalized to '{fixed}'")

    return c

def build_token_maps(tokenizer: EsmTokenizer):
    AA2ID = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AA20}
    ID2AA = {v:k for k,v in AA2ID.items()}
    AA_IDS = torch.tensor([AA2ID[a] for a in AA20], device=device)
    return AA2ID, ID2AA, AA_IDS

def validate_regex(seq: str, regex_allow: List[str], regex_forbid: List[str]) -> bool:
    if regex_allow:
        if not any(re.search(pat, seq) for pat in regex_allow):
            return False
    if regex_forbid:
        for pat in regex_forbid:
            if re.search(pat, seq):
                return False
    return True

def global_soft_penalty(seq: str, cfg: dict) -> float:
    pen = 0.0
    if not cfg: return 0.0
    max_cys = cfg.get("max_cys", None)
    if isinstance(max_cys, int):
        cys = seq.count("C")
        if cys > max_cys:
            pen += (cys - max_cys) * 1.0
    charge_range = cfg.get("charge_range", None)
    if isinstance(charge_range, list) and len(charge_range) == 2:
        ch = compute_charge(seq)
        lo, hi = int(charge_range[0]), int(charge_range[1])
        if ch < lo: pen += (lo - ch) * 0.5
        if ch > hi: pen += (ch - hi) * 0.5
    mh = cfg.get("max_homopolymer", None)
    if isinstance(mh, int) and mh > 0:
        run = max_homopolymer_run(seq)
        if run > mh: pen += (run - mh) * 1.0
    forbid_sub = cfg.get("forbid_substrings", [])
    for sub in forbid_sub:
        if sub and sub in seq:
            pen += 2.0
    return pen

def _subseq_charge(s: str) -> int:
    pos = sum(1 for ch in s if ch in POS_AA_CHARGE)
    neg = sum(1 for ch in s if ch in NEG_AA_CHARGE)
    return pos - neg

def _check_window_min_counts(seq_new: str, rules: List[dict]) -> bool:
    if not rules: return True
    L = len(seq_new)
    for r in rules:
        span = r.get("span")
        if not span or len(span) != 2: continue
        i, j = int(span[0]), int(span[1])
        i = max(1, min(L, i)); j = max(1, min(L, j))
        if i > j: i, j = j, i
        allow_set = r.get("allow_set","")
        need = int(r.get("min_count", 0))
        subseq = seq_new[i-1:j]
        allow_letters = set(list(allow_set) if isinstance(allow_set, str) else allow_set)
        cnt = sum(1 for ch in subseq if ch in allow_letters)
        if cnt < need:
            return False
    return True

def _check_window_charge(seq_new: str, rules: List[dict]) -> bool:
    if not rules: return True
    L = len(seq_new)
    for r in rules:
        span = r.get("span")
        if not span or len(span) != 2: continue
        i, j = int(span[0]), int(span[1])
        i = max(1, min(L, i)); j = max(1, min(L, j))
        if i > j: i, j = j, i
        subseq = seq_new[i-1:j]
        ch = _subseq_charge(subseq)
        lo = r.get("min_charge", None)
        hi = r.get("max_charge", None)
        if lo is not None and ch < int(lo): return False
        if hi is not None and ch > int(hi): return False
    return True

def to_masked_inputs(tokenizer, seq: str, mask_spans: List[Tuple[int,int]]):
    s_list = list(seq)
    L = len(seq)
    for (a,b) in mask_spans:
        a = clamp(a, 1, L); b = clamp(b, 1, L)
        if a>b: a,b=b,a
        for i in range(a-1, b):
            s_list[i] = tokenizer.mask_token
    masked = "".join(s_list)
    return masked, tokenizer(masked, add_special_tokens=True, return_tensors="pt").to(device)

@torch.no_grad()
def sum_nll_on_spans(model, tokenizer, seq: str, spans: List[Tuple[int,int]], AA2ID: Dict[str,int]) -> float:
    if not spans: return 0.0
    _, inputs = to_masked_inputs(tokenizer, seq, spans)
    out = model(**inputs)
    logits = out.logits[0]
    input_ids = inputs["input_ids"][0]
    mask_id = tokenizer.mask_token_id
    total_nll = 0.0
    for tok_idx in range(1, input_ids.size(0)-1):
        if input_ids[tok_idx].item() == mask_id:
            pos1 = tok_idx
            if 1 <= pos1 <= len(seq):
                true_aa = seq[pos1-1]
                aa_id = AA2ID.get(true_aa, None)
                if aa_id is None:
                    continue
                row = logits[tok_idx]
                logprob = torch.log_softmax(row, dim=-1)[aa_id].item()
                total_nll += -logprob
    return float(total_nll)

def apply_hard_masks_to_logits(logits_row: torch.Tensor, pos1: int,
                               constraints_for_decode: dict,
                               AA_IDS: torch.Tensor,
                               AA2ID: Dict[str,int]):
    locked = constraints_for_decode.get("locked", {})
    allowed = constraints_for_decode.get("allowed_set", {})
    banned  = constraints_for_decode.get("banned_set", {})

    # disallow non-AA20
    disallowed_all = set(range(logits_row.size(-1)))
    for aa_id in AA_IDS.tolist():
        disallowed_all.discard(aa_id)
    if disallowed_all:
        logits_row[list(disallowed_all)] = -1e9

    if pos1 in locked:
        aa = locked[pos1]
        keep = set([AA2ID.get(aa, -1)])
        for aa_id in AA_IDS.tolist():
            if aa_id not in keep:
                logits_row[aa_id] = -1e9
        return

    if pos1 in allowed and allowed[pos1]:
        allow_ids = set(AA2ID[a] for a in allowed[pos1] if a in AA2ID)
        for aa_id in AA_IDS.tolist():
            if aa_id not in allow_ids:
                logits_row[aa_id] = -1e9

    if pos1 in banned and banned[pos1]:
        ban_ids = set(AA2ID[a] for a in banned[pos1] if a in AA2ID)
        for aa_id in ban_ids:
            logits_row[aa_id] = -1e9

def _letters_to_ids(spec, AA2ID):
    if isinstance(spec, str):
        letters = list(spec)
    elif isinstance(spec, list):
        letters = spec
    else:
        letters = []
    return [AA2ID[a] for a in letters if a in AA2ID]

def _apply_soft_preferences_row(logits_row: torch.Tensor, pos1: int,
                                soft_prefs: List[dict], AA2ID: Dict[str,int]):
    if not soft_prefs: return
    for pref in soft_prefs:
        span = pref.get("span")
        if not span or len(span) != 2: continue
        i, j = int(span[0]), int(span[1])
        if i <= pos1 <= j:
            w = float(pref.get("weight", 0.0))
            if w == 0.0: continue
            ids = _letters_to_ids(pref.get("prefer_set",""), AA2ID)
            if ids:
                logits_row[ids] = logits_row[ids] + w

def _apply_soft_penalties_row(logits_row: torch.Tensor, pos1: int,
                              soft_pens: List[dict], AA2ID: Dict[str,int]):
    if not soft_pens: return
    for pen in soft_pens:
        span = pen.get("span")
        if not span or len(span) != 2: continue
        i, j = int(span[0]), int(span[1])
        if i <= pos1 <= j:
            w = float(pen.get("weight", 0.0))
            ids = _letters_to_ids(pen.get("penalize_set",""), AA2ID)
            if ids and w > 0:
                logits_row[ids] = logits_row[ids] - w

def top_k_filter_(logits: torch.Tensor, k: Optional[int]):
    if k is None or k <= 0: return logits
    topk_vals, _ = torch.topk(logits, k)
    thresh = topk_vals[..., -1].unsqueeze(-1)
    logits[logits < thresh] = -1e9
    return logits

def sample_token_from_logits(logits_row: torch.Tensor,
                             temperature: float = 1.0,
                             top_k: Optional[int] = None) -> int:
    x = logits_row.clone()
    if (temperature is None) or (temperature <= 0):
        return int(torch.argmax(x).item())
    x = top_k_filter_(x, top_k)
    x = x / float(temperature)
    probs = torch.softmax(x, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

# -----------------------------
# 新增：锁的连续段 + 合法插入点
# -----------------------------

def _contiguous_lock_segments(locked_positions: set) -> List[Tuple[int,int]]:
    """
    给定一堆锁的1-based位置，找出所有连续区段 [L,R]。
    """
    if not locked_positions:
        return []
    arr = sorted(list(locked_positions))
    segs = []
    start = arr[0]; prev = arr[0]
    for p in arr[1:]:
        if p == prev + 1:
            prev = p
        else:
            segs.append((start, prev))
            start = p
            prev = p
    segs.append((start, prev))
    return segs

def _position_inside_any_segment(pos: int, segs: List[Tuple[int,int]]) -> bool:
    for (L,R) in segs:
        if L <= pos <= R:
            return True
    return False

def _bucketed_candidate_insert_points(L0: int,
                                      n_bucket: int = 3,
                                      per_bucket: int = 8) -> List[int]:
    """
    返回候选插入点 p (1-based “在p之前插入”；p可以是1..L0+1)
    我们像之前那样三桶采样，避免只在C端。
    """
    pts = set()
    for b in range(n_bucket):
        lo = int(b     * (L0+1) / n_bucket) + 1
        hi = int((b+1) * (L0+1) / n_bucket)
        lo = clamp(lo, 1, L0+1)
        hi = clamp(hi, 1, L0+1)
        if hi < lo:
            hi = lo
        if per_bucket <= 1:
            mids = [(lo+hi)//2]
        else:
            mids = [
                int(round(lo + (hi-lo)*i/max(1,per_bucket-1)))
                for i in range(per_bucket)
            ]
        for p in mids:
            if 1 <= p <= L0+1:
                pts.add(p)
    pts = sorted(list(pts))
    return pts

@torch.no_grad()
def _score_fragment_insertion(
    base_seq: str,
    insert_before_pos1: int,
    fragment: str,
    flank_k: int,
    AA2ID,
    model,
    tokenizer
):
    """
    在 base_seq 里，把 fragment 插到 insert_before_pos1 (1-based) 之前。
    insert_before_pos1 可以是 L0+1 => 追加到末尾(尾插)。
    返回:
      score, motif_span_after, nll_motif, nll_flank, new_seq
    """
    L0 = len(base_seq)
    if insert_before_pos1 < 1 or insert_before_pos1 > (L0+1):
        return None

    new_seq = base_seq[:insert_before_pos1-1] + fragment + base_seq[insert_before_pos1-1:]

    ms = insert_before_pos1
    me = insert_before_pos1 + len(fragment) - 1

    spans_motif = [(ms, me)]
    nll_motif = sum_nll_on_spans(model, tokenizer, new_seq, spans_motif, AA2ID)

    left_span  = (max(1, ms - flank_k), ms-1)
    right_span = (me+1, min(len(new_seq), me + flank_k))
    spans_flank = []
    if left_span[0]  <= left_span[1]:  spans_flank.append(left_span)
    if right_span[0] <= right_span[1]: spans_flank.append(right_span)

    nll_flank = sum_nll_on_spans(model, tokenizer, new_seq, spans_flank, AA2ID)

    score = nll_motif + 0.25 * nll_flank
    return (
        score,
        (ms, me),
        nll_motif,
        nll_flank,
        new_seq
    )

@torch.no_grad()
def place_motif_insert_only(
    base_seq: str,
    motif_seq: str,
    left_linker: str,
    right_linker: str,
    flank_k: int,
    forbid_overlap_locked: bool,
    global_locked_positions: set,
    AA2ID,
    model,
    tokenizer,
):
    """
    纯插入版：
    - fragment = left_linker + motif_seq + right_linker
    - 我们只考虑把 fragment 插到“允许的gap”：
        * 候选点 p 来自 _bucketed_candidate_insert_points
        * 如果 forbid_overlap_locked=True，则禁止把插入点 p 落在某个连续锁区段 [L,R] 的内部
          （直觉：不要把motif打断一整段已经锁死的功能核心）
    - 评分用 _score_fragment_insertion
    - 返回 score 最小的候选
    """
    fragment = left_linker + motif_seq + right_linker
    frag_len = len(fragment)
    if frag_len == 0:
        return None  # nothing to insert

    L0 = len(base_seq)
    conti_locks = _contiguous_lock_segments(global_locked_positions)

    candidate_points = _bucketed_candidate_insert_points(
        L0,
        n_bucket=3,
        per_bucket=8,
    )

    best = None
    for p in candidate_points:
        # p 是“在第p位之前插入”
        # 例: p=1 => 前插; p=L0+1 => 末尾追加; 1<p<=L0 => 中间
        if forbid_overlap_locked:
            # 不允许在一个连续锁段的内部劈开它:
            # 如果 p 落在 [L,R] 的严格内部 (L < p <= R)，意味着我们把fragment塞进锁块里
            # 我们拒绝这种插入
            bad_here = False
            for (L,R) in conti_locks:
                if L < p <= R:
                    bad_here = True
                    break
            if bad_here:
                continue

        scored = _score_fragment_insertion(
            base_seq=base_seq,
            insert_before_pos1=p,
            fragment=fragment,
            flank_k=flank_k,
            AA2ID=AA2ID,
            model=model,
            tokenizer=tokenizer
        )
        if scored is None:
            continue
        score, (ms,me), nll_m, nll_f, new_seq = scored

        cand = {
            "insert_before": p,
            "new_seq": new_seq,
            "score": score,
            "nll_motif": nll_m,
            "nll_flank": nll_f,
            "fragment": fragment,
            "motif_span_after": (ms, me),
        }
        if (best is None) or (score < best["score"]):
            best = cand

    return best

# -----------------------------
# 前向 logits（保持不变）
# -----------------------------

@torch.no_grad()
def _forward_logits_for_masked(model, tokenizer, seq_chars: List[str],
                               mask_positions_1based: List[int]):
    s = seq_chars[:]
    for p in mask_positions_1based:
        s[p-1] = tokenizer.mask_token
    inputs = tokenizer("".join(s), add_special_tokens=True, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0]
    input_ids = inputs["input_ids"][0]
    mask_token_id = tokenizer.mask_token_id
    toks = [
        ti for ti in range(1, input_ids.size(0)-1)
        if input_ids[ti].item()==mask_token_id
    ]
    assert len(toks) == len(mask_positions_1based), "Mask alignment error"
    return logits

# -----------------------------
# inpaint 主循环（单个 motif）
# -----------------------------

@torch.no_grad()
def inpaint_unified_once(
    base_seq: str,
    single_motif_cfg: dict,
    constraints: dict,
    regex_allow: List[str],
    regex_forbid: List[str],
    global_soft: dict,
    decode: dict,
    model,
    tokenizer,
    AA2ID: Dict[str,int],
    ID2AA: Dict[int,str],
    AA_IDS: torch.Tensor,
    global_locked_positions: set,
):
    """
    单轮：
      1. motif 纯插入(place_motif_insert_only)
      2. 构造 mask window = 新插入motif区段 ± flank_k
         仅对这些邻域中非锁位点mask做修补 (Gibbs)
      3. 约束检验
    返回：
      new_seq, info, updated_locked_positions_set
    """

    temperature = float(decode.get("temperature", 0.6))
    top_k       = decode.get("top_k", 8); top_k = int(top_k) if (top_k is not None) else None
    num_candidates = int(decode.get("num_candidates", 64))
    max_retries    = int(decode.get("max_retries", 256))
    gibbs_iters = int(decode.get("gibbs_iters", 6))
    gibbs_resample_frac = float(decode.get("gibbs_resample_frac", 0.5))

    seq_work = base_seq
    info = {"stages":{}, "fail_stats":{"regex":0,"min_counts":0,"charge":0,"other":0}}
    motif_span_new = None
    motif_fragment = None

    # ---------- Step 1. 插入 ----------
    motif_seq = single_motif_cfg.get("sequence","") or ""
    left_linker = single_motif_cfg.get("left_linker","")
    right_linker = single_motif_cfg.get("right_linker","")
    flank_k = int(single_motif_cfg.get("flank_k", 8))
    forbid_overlap_locked = bool(single_motif_cfg.get("forbid_overlap_locked", True))

    if motif_seq:
        best = place_motif_insert_only(
            base_seq=seq_work,
            motif_seq=motif_seq,
            left_linker=left_linker,
            right_linker=right_linker,
            flank_k=flank_k,
            forbid_overlap_locked=forbid_overlap_locked,
            global_locked_positions=global_locked_positions,
            AA2ID=AA2ID,
            model=model,
            tokenizer=tokenizer,
        )

        if best is None:
            info["stages"]["implant"] = {"status":"failed_no_slot"}
        else:
            seq_work = best["new_seq"]
            motif_span_new = best["motif_span_after"]  # (ms, me)
            motif_fragment = best["fragment"]

            # 更新全局锁:
            # 1. 旧锁全部向右平移：插入点之后的坐标 += fragment_len
            insert_before = best["insert_before"]
            frag_len = len(motif_fragment)
            shifted_locked = set()
            for p in global_locked_positions:
                if p < insert_before:
                    shifted_locked.add(p)
                else:
                    shifted_locked.add(p + frag_len)

            # 2. 把新motif区域(除了'X')加锁
            ms, me = motif_span_new
            for offset, p in enumerate(range(ms, me+1)):
                aa_here = motif_fragment[offset] if offset < len(motif_fragment) else None
                if aa_here is None:
                    continue
                if aa_here == 'X':
                    continue
                shifted_locked.add(p)

            global_locked_positions = shifted_locked

            info["stages"]["implant"] = {
                "status": "ok",
                "span": [motif_span_new[0], motif_span_new[1]],
                "score": best["score"],
                "nll_motif": best["nll_motif"],
                "nll_flank": best["nll_flank"],
                "insert_before": insert_before,
                "frag_len": frag_len,
            }
    else:
        info["stages"]["implant"] = {"status":"skip_empty_motif"}

    # ---------- Step 2. mask window ----------
    seq_len_now = len(seq_work)
    mask_windows: List[Tuple[int,int]] = []

    # 2a. 只围绕新motif做局部修补（± flank_k）
    if motif_span_new:
        s_m, e_m = motif_span_new
        a = max(1, s_m - flank_k)
        b = min(seq_len_now, e_m + flank_k)
        mask_windows.append((a,b))

    # 2b. 仍然可以把 constraints 里显式指定的 span 带进来
    #     但是我们后面会在具体mask时过滤掉锁位点
    spans_from_rules: List[Tuple[int,int]] = []
    for key in ("soft_preferences","soft_penalties","window_min_counts","window_charge"):
        for r in constraints.get(key, []):
            s = r.get("span")
            if s and len(s)==2:
                i,j = int(s[0]), int(s[1])
                i = max(1, min(seq_len_now, i))
                j = max(1, min(seq_len_now, j))
                if i>j: i,j=j,i
                spans_from_rules.append((i,j))
    meta = constraints.get("_meta", {})
    if isinstance(meta, dict) and meta.get("span"):
        s = meta["span"]
        if isinstance(s, (list,tuple)) and len(s)==2:
            i,j = int(s[0]), int(s[1])
            i = max(1, min(seq_len_now, i))
            j = max(1, min(seq_len_now, j))
            if i>j: i,j=j,i
            spans_from_rules.append((i,j))

    # 合并 [motif±flank] 和 其它span
    for (i,j) in spans_from_rules:
        mask_windows.append((i,j))

    if not mask_windows:
        pen = global_soft_penalty(seq_work, global_soft)
        ok = validate_regex(seq_work, regex_allow, regex_forbid)
        info["stages"]["no_mask"] = {"penalty": pen, "regex_ok": ok}
        return seq_work, info, global_locked_positions

    mask_windows.sort()
    merged = []
    for w in mask_windows:
        if not merged or w[0] > merged[-1][1] + 1:
            merged.append([w[0], w[1]])
        else:
            merged[-1][1] = max(merged[-1][1], w[1])
    mask_windows = [tuple(x) for x in merged]

    # ---------- Step 3. Gibbs inpaint on mask windows ----------
    seq_chars = list(seq_work)

    # build locked map {pos:AA} after insertion
    locked_new_map = {}
    for p in global_locked_positions:
        if 1 <= p <= seq_len_now:
            locked_new_map[p] = seq_chars[p-1]

    # collect candidate mask positions:
    #  (a) inside mask_windows
    #  (b) NOT locked_new_map
    mask_marks = [False]*(seq_len_now+1)
    for (a,b) in mask_windows:
        a = clamp(a,1,seq_len_now); b = clamp(b,1,seq_len_now)
        if a>b: a,b=b,a
        for pos in range(a,b+1):
            mask_marks[pos] = True
    mask_positions = [
        p for p in range(1, seq_len_now+1)
        if mask_marks[p] and (p not in locked_new_map)
    ]

    if not mask_positions:
        pen = global_soft_penalty(seq_work, global_soft)
        info["stages"]["only_locked"] = {"penalty": pen}
        return "".join(seq_chars), info, global_locked_positions

    temperature = float(decode.get("temperature", 0.6))
    top_k       = decode.get("top_k", 8); top_k = int(top_k) if (top_k is not None) else None
    num_candidates = int(decode.get("num_candidates", 64))
    max_retries    = int(decode.get("max_retries", 256))
    gibbs_iters    = int(decode.get("gibbs_iters", 6))
    gibbs_resample_frac = float(decode.get("gibbs_resample_frac", 0.5))

    best_init = None
    best_score = (1e9, 1e9)
    attempts = 0

    constraints_for_decode = {
        "locked": locked_new_map,
        "allowed_set": constraints.get("allowed_set", {}),
        "banned_set": constraints.get("banned_set", {}),
    }

    while attempts < max_retries and (best_init is None) and attempts < num_candidates:
        attempts += 1
        logits_all = _forward_logits_for_masked(model, tokenizer, seq_chars, mask_positions)
        cur = seq_chars[:]
        for ti in mask_positions:
            row = logits_all[ti].clone()
            apply_hard_masks_to_logits(row, ti, constraints_for_decode, AA_IDS, AA2ID)
            _apply_soft_preferences_row(row, ti, constraints.get("soft_preferences", []), AA2ID)
            _apply_soft_penalties_row(row, ti, constraints.get("soft_penalties", []), AA2ID)
            aa_id = sample_token_from_logits(row, temperature=temperature, top_k=top_k)
            cur[ti-1] = ID2AA.get(aa_id, cur[ti-1])
        seq0 = "".join(cur)

        vio = 0
        if not validate_regex(seq0, regex_allow, regex_forbid):
            vio += 1; info["fail_stats"]["regex"] += 1
        if not _check_window_min_counts(seq0, constraints.get("window_min_counts", [])):
            vio += 1; info["fail_stats"]["min_counts"] += 1
        if not _check_window_charge(seq0, constraints.get("window_charge", [])):
            vio += 1; info["fail_stats"]["charge"] += 1
        pen = global_soft_penalty(seq0, global_soft)

        if (vio, pen) < best_score:
            best_score = (vio, pen)
            best_init = cur

    if best_init is None:
        best_init = seq_chars[:]
    seq_chars = best_init

    def window_stats(seq_s: str, rule: dict):
        i,j = rule['span']
        sseg = seq_s[i-1:j]
        cnt_de = sum(1 for ch in sseg if ch in NEG_AA_CHARGE)
        chg = _subseq_charge(sseg)
        return cnt_de, chg

    eta = 0.7
    for it in range(gibbs_iters):
        seq_now = "".join(seq_chars)
        satisfied = (
            validate_regex(seq_now, regex_allow, regex_forbid)
            and _check_window_min_counts(seq_now, constraints.get("window_min_counts", []))
            and _check_window_charge(seq_now, constraints.get("window_charge", []))
        )
        if satisfied:
            info["stages"]["gibbs"] = {"iters": it, "status": "satisfied"}
            break

        worst_spans = set()
        for r in constraints.get("window_min_counts", []):
            if not r.get('span'): continue
            i,j = r['span']
            cnt_de, _ = window_stats(seq_now, r)
            need = int(r.get('min_count',0))
            if cnt_de < need:
                worst_spans.add((i,j))
        for r in constraints.get("window_charge", []):
            if not r.get('span'): continue
            i,j = r['span']
            _, chg = window_stats(seq_now, r)
            hi = r.get('max_charge', None)
            if (hi is not None) and (chg > int(hi)):
                worst_spans.add((i,j))

        worst_pos = set()
        for (i,j) in worst_spans:
            for p in range(i,j+1):
                worst_pos.add(p)

        scored_positions = []
        for ti in mask_positions:
            aa = seq_chars[ti-1]
            sc = 0
            if aa in POS_AA_CHARGE: sc += 2
            if aa not in NEG_AA_CHARGE: sc += 1
            if ti in worst_pos: sc += 1
            scored_positions.append((sc, ti))
        scored_positions.sort(reverse=True)

        k = max(1, int(len(scored_positions) * gibbs_resample_frac))
        to_change = [ti for _,ti in scored_positions[:k]]

        logits_all = _forward_logits_for_masked(model, tokenizer, seq_chars, to_change)

        lambda_de, lambda_pos = 0.0, 0.0
        for r in constraints.get("window_min_counts", []):
            if not r.get('span'): continue
            cnt_de, _ = window_stats(seq_now, r)
            need = int(r.get('min_count',0))
            if cnt_de < need:
                lambda_de += eta * (need - cnt_de)
        for r in constraints.get("window_charge", []):
            if not r.get('span'): continue
            _, chg = window_stats(seq_now, r)
            hi = r.get('max_charge', None)
            if (hi is not None) and (chg > int(hi)):
                lambda_pos += eta * (chg - int(hi))

        lambda_de  = clamp(lambda_de,  0.0, 2.0)
        lambda_pos = clamp(lambda_pos, 0.0, 2.0)
        ids_DE  = _letters_to_ids("DE", AA2ID)
        ids_KRH = _letters_to_ids("KRH", AA2ID)

        for ti in to_change:
            row = logits_all[ti].clone()
            apply_hard_masks_to_logits(row, ti, constraints_for_decode, AA_IDS, AA2ID)
            _apply_soft_preferences_row(row, ti, constraints.get("soft_preferences", []), AA2ID)
            _apply_soft_penalties_row(row, ti, constraints.get("soft_penalties", []), AA2ID)

            if ids_DE:
                row[ids_DE] = row[ids_DE] + lambda_de
            if ids_KRH:
                row[ids_KRH] = row[ids_KRH] - lambda_pos

            aa_id = sample_token_from_logits(row,
                                             temperature=temperature,
                                             top_k=top_k)
            seq_chars[ti-1] = ID2AA.get(aa_id, seq_chars[ti-1])

        if it == gibbs_iters - 1:
            info["stages"]["gibbs"] = {"iters": gibbs_iters, "status": "max_iter"}

    seq_final = "".join(seq_chars)
    ok_regex = validate_regex(seq_final, regex_allow, regex_forbid)
    ok_min   = _check_window_min_counts(seq_final, constraints.get("window_min_counts", []))
    ok_chg   = _check_window_charge(seq_final, constraints.get("window_charge", []))
    if not ok_regex: info["fail_stats"]["regex"] += 1
    if not ok_min:   info["fail_stats"]["min_counts"] += 1
    if not ok_chg:   info["fail_stats"]["charge"] += 1

    info["final_check"] = {
        "regex_ok": ok_regex,
        "min_counts_ok": ok_min,
        "charge_ok": ok_chg,
        "penalty": global_soft_penalty(seq_final, global_soft),
        "mask_windows": mask_windows
    }

    return seq_final, info, global_locked_positions

# -----------------------------
# 信号肽段落（保持原逻辑）
# -----------------------------

def _fix_cleavage_site_in_sp(sp_seq: str):
    L = len(sp_seq)
    if L < 3:
        return sp_seq, {"reason": "sp_too_short"}

    arr = list(sp_seq)
    changes = {}
    for idx in [L-3, L-1]:
        aa = arr[idx]
        if (aa not in SMALL_AA) or (aa == 'P'):
            arr[idx] = 'A'
            changes[idx+1] = {"from": aa, "to": "A"}

    for i in range(max(0, L-6), L):
        aa = arr[i]
        if aa in POS_AA_CHARGE or aa in NEG_AA_CHARGE or aa == 'P':
            new_aa = 'A' if aa != 'G' else 'G'
            if new_aa != aa:
                arr[i] = new_aa
                if (i+1) not in changes:
                    changes[i+1] = {"from": aa, "to": new_aa}

    return "".join(arr), {"changes": changes}

def apply_signal_pep_if_any(
    seq_after_all_motifs: str,
    cfg: dict,
    model,
    tokenizer,
    AA2ID, ID2AA, AA_IDS
):
    sp_cfg = cfg.get("signal_pep", None)
    if not sp_cfg or not sp_cfg.get("sequence"):
        return seq_after_all_motifs, {"status":"no_signal_pep"}

    sp = sp_cfg["sequence"]
    if not sp.startswith("M"):
        sp = "M" + sp

    report = {"status": "sp_appended", "sp_len": len(sp)}

    seq_concat = sp + seq_after_all_motifs

    if sp_cfg.get("fix_cleavage", True):
        sp_fixed, fix_info = _fix_cleavage_site_in_sp(sp)
        if sp_fixed != sp:
            report["cleavage_fix"] = fix_info
            sp = sp_fixed
            seq_concat = sp + seq_after_all_motifs

    if sp_cfg.get("post_inpaint", False):
        L_sp = len(sp)
        left  = int(sp_cfg.get("post_window_left", 6))
        right = int(sp_cfg.get("post_window_right", 2))
        a = max(1, L_sp - left + 1)
        b = min(L_sp + right, len(seq_concat))

        temp_constraints = {
            "locked": {},
            "allowed_set": {},
            "banned_set": {},
            "soft_preferences": [
                {"span":[max(1,L_sp-6+1), L_sp],
                 "prefer_set":"ASGT","weight":0.6}
            ],
            "soft_penalties": [
                {"span":[max(1,L_sp-6+1), L_sp],
                 "penalize_set":"P","weight":1.0},
                {"span":[max(1,L_sp-6+1), L_sp],
                 "penalize_set":"KRHDE","weight":0.5},
            ],
            "window_min_counts": [],
            "window_charge": [
                {"span":[max(1,L_sp-6+1), L_sp],
                 "max_charge":0}
            ],
            "regex_allow": [],
            "regex_forbid": [],
            "_meta":{"span":[a,b]}
        }

        seq_local, _, _ = inpaint_unified_once(
            base_seq=seq_concat,
            single_motif_cfg={"sequence": "", "flank_k": 0},
            constraints=temp_constraints,
            regex_allow=[],
            regex_forbid=[],
            global_soft={},
            decode={
                "temperature":0.6,
                "top_k":8,
                "num_candidates":16,
                "max_retries":64,
                "gibbs_iters":3,
                "gibbs_resample_frac":0.6
            },
            model=model, tokenizer=tokenizer,
            AA2ID=AA2ID, ID2AA=ID2AA, AA_IDS=AA_IDS,
            global_locked_positions=set()
        )
        report["post_inpaint_span"] = [a,b]
        seq_concat = seq_local

    return seq_concat, report

# -----------------------------
# main (基本不变)
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to constraints.json (will be read, possibly updated with sequence_inpaint)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/path/to/ProtoCycle/pfam/esm/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc",
        help="Local ESM checkpoint dir",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to write final inpaint result json (machine-readable for runtime)",
    )
    args = parser.parse_args()

    cfg = load_constraints_json(args.json)

    base_seq     = cfg["sequence"]
    motif_list   = cfg.get("motif", [])
    regex_allow  = cfg.get("regex_allow", [])
    regex_forbid = cfg.get("regex_forbid", [])
    global_soft  = cfg.get("global_soft", {})
    decode       = cfg.get("decode", {})

    tokenizer = EsmTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model     = EsmForMaskedLM.from_pretrained(args.model_dir, local_files_only=True).to(device)
    model.eval()

    AA2ID, ID2AA, AA_IDS = build_token_maps(tokenizer)

    print(f"[INFO] Original length: {len(base_seq)}")
    if motif_list:
        for idx, m in enumerate(motif_list):
            print(f"[INFO] Motif[{idx}]: '{m.get('sequence','')}' "
                  f"(L={len(m.get('sequence',''))}) flank_k={m.get('flank_k',8)}")
    else:
        print("[INFO] Motif list empty; we'll just inpaint using constraints.")

    running_seq = base_seq
    global_locked_positions = set(cfg.get("locked", {}).keys())
    per_motif_infos: List[dict] = []

    motif_iter_list = motif_list if motif_list else [ { "sequence": "" } ]

    for idx, motif_cfg_single in enumerate(motif_iter_list):
        seq_before = running_seq

        seq_after, info, global_locked_positions = inpaint_unified_once(
            base_seq=seq_before,
            single_motif_cfg=motif_cfg_single,
            constraints=cfg,
            regex_allow=regex_allow,
            regex_forbid=regex_forbid,
            global_soft=global_soft,
            decode=decode,
            model=model,
            tokenizer=tokenizer,
            AA2ID=AA2ID,
            ID2AA=ID2AA,
            AA_IDS=AA_IDS,
            global_locked_positions=global_locked_positions,
        )

        running_seq = seq_after
        info["motif_index"] = idx
        info["motif_sequence"] = motif_cfg_single.get("sequence","")
        per_motif_infos.append(info)

    best_seq = running_seq

    final_seq = best_seq
    sp_report = {}
    if cfg.get("signal_pep", None) and cfg["signal_pep"].get("sequence"):
        print('Adding signal pep:')
        final_seq, sp_report = apply_signal_pep_if_any(
            seq_after_all_motifs=best_seq,
            cfg=cfg,
            model=model, tokenizer=tokenizer,
            AA2ID=AA2ID, ID2AA=ID2AA, AA_IDS=AA_IDS
        )

    # write back to constraints json (sequence_inpaint)
    cfg["sequence_inpaint"] = final_seq
    wrote_back = False
    try:
        with open(args.json, "w") as f_out:
            json.dump(cfg, f_out, indent=2)
        wrote_back = True
    except Exception as e:
        print(f"[WARN] failed to write sequence_inpaint back to {args.json}: {e}")

    print("\n=== ORIGINAL SEQ ===")
    print(base_seq)

    print("\n=== FINAL SEQ (after motif insertion + local Gibbs repair + optional SP prepend) ===")
    print(final_seq)

    debug_info = {
        "per_motif_info": per_motif_infos,
        "final_locked_positions": sorted(list(global_locked_positions)),
        "signal_pep": sp_report if sp_report else None,
        "wrote_back_to_json": wrote_back,
        "json_path": args.json if wrote_back else None,
    }

    # >>> NEW: write machine-readable output to args.out <<<
    result_payload = {
        "sequence": final_seq,
        "original_sequence": base_seq,
        "debug_info": debug_info,
    }
    try:
        with open(args.out, "w") as f_js:
            json.dump(result_payload, f_js, indent=2)
    except Exception as e:
        print(f"[ERROR] failed to write out result json to {args.out}: {e}")


    # print("\n[DEBUG INFO]")
    # print(json.dumps(debug_info, indent=2))


if __name__ == "__main__":
    main()
