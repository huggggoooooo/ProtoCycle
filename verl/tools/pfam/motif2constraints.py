#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
motif2constraints_fuzzy.py

改动要点：
- 文本检索改为相似度匹配（fuzzy + bag-of-words cosine）
- 支持两种输入：
    1) --pattern "YXXL" 直接指定模式
    2) --desc "CTP synthase N-terminal" 根据描述在 ELM 里找
- 先用 ELM classes.tsv 找到最相似的类，再用 instances.tsv 找到真实实例，
  然后(离线+UniProt API)拉序列并裁剪出motif片段；如果失败就用 Regex 兜底合成一个最短序列
- 合并回 constraints.json 的时候：
    motif 字段现在统一是一个 list[dict]。
    我们会把新motif按 sequence 去重后 append 进去，而不是覆盖。

运行示例：
  python motif2constraints_fuzzy.py \
    --json constraints.json \
    --classes elm_classes.tsv \
    --instances elm_instances.tsv \
    --desc "CTP synthase N-terminal" \
    --topk 8 --min_score 0.2 --prefer-taxa "homo sapiens,mouse"

  或者直接指定pattern（不走ELM检索）：
  python motif2constraints_fuzzy.py \
    --json constraints.json \
    --classes elm_classes.tsv \
    --instances elm_instances.tsv \
    --pattern "YXXL"
"""

import os, re, csv, io, codecs, json, time, argparse, sys
from typing import Dict, List, Tuple, Optional
from collections import Counter
import difflib
import requests  # 需要联网拿UniProt序列。如果你想彻底离线，可以把相关逻辑去掉或缓存。

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

# ----------------- Robust TSV reader -----------------
def read_tsv(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"TSV not found: {path}")
    with open(path, "rb") as fb:
        raw = fb.read()
    text = codecs.decode(raw, "utf-8-sig", errors="replace")

    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        delimiter = dialect.delimiter
        # heuristic: if we guessed ',' but file clearly has tabs, prefer '\t'
        if delimiter != "\t":
            first_lines = text.splitlines()[:5]
            if max((ln.count("\t") for ln in first_lines), default=0) >= 1:
                delimiter = "\t"
    except Exception:
        delimiter = "\t"

    rows = []
    f = io.StringIO(text)
    reader = csv.DictReader(f, delimiter=delimiter)
    for row in reader:
        clean = {}
        for k, v in row.items():
            if isinstance(k, list):
                k = " ".join(map(str, k))
            k = (k or "").strip().strip('"').strip("'")

            if isinstance(v, list):
                v = " ".join(map(str, v))
            elif v is None:
                v = ""
            else:
                v = str(v)
            v = v.strip().strip('"').strip("'")

            clean[k] = v
        rows.append(clean)
    return rows

# ----------------- JSON helpers -----------------
def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ----------------- forbid regex helpers -----------------
def violates_forbid(seq: str, regex_forbid: List[str]) -> bool:
    for pat in regex_forbid or []:
        try:
            if re.search(pat, seq):
                return True
        except re.error:
            # ignore invalid regex
            pass
    return False

# ----------------- Pattern instantiation -----------------
def instantiate_x_pattern(pattern: str, regex_forbid: List[str]) -> str:
    """
    把类似 "YXXL" / "PXXP" 变成一个具体序列，尽量避免 regex_forbid.
    """
    pat = (pattern or "").strip().upper()

    # 特殊小词库兼容
    if pat in ("YXXL","YXXF"):
        for c in ["YQRL","YQTL","YNRL","YSQL","YKNL"]:
            if not violates_forbid(c, regex_forbid):
                return c
    if pat == "PXXP":
        for c in ["PAAP","PTPP","PVVP"]:
            if not violates_forbid(c, regex_forbid):
                return c

    # 通用规则：X -> A
    out=[]
    for ch in pat:
        if ch in AA20:
            out.append(ch)
        elif ch in {"X","x","."}:
            out.append("A")
        elif ch in {"Φ","φ"}:
            out.append("L")
        else:
            out.append("A")
    seq="".join(out)

    # 如果 forbid 冲突，尝试把 'A' 位点换成别的中性氨基酸
    if violates_forbid(seq, regex_forbid):
        for i,ch in enumerate(seq):
            if ch=="A":
                for rep in ["Q","N","S","T","G"]:
                    cand = seq[:i]+rep+seq[i+1:]
                    if not violates_forbid(cand, regex_forbid):
                        return cand
    return seq

# ----------------- fuzzy + cosine similarity -----------------
WORD_RE = re.compile(r"[a-zA-Z0-9\-\_]+")

def normalize_text(s: str) -> str:
    return " ".join(WORD_RE.findall((s or "").lower()))

def bow_vec(s: str) -> Counter:
    return Counter(normalize_text(s).split())

def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    inter = set(a.keys()) & set(b.keys())
    num = sum(a[t]*b[t] for t in inter)
    den = (sum(v*v for v in a.values())**0.5) * (sum(v*v for v in b.values())**0.5)
    return num/den if den>0 else 0.0

def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(
        a=normalize_text(a),
        b=normalize_text(b)
    ).ratio()

def class_text_blob(row: Dict[str,str]) -> Dict[str,str]:
    """
    提取我们要比对的两个主要字段：名字和描述
    """
    return {
        "name": row.get("FunctionalSiteName",""),
        "desc": row.get("Description","")
    }

def score_class_fuzzy(row: Dict[str,str], query: str) -> float:
    """
    给 ELM class 打分：query vs (FunctionalSiteName, Description)
    + 少量 "#Instances" 可信度加权
    """
    blob = class_text_blob(row)
    q = normalize_text(query)
    qv = bow_vec(q)
    score = 0.0

    weights = {"name":0.45,"desc":0.35}
    for k,w in weights.items():
        s = blob[k]
        score += w * (
            0.6 * cosine(qv, bow_vec(s)) +
            0.4 * fuzzy_ratio(q, s)
        )

    # 如果这个类实例数多，略微加分（更可信）
    try:
        n = int(row.get("#Instances","0"))
        score += min(n, 100)/1000.0  # +0 ~ 0.1
    except Exception:
        pass

    return score

def topk_classes(classes: List[Dict[str,str]],
                 query: str,
                 k: int=8,
                 min_score: float=0.2) -> List[Tuple[float,Dict[str,str]]]:
    scored = [(score_class_fuzzy(r, query), r) for r in classes]
    scored = [x for x in scored if x[0] >= min_score]
    scored.sort(key=lambda x:x[0], reverse=True)
    return scored[:k]

# ----------------- Instances pick & UniProt fetch -----------------
def split_accessions(s: str) -> List[str]:
    if not s:
        return []
    return [p for p in re.split(r"[;,\s]+", s.strip()) if p]

def choose_instance_row(inst_rows: List[Dict[str,str]],
                        prefer_taxa: Optional[List[str]]=None) -> Optional[Dict[str,str]]:
    """
    在同一个 ELMIdentifier 下的多个实例中，挑一个代表：
    - true positive优先
    - prefer_taxa命中的物种优先
    - 最后按 ProteinName 稳定排序
    """
    if not inst_rows:
        return None

    def key(r):
        logic = (r.get("InstanceLogic","") or "").lower()
        tp = 0 if "true positive" in logic else 1

        org = (r.get("Organism","") or "").lower()
        tax_pen = 0 if (prefer_taxa and any(t in org for t in prefer_taxa)) else 1

        return (tp, tax_pen, r.get("ProteinName",""))

    return sorted(inst_rows, key=key)[0]

class UniProtClient:
    """
    超轻量 uniprot REST 拉取序列，用本地json做cache。
    如果你要彻底离线，可以在cache里预先塞好，或者改成直接不查远端。
    """
    BASE = "https://rest.uniprot.org/uniprotkb"

    def __init__(self, timeout=12, retries=3, sleep=0.5, cache_path=".uniprot_seq_cache.json"):
        self.timeout=timeout
        self.retries=retries
        self.sleep=sleep
        self.cache_path=cache_path
        self.cache={}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path,"r",encoding="utf-8") as f:
                    self.cache=json.load(f)
            except Exception:
                self.cache={}

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path,"w",encoding="utf-8") as f:
                json.dump(self.cache,f)
        except Exception:
            pass

    def fetch_sequence(self, acc: str) -> Optional[str]:
        acc = acc.strip()
        if not acc:
            return None
        # cache first
        if acc in self.cache:
            return self.cache[acc]

        url = f"{self.BASE}/{acc}?fields=accession,sequence&format=json"

        for i in range(self.retries):
            try:
                r = requests.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    # fallback: search endpoint
                    s_url = (
                        f"{self.BASE}/search?"
                        f"query=accession:{acc}&fields=accession,sequence&format=json&size=1"
                    )
                    r2 = requests.get(s_url, timeout=self.timeout)
                    if r2.ok:
                        js = r2.json()
                        hits = js.get("results",[])
                        if hits:
                            seq = (hits[0].get("sequence") or {}).get("value","")
                            if seq:
                                self.cache[acc]=seq
                                self._save_cache()
                                return seq
                    return None

                r.raise_for_status()
                js = r.json()
                seq = (js.get("sequence") or {}).get("value","")
                if seq:
                    self.cache[acc]=seq
                    self._save_cache()
                    return seq

            except Exception:
                time.sleep(self.sleep*(i+1))
        return None

# ----------------- Regex → shortest sequence (subset) -----------------
def synth_from_regex(regex: str) -> str:
    """
    把 ELM class 的 Regex（基本上是 PROSITE 风格）变成一个最短“代表性”多肽。
    我们覆盖常见子集：
      - [KRH]    -> K
      - . or x   -> A
      - A{3}     -> AAA
      - (ALT|PPP)-> 取第一个选项的代表串再递归
      - 量词 ? * + {n,m}，我们都走最小可行长度
    这不是完美的正则->序列编译器，但通常能给 inpaint 一个最小 motif scaffold。
    """
    SAFE_MAX_REPEAT = 6
    R = re.sub(r"\s+","",(regex or ""))
    out=[]
    i=0

    def apply_q(piece, j):
        # 看下一个字符是不是 ?,*,+,{n} 之类的量词
        if j>=len(R):
            return piece, j
        if R[j]=="?":   # 0 or 1
            return "", j+1
        if R[j]=="+":   # 1 or more => keep once
            return piece, j+1
        if R[j]=="*":   # 0 or more => choose 0
            return "", j+1
        if R[j]=="{":
            k=j+1
            while k<len(R) and R[k]!="}":
                k+=1
            if k<len(R):
                spec=R[j+1:k]
                # "A{3}", "A{2,4}" 之类，取最小次数
                m = int((spec.split(",")[0] if "," in spec else spec) or "1")
                m = max(0, min(m, SAFE_MAX_REPEAT))
                # piece 是一位氨基酸 or "A"
                return piece*m, k+1
        return piece, j

    def pick_not(chars:set)->str:
        for ch in "AQNSTGILMVFWYPKRHDECT":
            if ch not in chars:
                return ch
        return "A"

    while i<len(R):
        ch=R[i]

        # 直接字母
        if ch in AA20:
            piece=ch
            i+=1
            piece,i=apply_q(piece,i)
            out.append(piece)
            continue

        # 通配/任意
        if ch in {".","x","X"}:
            piece="A"
            i+=1
            piece,i=apply_q(piece,i)
            out.append(piece)
            continue

        # 字符集: [KRH] / [^P]
        if ch=="[":
            i+=1
            neg=False
            charset=set()
            if i<len(R) and R[i]=="^":
                neg=True
                i+=1
            while i<len(R) and R[i]!="]":
                if R[i] in AA20:
                    charset.add(R[i])
                i+=1
            i+=1  # skip ']'
            pick = (next(iter(charset)) if charset else "A") \
                   if not neg else pick_not(charset)
            piece, i = apply_q(pick, i)
            out.append(piece)
            continue

        # 分组 (ALT|PPP) -> 取第一个分支的递归生成
        if ch=="(":
            depth=1
            j=i+1
            inner=[]
            while j<len(R) and depth>0:
                if R[j]=="(":
                    depth+=1
                elif R[j]==")":
                    depth-=1
                    if depth==0:
                        break
                elif R[j]=="|" and depth==1:
                    # 停在分支边界，只拿第一个分支
                    break
                inner.append(R[j])
                j+=1
            i=j+1
            piece = synth_from_regex("".join(inner))
            piece, i = apply_q(piece, i)
            out.append(piece)
            continue

        # 其它符号直接跳过
        i+=1

    return "".join(out) or "A"

# ----------------- motif list merge helpers -----------------

def _normalize_motif_list(obj) -> List[dict]:
    """
    把 cfg["motif"] 统一成 list[dict]，并补齐标准字段。
    支持:
      - None
      - {}
      - {"sequence":"..."}
      - [{"sequence":"..."}, {...}]
    """
    if obj is None:
        base_list = []
    elif isinstance(obj, list):
        base_list = obj[:]
    elif isinstance(obj, dict):
        base_list = [obj]
    else:
        base_list = []

    normed = []
    for m in base_list:
        if not isinstance(m, dict):
            continue
        seq_str = str(m.get("sequence", "") or "")
        normed.append({
            "sequence": seq_str,
            "left_linker": m.get("left_linker", ""),
            "right_linker": m.get("right_linker", ""),
            "flank_k": int(m.get("flank_k", 8)),
            "alpha": float(m.get("alpha", 1.0)),
            "beta": float(m.get("beta", 0.25)),
            "forbid_overlap_locked": bool(m.get("forbid_overlap_locked", True))
        })
    return normed

def _append_motif_dedup(old_list: List[dict], new_item: dict) -> List[dict]:
    """
    把 new_item 追加到 old_list（list[dict]）里，按 sequence 去重。
    """
    seq_new = new_item.get("sequence","")
    out = []
    seen = set()
    for m in old_list:
        s = m.get("sequence","")
        if s not in seen:
            out.append(m)
            seen.add(s)
    if seq_new and seq_new not in seen:
        out.append(new_item)
    return out

def build_motif_dict(seq: str) -> dict:
    """
    生成标准 motif dict，用于放进 motif list。
    """
    return {
        "sequence": seq,
        "left_linker": "",
        "right_linker": "",
        "flank_k": 8,
        "alpha": 1.0,
        "beta": 0.25,
        "forbid_overlap_locked": True
    }

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--classes", default=os.path.join(_THIS_DIR, "elm_classes.csv"))
    ap.add_argument("--instances", default=os.path.join(_THIS_DIR, "elm_instances.csv"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pattern")    # 直接给 motif pattern (如 "YXXL")
    g.add_argument("--desc")       # 自然语言描述
    ap.add_argument("--topk", type=int, default=8,
                    help="候选 ELM 类别数量")
    ap.add_argument("--min_score", type=float, default=0.2,
                    help="最低相似度阈值(0~1)")
    ap.add_argument("--prefer-taxa", default="",
                    help='实例选择偏好物种（逗号分隔，如 "homo sapiens,mouse"）')
    ap.add_argument("--cache", default=".uniprot_seq_cache.json")
    args = ap.parse_args()

    cfg = load_json(args.json)
    regex_forbid = cfg.get("regex_forbid", [])

    classes = read_tsv(args.classes)
    instances = read_tsv(args.instances)

    # ------------- 决定最终 motif 序列 seq -------------
    if args.pattern:
        # 直接从pattern合成具体AA串
        seq = instantiate_x_pattern(args.pattern, regex_forbid)
        source = f"pattern:{args.pattern}"

    else:
        # fuzzy 找 ELM 类
        cands = topk_classes(
            classes,
            args.desc,
            k=args.topk,
            min_score=args.min_score
        )
        if not cands:
            sys.exit("ERROR: No ELM class passed the similarity threshold. Try lowering --min_score.")

        prefer_taxa = [
            t.strip().lower()
            for t in args.prefer_taxa.split(",")
            if t.strip()
        ]

        up = UniProtClient(cache_path=args.cache)

        seq = None
        source = None
        last_regex = None

        for sc, row in cands:
            elm_id = row.get("ELMIdentifier","")
            last_regex = row.get("Regex","") or last_regex

            # 找所有这个类别的实例
            inst_rows = [r for r in instances if (r.get("ELMIdentifier","")==elm_id)]
            inst = choose_instance_row(inst_rows, prefer_taxa=prefer_taxa) \
                   or (inst_rows[0] if inst_rows else None)
            if not inst:
                continue

            # 拿 accession(s)
            accs = []
            prim = inst.get("Primary_Acc","").strip()
            if prim:
                accs.append(prim)
            accs += split_accessions(inst.get("Accessions",""))

            start = int(inst.get("Start","0") or 0)
            end   = int(inst.get("End","0") or 0)
            if not (start and end and start <= end):
                continue

            # 依次尝试从 UniProt 拉全长，再切片
            for acc in accs:
                full = up.fetch_sequence(acc)
                if not full:
                    continue
                if end > len(full):
                    continue
                frag = full[start-1:end].upper()
                if frag and set(frag).issubset(AA20):
                    seq = frag
                    source = f"uniprot:{acc}:{start}-{end}:{elm_id};score={sc:.3f}"
                    break

            if seq:
                break

        # 如果上面都没成功，兜底用 regex 合成
        if not seq:
            seq = synth_from_regex(last_regex or "")
            source = f"regex_synth_fallback;topk={len(cands)}"

    # forbid 检查后的小修
    if violates_forbid(seq, regex_forbid):
        # 尝试把 'A' 调整成别的中性氨基酸避免违禁pattern
        fixed = seq
        for i,ch in enumerate(seq):
            if ch=="A":
                for rep in ["Q","N","S","T","G"]:
                    cand = seq[:i] + rep + seq[i+1:]
                    if not violates_forbid(cand, regex_forbid):
                        fixed = cand
                        break
                # 如果修成功就更新并继续尝试下一个 A
        seq = fixed

    # 最终 clean 一下
    seq = "".join([c for c in seq.upper() if c in AA20])
    if not seq:
        sys.exit("ERROR: resolved motif sequence is empty or invalid.")

    # ------------- 把 motif 追加进 cfg["motif"] 列表 -------------
    new_motif_entry = build_motif_dict(seq)

    # 1) 先把现有 cfg["motif"] 规范成 list[dict]
    old_motif_list = _normalize_motif_list(cfg.get("motif"))

    # 2) 把新的 motif append（按 sequence 去重）
    merged_motif_list = _append_motif_dedup(old_motif_list, new_motif_entry)

    # 3) 回写
    cfg["motif"] = merged_motif_list

    # 保存
    save_json(args.json, cfg)

    print(f"[OK] appended motif.sequence = {seq}")
    print(f"[INFO] source = {source}")
    print(f"[INFO] now total motifs = {len(cfg['motif'])}")

if __name__ == "__main__":
    main()
