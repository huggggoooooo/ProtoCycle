#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
go2seq.py
输入一条“GO 相关短语或句子”，**不需要你先给 GO:xxxx**。
流程：
  1) 用 QuickGO 的搜索接口把自然语言解析成 GO 术语（可返回 MF/BP/CC 三个侧面）。
  2) 用命中的 GO:ID 去 UniProt REST 搜索蛋白序列（可过滤 reviewed/kingdom）。

示例：
  python go2seq.py --text "L-glucuronate reductase activity; regulation of translation; membrane" --size 30 --reviewed --kingdom bacteria --debug
  python go2seq.py --text "regulate bacterial-type flagellum assembly" --size 50

可选参数：
  --aspect MF/BP/CC 限定某些侧面（可多次给）
  --evidence EXP/IDA/IMP/IEA ... 仅在 UniProt 端做 GO 证据过滤（尽量保守）

注意：脚本仅做“召回 scaffold”，不进行折叠/打分；建议后续把能落地到序列层面的信息转成你的 JSON 约束，再用 inpaint 精修。
"""
import os
import re, sys, json, argparse, time, random, ssl, warnings
from typing import List, Dict, Optional, Tuple, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

# ---- endpoints ----
QUICKGO_SEARCH_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/search"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# ---- robust session ----
UA_HEADERS = {
    "User-Agent": "desc2seq-agent (contact@example.com)",  # TODO: 换成你的邮箱/标识
    "Accept": "*/*",
    "Connection": "keep-alive",
}
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def make_session(verify_ssl: bool = False, retries: int = 4, backoff: float = 0.8) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=10))
    s.headers.update(UA_HEADERS)
    if not verify_ssl:
        ssl._create_default_https_context = ssl._create_unverified_context
    return s

# ---- utils ----
STOPWORDS = {
    "a","an","the","and","or","of","from","to","in","on","for","with","within","via","by","as","that","this",
    "pathway","process","biosynthesis","metabolism","metabolic","biosynthetic",
    "protein","catalyzing","catalyzes","catalyze","reactions","reaction","activity","activities",
    "high","efficiency","specificity","regulate","regulation","regulatory","assembly","type"
}
TOKEN_RE = re.compile(r"[A-Za-z0-9\-\(\)\+\./]+")

def tokenize_keep_informative(text: str) -> List[str]:
    toks = [t for t in TOKEN_RE.findall(text) if t]
    out = []
    for t in toks:
        tl = t.lower()
        if tl in STOPWORDS:  continue
        if len(tl) < 3:      continue
        if tl.isdigit():     continue
        out.append(t)
    seen=set(); uniq=[]
    for t in out:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

# ---- A. QuickGO 搜索（自然语言→GO 术语） ----

ASPECT_MAP = {
    "MF": "molecular_function",
    "BP": "biological_process",
    "CC": "cellular_component",
}

ASPECT_INV = {v:k for k,v in ASPECT_MAP.items()}


def build_go_queries(text: str) -> List[Tuple[str, Dict[str,str]]]:
    """生成一组 QuickGO 搜索参数 (label, params)。
    - Q1: 整句短语（加引号）
    - Q2: 信息量最高 tokens 的 AND
    - Q3: AND 逐步放宽
    - Q4: OR 扩展
    - Q5: n-gram 短语 OR
    """
    variants: List[Tuple[str, Dict[str,str]]] = []
    full = " ".join(text.strip().split())
    if full:
        variants.append(("Q1_full_phrase", {"query": f"\"{full}\""}))

    toks = tokenize_keep_informative(text)
    def tok_score(t: str) -> Tuple[int,int]:
        bonus = int(bool(re.search(r"[\-\(\)\+\dA-Z]", t)))
        return (bonus, len(t))
    toks_sorted = sorted(toks, key=tok_score, reverse=True)

    for k in (5,4,3):
        if len(toks_sorted) >= k:
            variants.append((f"Q2_and_top{k}", {"query": " ".join(toks_sorted[:k])}))
    for k in range(min(5, len(toks_sorted))-1, 1, -1):
        variants.append((f"Q3_and_{k}", {"query": " ".join(toks_sorted[:k])}))
    if toks_sorted:
        top_or = toks_sorted[:min(10, max(6, len(toks_sorted)))]
        variants.append(("Q4_or_top", {"query": " OR ".join(top_or)}))

    grams=[]
    for n in (3,2):
        for i in range(len(toks_sorted)-n+1):
            grams.append(" ".join(toks_sorted[i:i+n]))
    grams = grams[:12]
    if grams:
        variants.append(("Q5_ngram_or", {"query": " OR ".join(grams)}))

    return variants


def quickgo_search(session: requests.Session,
                   label: str,
                   params: Dict[str,str],
                   aspect_filter: Optional[Set[str]],
                   timeout: int,
                   verify_ssl: bool,
                   debug: bool) -> List[Dict]:
    """调用 QuickGO /search，把自然语言查成 GO 术语列表。
    返回字段：id, name, aspect(MF/BP/CC), definition, score(启发式), query_label
    """
    p = {
        "query": params.get("query", ""),
        "rows": "50",
    }
    r = session.get(QUICKGO_SEARCH_URL, params=p, timeout=timeout, verify=verify_ssl)
    if debug:
        sys.stderr.write(f"[GET][QuickGO] {r.request.url} -> {r.status_code}\n"); sys.stderr.flush()
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    results = []
    for item in (data.get("results") or []):
        go_id = item.get("id")
        name  = item.get("name")
        aspect_long = item.get("aspect")
        aspect = ASPECT_INV.get(aspect_long, aspect_long)
        if aspect_filter and aspect not in aspect_filter:
            continue
        definition = None
        try:
            definition = item.get("definition", {}).get("text")
        except Exception:
            definition = None
        # 启发式打分：匹配度 * 长度差惩罚
        score = float(item.get("relevance", 0.0)) if isinstance(item.get("relevance", 0.0), (int,float)) else 0.0
        results.append({
            "id": go_id,
            "name": name,
            "aspect": aspect,
            "definition": definition,
            "score": score,
            "query_label": label
        })
    # 排序：score desc，再按 MF>BP>CC 稍微优先（可根据需要调整）
    aspect_rank = {"MF":0, "BP":1, "CC":2}
    results.sort(key=lambda x: (-(x.get("score") or 0.0), aspect_rank.get(x.get("aspect"), 9)))
    return results

# ---- B. UniProt：用 GO:ID 拉序列 ----
KINGDOM_FILTER = {
    "bacteria": "taxonomy_id:2",
    "archaea": "taxonomy_id:2157",
    "eukaryota": "taxonomy_id:2759",
}

EVIDENCE_WHITELIST = {  # 常见 GO 证据码
    "EXP","IDA","IPI","IMP","IGI","IEP","ISS","ISO","ISA","ISM","IGC","IBA","IBD","IKR","IRD","RCA","TAS","NAS","IC","ND","IEA"
}

def uniprot_by_go(go_ids: List[str],
                  session: requests.Session,
                  size: int,
                  reviewed_only: bool,
                  kingdom: Optional[str],
                  evidence: Optional[List[str]],
                  timeout: int,
                  verify_ssl: bool,
                  debug: bool) -> List[Dict]:
    if not go_ids:
        return []

    all_results = []

    for gid in go_ids:
        gid_clean = gid.split(":")[1] if ":" in gid else gid
        go_terms = [f"go:{gid_clean}"]
        base = "(" + " OR ".join(go_terms) + ")"
        filters = ["fragment:false"]
        if reviewed_only:
            filters.append("reviewed:true")
        if kingdom and kingdom.lower() in KINGDOM_FILTER:
            filters.append(KINGDOM_FILTER[kingdom.lower()])
        if evidence:
            evs = [e for e in (evidence or []) if e in EVIDENCE_WHITELIST]
            if evs:
                filters.append("(go_evidence:" + " OR ".join(evs) + ")")

        query = base + " AND " + " AND ".join(filters)

        params = {
            "query": query,
            "fields": "accession,id,protein_name,organism_name,length,sequence,reviewed",
            "format": "json",
            "size": str(max(1, min(size, 500))),
        }

        try:
            r = session.get(UNIPROT_SEARCH_URL, params=params, timeout=timeout, verify=verify_ssl)
            if debug:
                sys.stderr.write(f"[GET][UniProt] {r.request.url} -> {r.status_code}\n"); sys.stderr.flush()
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception as e:
            if debug:
                sys.stderr.write(f"[ERROR] {gid}: {e}\n"); sys.stderr.flush()
            continue

        for e in data.get("results", []) or []:
            seq = (e.get("sequence") or {}).get("value") if isinstance(e.get("sequence"), dict) else e.get("sequence")
            if not seq:
                continue
            title = None
            try:
                title = (e.get("proteinDescription", {})
                           .get("recommendedName", {})
                           .get("fullName", {})
                           .get("value"))
            except Exception:
                title = None
            title = title or e.get("protein_name") or e.get("id")
            if title and len(title) > 120:
                title = title[:117] + "..."
            aspects = {"MF": [], "BP": [], "CC": []}
            for k, key in [("MF","go(a)"),("BP","go(b)"),("CC","go(c)")]:
                vals = e.get(key)
                if isinstance(vals, list):
                    aspects[k] = [v.get("value") for v in vals if isinstance(v, dict) and v.get("value")]
            all_results.append({
                "accession": e.get("primaryAccession") or e.get("accession"),
                "title": title,
                "organism": (e.get("organism") or {}).get("scientificName") or e.get("organism_name"),
                "length": (e.get("sequence") or {}).get("length") if isinstance(e.get("sequence"), dict) else e.get("length"),
                "sequence": seq,
            })

    return all_results


# ---- 主流程 ----

def main():
    ap = argparse.ArgumentParser(description="GO phrase/sentence → QuickGO → UniProt sequences (no explicit GO: required)")
    ap.add_argument("--text", required=True, help="GO 相关短语或句子（任意句式）")
    ap.add_argument("--size", type=int, default=30, help="返回条数（<=500）")
    ap.add_argument("--timeout", type=int, default=25, help="单请求超时秒数")
    ap.add_argument("--reviewed", action="store_true", help="仅返回 Swiss-Prot（默认包含 TrEMBL）")
    ap.add_argument("--kingdom", choices=["bacteria","archaea","eukaryota"], help="生物域过滤（可选）")
    ap.add_argument("--aspect", action="append", choices=["MF","BP","CC"], help="限定 GO 侧面（可多次给）")
    ap.add_argument("--evidence", action="append", help="GO 证据过滤（如 EXP/IDA/IMP/IEA，可多次给）")
    ap.add_argument("--verify", action="store_true", help="开启严格证书校验（默认关闭以提高稳定性）")
    ap.add_argument("--debug", action="store_true", help="打印请求调试日志到 stderr")
    ap.add_argument("--json", type=str, default=None,
                    help="如果提供路径，则把 accession+sequence 的精简列表写入该文件")
    args = ap.parse_args()

    try:
        session = make_session(verify_ssl=args.verify)

        # 1) QuickGO：自然语言 → GO 候选术语
        qlist = build_go_queries(args.text)
        if args.debug:
            sys.stderr.write(
                "[GO-QUERIES]\n" +
                "\n".join([f"  - {lbl}: {p['query']}" for lbl, p in qlist]) +
                "\n"
            )
            sys.stderr.flush()

        aspect_filter = set(args.aspect) if args.aspect else None

        go_hits: List[Dict] = []
        used_label = None
        for lbl, params in qlist:
            rows = quickgo_search(
                session,
                lbl,
                params,
                aspect_filter,
                timeout=args.timeout,
                verify_ssl=args.verify,
                debug=args.debug
            )
            if rows:
                go_hits = rows
                used_label = lbl
                break

        if not go_hits:
            print(json.dumps({
                "error": "No GO term matched from QuickGO (text-only search).",
                "input_text": args.text,
                "queries_tried": [lbl for lbl, _ in qlist],
                "aspect_filter": list(aspect_filter) if aspect_filter else None
            }, ensure_ascii=False, indent=2))
            sys.exit(1)

        # 2) 取前若干 GO:IDs（混合 MF/BP/CC），去 UniProt 拉序列
        #    策略：优先 MF，然后 BP/CC；总量≤12
        mf = [x for x in go_hits if x.get("aspect") == "MF"]
        bp = [x for x in go_hits if x.get("aspect") == "BP"]
        cc = [x for x in go_hits if x.get("aspect") == "CC"]
        picked = (mf[:6] + bp[:4] + cc[:2]) or go_hits[:12]
        go_ids = [x.get("id") for x in picked if x.get("id")]

        seqs = uniprot_by_go(
            go_ids=go_ids,
            session=session,
            size=max(1, args.size),
            reviewed_only=args.reviewed,
            kingdom=args.kingdom,
            evidence=args.evidence,
            timeout=args.timeout,
            verify_ssl=args.verify,
            debug=args.debug
        )

        if not seqs:
            print(json.dumps({
                "error": "GO matched, but UniProt retrieval by GO returned no sequences.",
                "go_top": picked,
                "go_ids": go_ids
            }, ensure_ascii=False, indent=2))
            sys.exit(2)

        out = {
            "task": "GO-based initial sequence retrieval (QuickGO→UniProt)",
            "input_text": args.text,
            "go_query_used": used_label,
            "go_candidates": picked,
            "count": len(seqs),
            "results": seqs
        }

        # ---- 新增部分：如果传了 --json，就落地 accession+sequence 的精简列表 ----
        if args.json:
            minimal = [
                {
                    "accession": item.get("accession"),
                    "sequence": item.get("sequence"),
                }
                for item in seqs
            ]
            if os.path.exists(args.json) and os.path.getsize(args.json) > 0:
                try:
                    with open(args.json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except Exception:
                    data = []
            else:
                data = []

            data.extend(minimal)

            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if args.debug:
                sys.stderr.write(f"[INFO] Saved {len(minimal)} entries to {args.json}\n")
                sys.stderr.flush()

        # 正常 stdout 输出完整信息
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({
            "error": f"{type(e).__name__}: {e}",
            "input_text": args.text
        }, ensure_ascii=False, indent=2))
        sys.exit(9)


if __name__ == "__main__":
    main()
