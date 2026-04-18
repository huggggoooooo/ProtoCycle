#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pathway2seq_rhea_free.py
输入一条“pathway 相关的自然语言”，不做任何本地化学词规则/映射；
只用 Rhea 的自由文本检索（多策略查询），命中后用 Rhea/EC 去 UniProt 拉序列。

示例：
  python pathway2seq_rhea_free.py --text "Lipid metabolism, malonyl-CoA biosynthesis pathway" --size 30 --debug
  python pathway2seq_rhea_free.py --text "A protein that catalyzes reactions within the CoA from (R)-pantothenate: step 3/5" --reviewed
"""
import os
import re, sys, json, argparse, time, random, ssl, warnings
from typing import List, Dict, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
RHEA_SEARCH_URL    = "https://www.rhea-db.org/rhea/"

# -------------------- 稳健网络层 --------------------
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

def _jitter(a=0.25, b=0.65):
    time.sleep(a + random.random() * (b - a))

def _safe_get_text(session: requests.Session, url: str, params: dict | None, timeout: int, verify_ssl: bool, debug: bool) -> Optional[str]:
    _jitter()
    r = session.get(url, params=params, timeout=timeout, verify=verify_ssl)
    if debug:
        sys.stderr.write(f"[GET] {r.request.url} -> {r.status_code}\n"); sys.stderr.flush()
    if r.status_code != 200 or not r.text:
        return None
    return r.text

def _safe_get_json(session: requests.Session, url: str, params: dict | None, timeout: int, verify_ssl: bool, debug: bool) -> Optional[dict]:
    _jitter()
    r = session.get(url, params=params, timeout=timeout, verify=verify_ssl)
    if debug:
        sys.stderr.write(f"[GET] {r.request.url} -> {r.status_code}\n"); sys.stderr.flush()
    if r.status_code == 204 or not r.content or not r.text.strip():
        return None
    try:
        return r.json()
    except Exception:
        return None

# -------------------- A. 纯文本 → Rhea 查询生成（无本地化学词规则） --------------------
STOPWORDS = {
    # 极简英文功能词/泛词（不会影响语义的常见噪声）
    "a","an","the","and","or","of","from","to","in","on","for","with","within","via","by","as","that","this",
    "pathway","process","biosynthesis","metabolism","metabolic","biosynthetic",
    "protein","catalyzing","catalyzes","catalyze","reactions","reaction",
    "high","efficiency","specificity"
}

TOKEN_RE = re.compile(r"[A-Za-z0-9\-\(\)\+\./]+")

def tokenize_keep_informative(text: str) -> List[str]:
    """
    不做领域规则，只做一般分词与去停用词，保留带符号/连字符/大小写的 token。
    """
    toks = [t for t in TOKEN_RE.findall(text) if t]
    # 去停用词 & 长度筛选（>=3）& 去全数字
    out = []
    for t in toks:
        tl = t.lower()
        if tl in STOPWORDS:  continue
        if len(tl) < 3:      continue
        if tl.isdigit():     continue
        out.append(t)
    # 去重保序
    seen=set(); uniq=[]
    for t in out:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def build_rhea_queries(text: str, max_queries: int = 20) -> List[Tuple[str,str]]:
    """
    生成一组“由严到松”的 Rhea 查询（仅基于输入文本 & 一般分词），返回 (label, query) 列表。
    - Q1  整句精确短语
    - Q2  按信息量排序取 top-k token 的 AND
    - Q3  逐步放宽 AND（递减 token 数）
    - Q4  OR 扩展（top tokens 的 OR）
    - Q5  n-gram 短语 OR（2~3 gram）
    """
    variants = []
    full = " ".join(text.strip().split())
    if full:
        variants.append(("Q1_full_phrase", f"\"{full}\""))

    toks = tokenize_keep_informative(text)
    # 简单信息量：带连字符/括号/数字/大写的 token 权重更高，其次按长度
    def tok_score(t: str) -> Tuple[int,int]:
        bonus = int(bool(re.search(r"[\-\(\)\+\dA-Z]", t)))
        return (bonus, len(t))
    toks_sorted = sorted(toks, key=tok_score, reverse=True)

    # Q2: top-k AND（k=5,4,3）
    for k in (5,4,3):
        if len(toks_sorted) >= k:
            block = " AND ".join(f"\"{w}\"" for w in toks_sorted[:k])
            variants.append((f"Q2_and_top{k}", block))

    # Q3: 逐步缩小 AND 的词数直到 2
    for k in range(min(5, len(toks_sorted))-1, 1, -1):
        block = " AND ".join(f"\"{w}\"" for w in toks_sorted[:k])
        variants.append((f"Q3_and_{k}", block))

    # Q4: OR 扩展（前 6~10 个有信息量的词）
    if toks_sorted:
        top_or = toks_sorted[:min(10, max(6, len(toks_sorted)))]
        block = " OR ".join(f"\"{w}\"" for w in top_or)
        variants.append(("Q4_or_top", block))

    # Q5: n-gram 短语 OR（2-gram 与 3-gram）
    grams = []
    for n in (3,2):
        for i in range(len(toks_sorted)-n+1):
            grams.append(" ".join(toks_sorted[i:i+n]))
    grams = grams[:12]
    if grams:
        block = " OR ".join(f"\"{g}\"" for g in grams)
        variants.append(("Q5_ngram_or", block))

    # 限制总数，避免过多请求
    return variants[:max_queries]

def rhea_search(session: requests.Session,
                label: str,
                query: str,
                limit: int,
                timeout: int,
                verify_ssl: bool,
                debug: bool) -> List[Dict]:
    """
    调 Rhea TSV 接口：columns=rhea-id,equation,ec,uniprot
    """
    params = {
        "query": query,
        "columns": "rhea-id,equation,ec,uniprot",
        "format": "tsv",
        "limit": str(limit),
    }
    txt = _safe_get_text(session, RHEA_SEARCH_URL, params, timeout, verify_ssl, debug)
    if debug:
        sys.stderr.write(f"[RHEA][{label}] {query}\n"); sys.stderr.flush()
    if not txt:
        return []
    lines = txt.strip().splitlines()
    if len(lines) <= 1:
        return []
    hdr = lines[0].split("\t")
    rows=[]
    for line in lines[1:]:
        cols = line.split("\t")
        row = dict(zip(hdr, cols))
        ecs = [e.strip() for e in row.get("EC number", "").split(";") if e.strip()]
        rid = row.get("Reaction identifier", "").strip()
        rid = rid if rid.startswith("RHEA:") else f"RHEA:{rid}"
        uni_count = row.get("uniprot") or row.get("UniProt") or "0"
        try:
            uni_count = int(uni_count)
        except Exception:
            uni_count = 0
        rows.append({
            "rhea_id": rid,
            "equation": row.get("Equation", ""),
            "ecs": ecs,
            "uniprot_count": uni_count,
            "query_label": label
        })
    # 排序：UniProt 标注数优先；包含更多高信息 token 的查询通常已在前
    rows.sort(key=lambda r: r["uniprot_count"], reverse=True)
    return rows

# -------------------- UniProt：用 Rhea/EC 拉序列 --------------------
KINGDOM_FILTER = {
    "bacteria": "taxonomy_id:2",
    "archaea": "taxonomy_id:2157",
    "eukaryota": "taxonomy_id:2759",
}

def uniprot_by_rhea_or_ec(rhea_ids: List[str],
                          ec_list: List[str],
                          session: requests.Session,
                          size: int,
                          reviewed_only: bool,
                          kingdom: Optional[str],
                          timeout: int,
                          verify_ssl: bool,
                          debug: bool) -> List[Dict]:
    rhea_terms, ec_terms = [], []
    for rid in (rhea_ids or []):
        rid = (rid or "").strip()
        if not rid:
            continue
        num = rid.split(":")[1] if ":" in rid else rid
        if num.isdigit():
            rhea_terms.append(f"rhea:{num}")
    for ec in (ec_list or []):
        ec_clean = (ec or "").upper().replace("EC:", "").replace("EC ", "").strip()
        if ec_clean and all(p.isdigit() for p in ec_clean.split(".")):
            ec_terms.append(f"ec:{ec_clean}")
    if not rhea_terms and not ec_terms:
        return []

    term_groups = []
    if rhea_terms: term_groups.append("(" + " OR ".join(rhea_terms) + ")")
    if ec_terms:   term_groups.append("(" + " OR ".join(ec_terms) + ")")
    base = " OR ".join(term_groups) if len(term_groups) > 1 else term_groups[0]

    filters = ["fragment:false"]
    if reviewed_only:
        filters.append("reviewed:true")
    if kingdom and kingdom.lower() in KINGDOM_FILTER:
        filters.append(KINGDOM_FILTER[kingdom.lower()])
    query = f"{base} AND {' AND '.join(filters)}"

    params = {
        "query": query,
        "fields": "accession,id,protein_name,organism_name,length,sequence,rhea,cc_catalytic_activity,reviewed",
        "format": "json",
        "size": str(max(1, min(size, 500))),
    }
    data = _safe_get_json(session, UNIPROT_SEARCH_URL, params, timeout, verify_ssl, debug)
    if not data or not isinstance(data, dict):
        return []
    out = []
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
        out.append({
            "accession": e.get("primaryAccession") or e.get("accession"),
            "title": title,
            "organism": (e.get("organism") or {}).get("scientificName") or e.get("organism_name"),
            "length": (e.get("sequence") or {}).get("length") if isinstance(e.get("sequence"), dict) else e.get("length"),
            "sequence": seq,
            "match_fields": (["rhea"] if rhea_terms else []) + (["ec"] if ec_terms else [])
        })
    return out

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser(description="Pathway text → Rhea → UniProt sequences (no local chem rules/maps).")
    ap.add_argument("--text", required=True, help="pathway 相关自然语言（任意句式）")
    ap.add_argument("--size", type=int, default=20, help="返回条数（<=500）")
    ap.add_argument("--timeout", type=int, default=25, help="单请求超时秒数")
    ap.add_argument("--reviewed", action="store_true", help="仅返回 Swiss-Prot（默认包含 TrEMBL）")
    ap.add_argument("--kingdom", choices=["bacteria", "archaea", "eukaryota"], help="生物域过滤（可选）")
    ap.add_argument("--verify", action="store_true", help="开启严格证书校验（默认关闭以提高稳定性）")
    ap.add_argument("--debug", action="store_true", help="打印请求调试日志到 stderr")
    ap.add_argument(
        "--json",
        type=str,
        default=None,
        help="如果提供路径，则把 accession+sequence 的精简列表写入该文件"
    )

    args = ap.parse_args()

    try:
        session = make_session(verify_ssl=args.verify)

        # 1) 生成 Rhea 候选查询（纯文本策略；不抽化学词、不用本地映射）
        qlist = build_rhea_queries(args.text, max_queries=18)
        if args.debug:
            sys.stderr.write(
                "[RHEA-QUERIES]\n" +
                "\n".join([f"  - {lbl}: {q}" for lbl, q in qlist]) +
                "\n"
            )
            sys.stderr.flush()

        rhea_rows_all = []
        used_label = None
        for lbl, q in qlist:
            rows = rhea_search(
                session,
                lbl,
                q,
                limit=50,
                timeout=args.timeout,
                verify_ssl=args.verify,
                debug=args.debug
            )
            if rows:
                rhea_rows_all = rows
                used_label = lbl
                break

        if not rhea_rows_all:
            print(json.dumps({
                "error": "No matching Rhea reaction found from the input text (text-only queries).",
                "input_text": args.text,
                "queries_tried": [lbl for lbl, _ in qlist]
            }, ensure_ascii=False, indent=2))
            sys.exit(1)

        # 2) 聚合 Rhea/EC，拉 UniProt 序列
        rhea_ids = list({row["rhea_id"] for row in rhea_rows_all[:10]})
        ec_list  = list({ec for row in rhea_rows_all for ec in (row.get("ecs") or [])})

        seqs = uniprot_by_rhea_or_ec(
            rhea_ids=rhea_ids,
            ec_list=ec_list,
            session=session,
            size=max(1, args.size),
            reviewed_only=args.reviewed,
            kingdom=args.kingdom,
            timeout=args.timeout,
            verify_ssl=args.verify,
            debug=args.debug
        )

        if not seqs:
            print(json.dumps({
                "error": "Rhea matched, but UniProt retrieval by Rhea/EC returned no sequences.",
                "rhea_top": rhea_rows_all[:5],
                "rhea_ids": rhea_ids,
                "ecs": ec_list
            }, ensure_ascii=False, indent=2))
            sys.exit(2)

        out = {
            "task": "pathway initial sequence retrieval (Rhea text-only search)",
            "input_text": args.text,
            "rhea_query_used": used_label,
            "rhea_candidates": rhea_rows_all[:5],
            "count": len(seqs),
            "results": seqs
        }

        # ---- 新增部分：如果传了 --json，就把 accession+sequence 精简列表写入文件 ----
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

        # 正常 stdout 输出完整版
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
