#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dna_binding2seq.py
用 DNA-binding 关键词在 UniProt 检索初始序列（支持多关键词、reviewed 过滤、健壮 JSON 解析）

示例：
  python dna_binding2seq.py --keywords "HMG box, DNA-bending" --size 20 --reviewed --debug
  python dna_binding2seq.py --keywords "homeobox" --size 10
"""
import os
import sys, json, time, random, argparse, ssl, warnings
from typing import List, Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UA_HEADERS = {
    "User-Agent": "desc2seq-agent (contact@example.com)",
    "Accept": "application/json",
    "Connection": "keep-alive",
}

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def make_session(verify_ssl: bool, retries: int = 4, backoff: float = 0.8) -> requests.Session:
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

def jitter(a=0.25, b=0.65):
    time.sleep(a + random.random() * (b - a))

def safe_get_json(session: requests.Session, url: str, params: dict, timeout: int, verify_ssl: bool, debug: bool):
    """更稳的 JSON 解析：对 204/空响应/HTML 做兜底"""
    jitter()
    r = session.get(url, params=params, timeout=timeout, verify=verify_ssl)
    if debug:
        sys.stderr.write(f"[GET] {r.request.url}  -> {r.status_code}\n")
        sys.stderr.flush()
    # 204/空
    if r.status_code == 204 or not r.content or not r.text.strip():
        return None
    # 非 JSON 兜底
    txt = r.text.strip()
    if not txt.startswith("{") and not txt.startswith("["):
        if debug:
            sys.stderr.write(f"[WARN] Non-JSON response preview: {txt[:160]}\n")
            sys.stderr.flush()
        return None
    try:
        return r.json()
    except Exception as e:
        if debug:
            sys.stderr.write(f"[WARN] JSON parse failed: {e}\n")
            sys.stderr.flush()
        return None

def build_uniprot_query(keywords: List[str], reviewed_only: bool) -> str:
    """
    基础条件锁定 DNA-binding：
      - UniProt keyword: "DNA-binding"
      - GO:0003677
    再叠加用户关键词（OR）增强召回，再用 AND 收紧。
    """
    # 基础 DNA-binding 条件
    dna_core = '(keyword:"DNA-binding" OR go:0003677)'
    # 用户关键词（可出现在蛋白名/域/注释等）
    kws = [k.strip() for k in keywords if k and k.strip()]
    if kws:
        # 对每个词加引号，允许短语；用 OR 聚合
        or_block = "(" + " OR ".join([f'"{k}"' for k in kws]) + ")"
        q = f"{dna_core} AND {or_block}"
    else:
        q = dna_core
    if reviewed_only:
        q = f"({q}) AND reviewed:true"
    return q

def uniprot_search_sequences(session: requests.Session,
                             query: str,
                             size: int,
                             timeout: int,
                             verify_ssl: bool,
                             debug: bool) -> List[Dict]:
    size = max(1, min(size, 500))
    params = {
        "query": query,
        "fields": ",".join([
            "accession",
            "id",
            "protein_name",
            "organism_name",
            "length",
            "reviewed",
            "cc_function",
            "go_f",
            "sequence"
        ]),
        "format": "json",
        "size": str(size),
    }
    data = safe_get_json(session, UNIPROT_SEARCH, params, timeout, verify_ssl, debug)
    if not data or not isinstance(data, dict):
        return []
    out = []
    for e in data.get("results", []):
        seq, seqlen = None, None
        seq_field = e.get("sequence")
        if isinstance(seq_field, dict):
            seq = seq_field.get("value"); seqlen = seq_field.get("length")
        elif isinstance(seq_field, str):
            seq = seq_field; seqlen = e.get("length") or (len(seq) if seq else None)
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
        org = (e.get("organism") or {}).get("scientificName") or e.get("organism_name")
        acc = e.get("primaryAccession") or e.get("accession")

        func_snips = []
        for c in (e.get("comments") or []):
            ctype = c.get("commentType") or c.get("type")
            if str(ctype).upper() == "FUNCTION":
                texts = c.get("texts") or c.get("text") or []
                if isinstance(texts, list):
                    for t in texts:
                        val = (t.get("value") if isinstance(t, dict) else str(t)).strip()
                        if val: func_snips.append(val)
                else:
                    val = (texts.get("value") if isinstance(texts, dict) else str(texts)).strip()
                    if val: func_snips.append(val)

        out.append({
            "accession": acc,              # ✅ 新增 accession 字段
            "id": e.get("uniProtkbId") or e.get("id"),
            "title": title,
            "organism": org,
            "length": seqlen or e.get("length"),
            "sequence": seq,
            "function_notes": func_snips[:3] or None
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Search UniProt for DNA-binding proteins by keywords and return sequences.")
    ap.add_argument("--keywords", type=str, default="", help="逗号分隔的关键词，如: 'HMG box, homeobox, helix-turn-helix'")
    ap.add_argument("--size", type=int, default=20, help="返回序列条数（<=500）")
    ap.add_argument("--timeout", type=int, default=20, help="单请求超时（秒）")
    ap.add_argument("--reviewed", action="store_true", help="仅返回 Swiss-Prot")
    ap.add_argument("--verify", action="store_true", help="开启证书校验（默认关闭以提高稳定性）")
    ap.add_argument("--debug", action="store_true", help="打印调试日志到 stderr")
    ap.add_argument("--json", type=str, default=None, help="输出结果保存到指定 JSON 文件（仅包含 accession 与 sequence）")
    args = ap.parse_args()

    try:
        session = make_session(verify_ssl=args.verify)
        kws = [k.strip() for k in args.keywords.split(",")] if args.keywords else []
        query = build_uniprot_query(kws, reviewed_only=args.reviewed)
        if args.debug:
            sys.stderr.write(f"[QUERY] {query}\n"); sys.stderr.flush()

        items = uniprot_search_sequences(
            session=session,
            query=query,
            size=args.size,
            timeout=args.timeout,
            verify_ssl=args.verify,
            debug=args.debug
        )

        if not items:
            print(json.dumps({
                "error": "No sequences found for the given DNA-binding keywords.",
                "keywords": kws,
                "query": query
            }, ensure_ascii=False, indent=2))
            sys.exit(1)

        out = {
            "task": "dna-binding initial sequence retrieval",
            "keywords": kws,
            "query": query,
            "count": len(items),
            "results": items
        }

        # ---- 新增：保存 accession + sequence 到 JSON 文件 ----
        if args.json:
            minimal = [{"accession": it.get("accession"), "sequence": it.get("sequence")} for it in items]
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
                sys.stderr.write(f"[INFO] Saved {len(minimal)} entries to {args.json}\n"); sys.stderr.flush()

        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False, indent=2))
        sys.exit(9)


if __name__ == "__main__":
    main()
