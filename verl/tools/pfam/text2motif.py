#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文本/关键词 -> PROSITE（经 InterPro API）-> PS号列表 -> get_prosite_entry 取 DE/PA
                -> （可选）ScanProsite 拉 subseq 命中

依赖：
  - biopython（Bio.ExPASy）
  - 可选：beautifulsoup4（本脚本不必需）
用法示例：
  python text_or_keywords2prosite_interpro.py --text "Bipartite nuclear localization signal, AHA motif" --topk 8 --scan-examples 3 --debug
"""

import ssl
import sys
import re
import json
import gzip
import argparse
import urllib.parse
import urllib.request
from io import BytesIO
from typing import List, Dict, Optional

from Bio.ExPASy import get_prosite_entry, ScanProsite

# ---------- 基本设置 ----------
INTERPRO_SEARCH = "https://www.ebi.ac.uk/interpro/api/entry/prosite/search"
PROSITE_ENTRY_PREFIX = "https://prosite.expasy.org/"

def make_ssl_unverified_opener():
    ctx = ssl._create_unverified_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    opener.addheaders = [
        ("User-Agent", "desc2seq-agent (contact@example.com)"),
        ("Accept", "application/json"),
        ("Accept-Encoding", "gzip, deflate"),
        ("Connection", "keep-alive"),
    ]
    urllib.request.install_opener(opener)
    return opener

def http_get_text(url: str, timeout: int = 15) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
            # 处理 gzip
            if r.headers.get("Content-Encoding", "").lower() == "gzip":
                data = gzip.GzipFile(fileobj=BytesIO(data)).read()
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return data.decode("latin-1", errors="ignore")
    except Exception:
        return None

# ---------- 工具 ----------
def split_keywords(text: str) -> List[str]:
    parts = re.split(r"[,\uFF0C;\uFF1B]+", text)
    kws = [p.strip() for p in parts if p and p.strip()]
    if text.strip():
        kws.append(text.strip())
    out, seen = [], set()
    for k in kws:
        kl = k.lower()
        if kl not in seen:
            out.append(k); seen.add(kl)
    return out

# ---------- 关键：用 InterPro API 查 PROSITE ----------
def interpro_search_prosite(query: str, page_size: int = 25, timeout: int = 15, debug: bool = False) -> List[Dict]:
    """
    调 InterPro：/interpro/api/entry/prosite/search?query=...
    返回含 accession(PSxxxxx)、name/short_name 等的条目列表
    """
    params = {
        "query": query,
        "page_size": str(page_size),
    }
    url = INTERPRO_SEARCH + "?" + urllib.parse.urlencode(params)
    txt = http_get_text(url, timeout=timeout)
    if debug:
        sys.stderr.write(f"[InterPro] {url} -> {'OK' if txt else 'FAIL'}\n"); sys.stderr.flush()
    if not txt:
        return []
    try:
        data = json.loads(txt)
    except Exception:
        return []
    results = []
    for item in data.get("results", []):
        md = item.get("metadata", {})
        acc = md.get("accession") or md.get("accession_id") or md.get("ac") or ""
        if not acc or not re.fullmatch(r"PS\d{5}", acc):
            continue
        name = md.get("name") or md.get("title") or ""
        short = md.get("short_name") or ""
        results.append({"accession": acc, "name": name, "short_name": short})
    return results

# ---------- 取 PROSITE 平面文本里的 DE/PA ----------
def parse_prosite_entry_fields(signature_ac: str, debug: bool = False) -> Optional[Dict]:
    try:
        h = get_prosite_entry(signature_ac)
        raw = h.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        if debug:
            sys.stderr.write(f"[get_prosite_entry] {signature_ac} error: {e}\n"); sys.stderr.flush()
        return None

    ac = None
    de = []
    pa_lines = []
    for line in raw.splitlines():
        line = line.rstrip("\n")
        if line.startswith("AC"):
            m = re.search(r"(PS\d{5})", line)
            if m: ac = m.group(1)
        elif line.startswith("DE"):
            de.append(line[5:].strip())
        elif line.startswith("PA"):
            pa_lines.append(line[5:].strip())

    desc = " ".join(de).strip() if de else None
    pattern = None
    if pa_lines:
        joined = " ".join(pa_lines).strip()
        if joined.endswith("."):
            joined = joined[:-1].strip()
        pattern = joined or None

    if not (ac or pattern or desc):
        return None
    return {"accession": ac, "description": desc, "pattern": pattern}

# ---------- （可选）ScanProsite 拉 subseq 命中 ----------
def scan_examples_by_ps(signature_ac: str, max_hits: int = 3, debug: bool = False) -> List[Dict]:
    try:
        handle = ScanProsite.scan(sig=signature_ac, output="xml", db="swissprot")
        results = ScanProsite.read(handle)
    except Exception as e:
        if debug:
            sys.stderr.write(f"[PSScan] {signature_ac} error: {e}\n"); sys.stderr.flush()
        return []
    hits = []
    for r in results[:max_hits]:
        acc = r.get("sequence_ac")
        s = int(r.get("start", 0))
        e = int(r.get("stop", 0))
        frag = r.get("sequence_fragment") or None
        hits.append({"uniprot_ac": acc, "start": s, "end": e, "subsequence": frag})
    return hits

# ---------- 主逻辑 ----------
def find_motifs_by_text_via_interpro(text: str, topk: int = 10, scan_examples: int = 0, timeout: int = 15, debug: bool = False) -> Dict:
    make_ssl_unverified_opener()
    queries = split_keywords(text)

    # 1) 聚合 InterPro 搜到的 PROSITE 条目
    ps_list = []
    seen = set()
    for q in queries:
        items = interpro_search_prosite(q, page_size=topk, timeout=timeout, debug=debug)
        for it in items:
            acc = it["accession"]
            if acc not in seen:
                ps_list.append(it)
                seen.add(acc)
        if len(ps_list) >= topk:
            break
    ps_list = ps_list[:topk]

    # 2) 逐个 PS 号取 DE/PA，并可选扫描示例
    out = []
    for it in ps_list:
        acc = it["accession"]
        entry = parse_prosite_entry_fields(acc, debug=debug)
        if not entry:
            continue
        rec = {**it, **entry, "url": PROSITE_ENTRY_PREFIX + acc}
        if scan_examples > 0:
            rec["examples"] = scan_examples_by_ps(acc, max_hits=scan_examples, debug=debug)
        out.append(rec)

    return {"input": text, "queries": queries, "count": len(out), "candidates": out}

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Text/keywords -> PROSITE via InterPro API -> PS entries -> DE/PA (+optional ScanProsite examples)")
    ap.add_argument("--text", required=True, help="自然语言或关键词（逗号分隔也行）")
    ap.add_argument("--topk", type=int, default=10, help="最多返回多少个 PROSITE PS 号")
    ap.add_argument("--scan-examples", type=int, default=0, help="每个 PS 号返回多少个 ScanProsite 命中（0 关闭）")
    ap.add_argument("--timeout", type=int, default=15, help="单请求超时（秒）")
    ap.add_argument("--debug", action="store_true", help="调试日志")
    args = ap.parse_args()

    res = find_motifs_by_text_via_interpro(
        text=args.text,
        topk=args.topk,
        scan_examples=args.scan_examples,
        timeout=args.timeout,
        debug=args.debug
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
