#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pathway2constraint_fixed.py

把 pathway/功能文本 → UniProt 结构化检索 → 生成可操作的设计约束(JSON)。
修复点：
- 去掉无效的 `pathway:` 查询字段，仅使用 `cc_pathway`.
- 去掉无效的 `feature` 返回字段，仅保留 ft_* 合法字段。
- 对 cc_pathway 文本做清洗（去标点、取前半句、加通配）以降低 400 的概率。
"""

import re
import sys
import json
import argparse
from typing import List, Dict, Any, Set

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UA_HEADERS = {
    "User-Agent": "pathway2constraint/1.1 (contact@example.com)",
    "Accept": "application/json",
    "Connection": "keep-alive",
}

def make_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
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
    return s

# ---------- 文本解析 ----------
COFACTOR_WORDS = [
    "FAD","FMN","NAD","NADH","NADP","NADPH","SAM",
    "ATP","ADP","GTP","MG","MG2+","MN","MN2+","ZN","ZN2+",
    "HEME","HEM","PLP","TPP","BIOTIN","FE-S","FE2S2","FE4S4","CU","CU2+",
]

def extract_ecs_from_text(text: str) -> List[str]:
    ecs = []
    # EC: 带“EC”
    ecs += re.findall(r'(?:EC[\s:]*)(\d+(?:\.\d+){0,3}(?:\.-?)?)', text, flags=re.IGNORECASE)
    # EC: 纯 “x.x.x.x/-” 模式
    ecs += re.findall(r'(?<!\d)(\d+\.\d+\.\d+(?:\.\d+|-))(?!\d)', text)
    out, seen = [], set()
    for e in ecs:
        e = e.upper().strip().replace("EC ", "")
        if e and e not in seen:
            seen.add(e); out.append(e)
    return out

def extract_cofactors_from_text(text: str) -> List[str]:
    low = text.lower()
    out, seen = [], set()
    for w in COFACTOR_WORDS:
        if w.lower() in low and w not in seen:
            seen.add(w); out.append(w)
    return out

def sanitize_pathway_phrase(text: str) -> str:
    """
    只保留字母数字空格与连字符，截到逗号/分号/句号前，并在末尾加通配 *。
    避免过长/带标点的 cc_pathway 查询引发 400。
    """
    # 先截取第一段（逗号/分号/句号前）
    head = re.split(r'[，,;；。\.]', text, maxsplit=1)[0]
    head = head.strip()
    # 只保留字母数字空格和 - + ()
    head = re.sub(r'[^0-9A-Za-z \-\+\(\)]', ' ', head)
    head = re.sub(r'\s+', ' ', head).strip()
    if len(head) > 0 and not head.endswith('*'):
        head = head + '*'
    return head

# ---------- UniProt 查询 ----------
UNIPROT_FIELDS = ",".join([
    "accession","id","protein_name","organism_name","annotation_score",
    "ec","cc_cofactor","cc_pathway","cc_catalytic_activity",
    "keyword","go_p","go_f",
    "ft_binding","ft_metal","ft_site",
    "xref_interpro"
])

def build_uniprot_query(text: str, ecs: List[str], cofactors: List[str], reviewed: bool) -> str:
    clauses = []

    # cc_pathway：清洗后再查
    p_clause = sanitize_pathway_phrase(text)
    if p_clause:
        clauses.append(f'cc_pathway:"{p_clause}"')

    # EC
    for e in ecs:
        clauses.append(f'ec:{e}')

    # cofactors（保守一些：不要一次性塞太多；对同类词做去重）
    for cf in sorted(set(cofactors)):
        clauses.append(f'cc_cofactor:"{cf}"')

    if not clauses:
        # 兜底：全局关键词回退（注意：可能召回很大）
        qtext = re.sub(r'["]', ' ', text).strip()
        clauses.append(f'({qtext})')

    q = " OR ".join(clauses)
    if reviewed:
        q = f'({q}) AND reviewed:true'
    return q

def uniprot_search(session: requests.Session, query: str, size: int, timeout: int, debug: bool) -> List[dict]:
    params = {
        "query": query,
        "fields": UNIPROT_FIELDS,
        "format": "json",
        "size": str(max(1, min(size, 500))),
    }
    if debug:
        sys.stderr.write(f"[UniProt] query={query}\n")
    r = session.get(UNIPROT_SEARCH, params=params, timeout=timeout)
    if debug:
        sys.stderr.write(f"[GET] {r.request.url}\n")
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])

# ---------- 解析 UniProt 结果 ----------
def collect_ec(entry: dict) -> List[str]:
    ecs = entry.get("ec") or entry.get("ecNumbers") or []
    out = []
    for e in ecs:
        if isinstance(e, str): out.append(e)
        elif isinstance(e, dict):
            v = e.get("value") or e.get("id")
            if v: out.append(v)
    return out

def collect_keywords(entry: dict) -> List[str]:
    out = []
    for k in (entry.get("keywords") or []):
        nm = k.get("name") or k.get("value")
        if nm: out.append(nm)
    return out

def collect_go_terms(entry: dict) -> List[str]:
    out = []
    for fld in ("go_p", "go_f", "goTerms"):
        for g in (entry.get(fld) or []):
            nm = g.get("term") or g.get("name") or g.get("id")
            if nm: out.append(nm)
    return out

def collect_cofactors_from_comments(entry: dict) -> List[str]:
    out = []
    # 扁平 cc_cofactor
    if isinstance(entry.get("cc_cofactor"), list):
        for c in entry["cc_cofactor"]:
            if isinstance(c, str): out.append(c.upper())
            elif isinstance(c, dict):
                v = c.get("value") or c.get("name")
                if v: out.append(v.upper())
    # 嵌套 comments（有的返回里也会带）
    for c in (entry.get("comments") or []):
        if c.get("commentType") == "COFACTOR":
            for co in c.get("cofactors", []):
                nm = (co.get("cofactor") or {}).get("name")
                if nm: out.append(nm.upper())
    return out

def collect_pathways(entry: dict) -> List[str]:
    out = []
    if isinstance(entry.get("cc_pathway"), list):
        for p in entry["cc_pathway"]:
            if isinstance(p, str): out.append(p)
            elif isinstance(p, dict):
                v = p.get("value") or p.get("name")
                if v: out.append(v)
    for c in (entry.get("comments") or []):
        if c.get("commentType") == "PATHWAY":
            txt = c.get("text") or c.get("value") or ""
            if isinstance(txt, str) and txt:
                out.append(txt)
            elif isinstance(txt, list):
                for t in txt:
                    if isinstance(t, str): out.append(t)
                    elif isinstance(t, dict):
                        v = t.get("value") or t.get("text")
                        if v: out.append(v)
    return out

def collect_rhea_ids(entry: dict) -> List[str]:
    out = []
    # cc_catalytic_activity 扁平文本里常出现 RHEA:xxxxx
    if isinstance(entry.get("cc_catalytic_activity"), list):
        for c in entry["cc_catalytic_activity"]:
            s = c if isinstance(c, str) else str(c)
            for m in re.findall(r'RHEA:(\d+)', s.upper()):
                out.append(m)
    for c in (entry.get("comments") or []):
        if c.get("commentType") == "CATALYTIC_ACTIVITY":
            rxn = c.get("reaction") or {}
            rid = rxn.get("rheaId") or rxn.get("rheaID")
            if rid: out.append(str(rid))
            for x in (rxn.get("reactionCrossReferences") or []):
                if x.get("database") == "Rhea":
                    xid = x.get("id") or x.get("identifier")
                    if xid: out.append(str(xid))
    # 去重
    seen, res = set(), []
    for r in out:
        if r not in seen:
            seen.add(r); res.append(r)
    return res

def collect_interpro_entries(entry: dict) -> List[str]:
    out = []
    xi = entry.get("xref_interpro") or entry.get("uniProtKBCrossReferences") or []
    for x in xi:
        if isinstance(x, str):
            if x.startswith(("IPR","PF")): out.append(x)
        elif isinstance(x, dict):
            db = (x.get("database") or x.get("type") or "").lower()
            if db in ("interpro","pfam","smart","prosite"):
                idx = x.get("id") or x.get("identifier")
                if idx: out.append(idx)
    return out

def collect_sites_and_features(entry: dict) -> List[dict]:
    feats = entry.get("features") or []
    out = []
    for ft in feats:
        tp = ft.get("type")
        if tp not in ("BINDING","METAL","SITE"): continue
        loc = ft.get("location") or {}
        begin = (loc.get("start") or {}).get("value")
        end   = (loc.get("end")   or {}).get("value")
        desc  = ft.get("description") or ft.get("featureId") or ""
        lig   = None
        ligands = ft.get("ligand") or ft.get("ligands")
        if isinstance(ligands, dict):
            lig = ligands.get("name")
        elif isinstance(ligands, list) and ligands:
            lig = ligands[0].get("name")
        item = {
            "type": tp,
            "begin": int(begin) if str(begin).isdigit() else begin,
            "end": int(end) if str(end).isdigit() else end,
            "description": desc,
            "ligand": lig
        }
        out.append(item)
    return out

def confidence_score(n_hits: int, swissprot_ratio: float, has_ec: bool, has_rhea: bool, has_cof: bool) -> float:
    base = min(1.0, n_hits / 50.0) * 0.4
    base += min(1.0, swissprot_ratio) * 0.3
    base += (0.1 if has_ec else 0.0)
    base += (0.1 if has_rhea else 0.0)
    base += (0.1 if has_cof else 0.0)
    return round(min(base, 0.99), 3)

def pathway_to_constraints(text: str, topn: int = 50, reviewed: bool = True, timeout: int = 25, debug: bool = False) -> Dict[str, Any]:
    session = make_session()

    ecs_from_text = extract_ecs_from_text(text)
    cof_from_text = extract_cofactors_from_text(text)

    query = build_uniprot_query(text, ecs_from_text, cof_from_text, reviewed=reviewed)
    results = uniprot_search(session, query, size=topn, timeout=timeout, debug=debug)

    ec_set: Set[str] = set(ecs_from_text)
    cofactor_set: Set[str] = set(cof_from_text)
    domain_set: Set[str] = set()
    motif_set: Set[str] = set()  # 本版不强行产出 motif
    go_set: Set[str] = set()
    rhea_set: Set[str] = set()
    pathway_texts: Set[str] = set()
    sites_all: List[dict] = []
    evidence: List[dict] = []

    swissprot_hits = 0
    n_hits = len(results)
    if reviewed:
        swissprot_hits = n_hits  # 查询已限制 reviewed:true

    for ent in results:
        acc = ent.get("primaryAccession") or ent.get("accession") or ent.get("uniProtkbId")
        if not acc:
            continue

        for e in collect_ec(ent): ec_set.add(e)
        for c in collect_cofactors_from_comments(ent): cofactor_set.add(c.upper())
        for p in collect_pathways(ent): pathway_texts.add(p)
        for r in collect_rhea_ids(ent): rhea_set.add(r)
        for g in collect_go_terms(ent): go_set.add(g)
        for kw in collect_keywords(ent): go_set.add(kw)
        for d in collect_interpro_entries(ent): domain_set.add(d)
        feats = collect_sites_and_features(ent)
        if feats:
            sites_all.extend([{"accession": acc, **f} for f in feats])

        evidence.append({"accession": acc, "url": f"https://www.uniprot.org/uniprotkb/{acc}"})

    swiss_ratio = (swissprot_hits / n_hits) if n_hits > 0 else 0.0

    out = {
        "query_text": text,
        "ec_candidates": sorted(ec_set),
        "cofactors": sorted(cofactor_set),
        "domains": sorted(domain_set),
        "motifs": sorted(motif_set),     # 可能为空
        "rhea_ids": sorted(rhea_set),
        "go_terms": sorted(go_set),
        "pathway_comments": sorted(pathway_texts),
        "sites": sites_all,
        "evidence": evidence,
        "meta": {
            "uniprot_hits": n_hits,
            "reviewed_filter": reviewed,
            "confidence": confidence_score(
                n_hits=n_hits,
                swissprot_ratio=swiss_ratio,
                has_ec=len(ec_set) > 0,
                has_rhea=len(rhea_set) > 0,
                has_cof=len(cofactor_set) > 0
            ),
            "query_used": query
        }
    }
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert pathway text to actionable constraints via UniProt (fixed).")
    ap.add_argument("--text", required=True, help="Pathway/Function description.")
    ap.add_argument("--topn", type=int, default=50, help="Max UniProt hits to aggregate (≤500).")
    ap.add_argument("--reviewed", action="store_true", help="Restrict to reviewed (Swiss-Prot).")
    ap.add_argument("--timeout", type=int, default=25, help="HTTP timeout seconds.")
    ap.add_argument("--debug", action="store_true", help="Debug prints.")
    args = ap.parse_args()

    try:
        out = pathway_to_constraints(
            text=args.text,
            topn=args.topn,
            reviewed=args.reviewed,
            timeout=args.timeout,
            debug=args.debug
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(0)
    except requests.HTTPError as e:
        print(json.dumps({"error": f"HTTPError: {e}", "where": "UniProt REST", "text": args.text}, ensure_ascii=False, indent=2))
        sys.exit(2)
    except requests.RequestException as e:
        print(json.dumps({"error": f"RequestException: {e}", "where": "UniProt REST/network", "text": args.text}, ensure_ascii=False, indent=2))
        sys.exit(3)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}", "text": args.text}, ensure_ascii=False, indent=2))
        sys.exit(9)

if __name__ == "__main__":
    main()
