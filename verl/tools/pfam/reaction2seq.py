#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reaction_text -> Rhea IDs/EC -> UniProt sequences (JSON for LLM)

用法:
  python reaction2seq.py --reaction "H2O + L-kynurenine = anthranilate + H(+) + L-alanine" --size 10 --unreviewed
  # 可选：--verify 开启严格证书校验，--debug 打印 HTTP 调试日志
"""
import os
import argparse, sys, json, re, time, random, ssl, warnings
from typing import List, Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

RHEA_SEARCH_URL = "https://www.rhea-db.org/rhea/"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# ---------- 稳健网络层：Session / Retry / Backoff / Headers / JSON安全解析 ----------

UA_HEADERS = {
    "User-Agent": "desc2seq-agent (contact@example.com)",  # TODO: 换成你的邮箱/标识
    "Accept": "*/*",
    "Connection": "keep-alive",
}

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def make_session(verify_ssl: bool = False, retries: int = 4, backoff: float = 0.8) -> requests.Session:
    """
    更稳的 HTTPS Session：自动重试、指数退避、Keep-Alive、可选关闭证书校验
    """
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
        # 关闭证书校验以规避校园网/代理的握手异常；要严格安全可传 True
        ssl._create_default_https_context = ssl._create_unverified_context
    return s

def _jitter(a=0.25, b=0.65):
    """请求间随机抖动，降低 429 / 链路拥塞概率"""
    time.sleep(a + random.random() * (b - a))

def _safe_get_json(session: requests.Session, url: str, params: dict | None, timeout: int, verify_ssl: bool, debug: bool):
    """
    稳健的 JSON 请求：容错空响应/非 JSON，打印可选调试日志
    """
    _jitter()
    r = session.get(url, params=params, timeout=timeout, verify=verify_ssl)
    if debug:
        sys.stderr.write(f"[GET] {r.request.url} -> {r.status_code}\n"); sys.stderr.flush()
    if r.status_code == 204 or not r.content or not r.text.strip():
        return None
    txt = r.text.strip()
    if not txt.startswith("{") and not txt.startswith("["):
        if debug:
            sys.stderr.write(f"[WARN] Non-JSON response preview: {txt[:160]}\n"); sys.stderr.flush()
        return None
    try:
        return r.json()
    except Exception as e:
        if debug:
            sys.stderr.write(f"[WARN] JSON parse failed: {e}\n"); sys.stderr.flush()
        return None

# ---------- Rhea 搜索：由反应文本抽词 -> 调 Rhea TSV 接口 ----------

def rhea_search_from_reaction_text(reaction_text: str,
                                   session: requests.Session,
                                   limit: int = 50,
                                   timeout: int = 30,
                                   verify_ssl: bool = False,
                                   debug: bool = False):
    """
    用参与物名字做查询：把反应按 = 与 + 切分，抽取化学名词，合取检索。
    返回: [{"rhea_id": "RHEA:16813", "equation": "...", "ecs": ["EC 3.7.1.3"], "uniprot_count": 15633}, ...]
    """
    parts = re.split(r"[=+]", reaction_text)
    tokens = [t.strip() for t in parts if t.strip()]
    query = " AND ".join(f'"{t}"' for t in tokens)

    params = {
        "query": query,
        "columns": "rhea-id,equation,ec,uniprot",
        "format": "tsv",
        "limit": str(limit),
    }
    _jitter()
    r = session.get(RHEA_SEARCH_URL, params=params, timeout=timeout, verify=verify_ssl)
    if debug:
        sys.stderr.write(f"[GET] {r.request.url} -> {r.status_code}\n"); sys.stderr.flush()
    if r.status_code != 200 or not r.text:
        return []

    lines = r.text.strip().splitlines()
    if len(lines) <= 1:
        return []

    hdr = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cols = line.split("\t")
        row = dict(zip(hdr, cols))
        ecs = [e.strip() for e in row.get("EC number", "").split(";") if e.strip()]
        # Rhea 主键整理
        rid = row.get("Reaction identifier", "").strip()
        rid = rid if rid.startswith("RHEA:") else f"RHEA:{rid}"
        # UniProt计数字段名有时为 'uniprot' 或 'UniProt'，兼容处理
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
        })

    # 排序：优先“方程字符串包含所有 token” + “UniProt 标注数高”
    def score(rw):
        eq = (rw["equation"] or "").lower()
        hit_all = all(t.lower() in eq for t in tokens)
        return (1 if hit_all else 0, rw["uniprot_count"])

    rows.sort(key=score, reverse=True)
    if debug:
        sys.stderr.write("[DEBUG] Rhea candidates (top 5 shown):\n")
        for rr in rows[:5]:
            sys.stderr.write(f"  - {rr['rhea_id']}  uni={rr['uniprot_count']}  eq={rr['equation']}\n")
        sys.stderr.flush()
    return rows

# ---------- UniProt 搜索：由 Rhea / EC 抽词 -> 拉取序列 JSON ----------

def _clean_ec(ec: str) -> Optional[str]:
    ec_clean = (ec or "").upper().replace("EC:", "").replace("EC ", "").strip()
    if not ec_clean:
        return None
    parts = ec_clean.split(".")
    if not all(p.isdigit() for p in parts):
        return None
    return ".".join(parts)

def uniprot_by_rhea_or_ec(rhea_ids: List[str],
                          ec_list: List[str],
                          session: requests.Session,
                          size: int = 10,
                          reviewed_only: bool = True,
                          timeout: int = 30,
                          verify_ssl: bool = False,
                          debug: bool = False) -> List[Dict]:
    """
    通过 Rhea 或 EC 搜索 UniProt，并直接拿到序列，返回精简结果供 LLM 用。
    """
    # 组装 Rhea 术语
    rhea_terms = []
    for rid in (rhea_ids or []):
        rid = (rid or "").strip()
        if not rid:
            continue
        num = rid.split(":")[1] if ":" in rid else rid
        if num.isdigit():
            rhea_terms.append(f"rhea:{num}")

    # 组装 EC 术语
    ec_terms = []
    for ec in (ec_list or []):
        ec_fixed = _clean_ec(ec)
        if ec_fixed:
            ec_terms.append(f"ec:{ec_fixed}")

    if not rhea_terms and not ec_terms:
        return []

    term_groups = []
    if rhea_terms:
        term_groups.append("(" + " OR ".join(rhea_terms) + ")")
    if ec_terms:
        term_groups.append("(" + " OR ".join(ec_terms) + ")")
    base = " OR ".join(term_groups) if len(term_groups) > 1 else term_groups[0]

    filters = ["fragment:false"]
    if reviewed_only:
        filters.append("reviewed:true")
    query = f"{base} AND {' AND '.join(filters)}"

    params = {
        "query": query,
        "fields": "accession,id,protein_name,organism_name,length,sequence,rhea,cc_catalytic_activity,reviewed",
        "format": "json",
        "size": str(max(1, min(size, 500))),
    }

    if debug:
        sys.stderr.write(f"[QUERY UniProt] {query}\n"); sys.stderr.flush()

    data = _safe_get_json(session, UNIPROT_SEARCH_URL, params, timeout, verify_ssl, debug)
    if not data or not isinstance(data, dict):
        return []

    out = []
    for e in data.get("results", []) or []:
        # 取序列
        seq, seqlen = None, None
        seq_field = e.get("sequence")
        if isinstance(seq_field, dict):
            seq = seq_field.get("value")
            seqlen = seq_field.get("length")
        elif isinstance(seq_field, str):
            seq = seq_field
            seqlen = e.get("length")
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
            "length": seqlen or e.get("length"),
            "sequence": seq,
            "match_fields": ["cc_catalytic_activity/rhea"] if rhea_terms else ["ec"]
        })
    return out

# ---------- 主程序 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaction", required=True, help="反应方程文本，如: H2O + L-kynurenine = anthranilate + H(+) + L-alanine")
    ap.add_argument("--size", type=int, default=10, help="返回序列条数（默认10，<=500）")
    ap.add_argument("--timeout", type=int, default=30, help="请求超时（秒）")
    ap.add_argument("--unreviewed", action="store_true", help="包含 TrEMBL（默认只取 Swiss-Prot）")
    ap.add_argument("--verify", action="store_true", help="开启严格证书校验（默认关闭以提高稳定性）")
    ap.add_argument("--debug", action="store_true", help="打印请求调试日志到 stderr")
    ap.add_argument("--json", type=str, default=None,
                    help="如果提供路径，则保存 accession+sequence 的精简列表")
    args = ap.parse_args()

    try:
        session = make_session(verify_ssl=args.verify)

        # 1) Rhea：由反应文本检索候选
        rhea_rows = rhea_search_from_reaction_text(
            args.reaction,
            session=session,
            limit=50,
            timeout=args.timeout,
            verify_ssl=args.verify,
            debug=args.debug
        )
        if not rhea_rows:
            print(json.dumps({
                "error": "No matching Rhea reaction found from reaction text.",
                "reaction": args.reaction
            }, ensure_ascii=False, indent=2))
            sys.exit(1)

        # 2) 聚合 Rhea IDs 与 ECs
        rhea_ids = list({row["rhea_id"] for row in rhea_rows[:10]})
        ec_set = set()
        for row in rhea_rows:
            for ec in row.get("ecs", []) or []:
                ec_set.add(ec)
        ec_list = list(ec_set)

        # 3) UniProt：由 Rhea/EC 拉取序列
        seqs = uniprot_by_rhea_or_ec(
            rhea_ids=rhea_ids,
            ec_list=ec_list,
            session=session,
            size=max(1, args.size),
            reviewed_only=not args.unreviewed,
            timeout=args.timeout,
            verify_ssl=args.verify,
            debug=args.debug
        )
        if not seqs:
            print(json.dumps({
                "error": "No UniProt sequences found from Rhea/EC mapping.",
                "reaction": args.reaction,
                "rhea_ids": rhea_ids,
                "ecs": ec_list
            }, ensure_ascii=False, indent=2))
            sys.exit(2)

        out = {
            "reaction": args.reaction,
            "rhea_candidates": rhea_rows[:5],
            "count": len(seqs),
            "results": seqs
        }

        # ---- 新增部分：--json 输出精简结果 ----
        if args.json:
            minimal = [
                {"accession": s.get("accession"), "sequence": s.get("sequence")}
                for s in seqs
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

        # 正常输出完整版
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({
            "error": f"{type(e).__name__}: {e}",
            "reaction": args.reaction
        }, ensure_ascii=False, indent=2))
        sys.exit(3)


if __name__ == "__main__":
    main()
