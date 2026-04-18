#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Desc2Seq: 从功能描述查询 UniProtKB 并返回包含蛋白序列 + 关键位点信息 的 JSON。

新增：
- 对每个 accession 追加一次 /uniprotkb/{accession}.json 请求，
  解析 features 中的关键位点/区域；没有则置为 null。
"""
import os
import sys
import json
import argparse
import requests
import ssl, time, random, warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

UA_HEADERS = {
    "User-Agent": "desc2seq-agent",
    "Accept": "application/json",
    "Connection": "keep-alive",
}

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def make_session(verify_ssl: bool = False, retries: int = 4, backoff: float = 0.8) -> requests.Session:
    """更稳的 HTTPS Session：重试、退避、Keep-Alive、可选关闭证书校验"""
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
        # 关闭证书校验以规避校园网/代理的握手异常
        ssl._create_default_https_context = ssl._create_unverified_context
    return s

def _jitter(a=0.25, b=0.65):
    """给请求之间加一点随机抖动，降低 429 风险"""
    time.sleep(a + random.random() * (b - a))

def _safe_get_json(session: requests.Session, url: str, params: dict | None, timeout: int, verify_ssl: bool, debug: bool):
    """稳健的 JSON 请求与解析：容错空响应/非 JSON"""
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

from urllib.parse import quote_plus

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"

FIELDS = "accession,id,protein_name,gene_names,organism_name,length,sequence"

# ---- 你原来的函数 ----
def build_uniprot_query(text_query: str, organism: str | None, reviewed_only: bool) -> str:
    parts = [f'("{text_query}") OR {text_query}']
    if organism:
        parts.append(f'organism_name:"{organism}"')
    if reviewed_only:
        parts.append("reviewed:true")
    return " AND ".join(parts)

def fetch_uniprot_results(query: str,
                          size: int,
                          timeout: int = 30,
                          session: requests.Session | None = None,
                          verify_ssl: bool = False,
                          debug: bool = False):
    """
    通过 /uniprotkb/search 拉取结果（分页处理），带会话重试与证书开关
    """
    BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
    FIELDS = "accession,id,protein_name,gene_names,organism_name,length,sequence"

    params = {
        "query": query,
        "fields": FIELDS,
        "size": min(size, 500),
        "format": "json",
    }
    results = []
    url = BASE_URL
    s = session or make_session(verify_ssl=verify_ssl)

    while True:
        data = _safe_get_json(s, url, params if url == BASE_URL else None, timeout, verify_ssl, debug)
        if not data:
            # 返回已有结果，避免抛异常中断全流程
            return results

        page_results = data.get("results", []) or []
        results.extend(page_results)
        if len(results) >= size:
            return results[:size]

        next_link = None
        links = data.get("links", {}) or {}
        if isinstance(links, dict):
            next_link = links.get("next")

        if not next_link:
            link_header = s.get(url, params=params if url == BASE_URL else None, timeout=timeout, verify=verify_ssl).headers.get("Link", "")
            for seg in link_header.split(","):
                if 'rel="next"' in seg:
                    next_link = seg.split(";")[0].strip().strip("<>").strip()
                    break

        if not next_link:
            return results

        url = next_link
        params = None  # 后续页不再传 params


def normalize_entry(e: dict) -> dict | None:
    seq = None
    if isinstance(e.get("sequence"), dict):
        seq = e["sequence"].get("value")
        seqlen = e["sequence"].get("length")
    elif isinstance(e.get("sequence"), str):
        seq = e["sequence"]
        seqlen = e.get("length")
    else:
        seqlen = None

    if not seq:
        return None

    return {
        "accession": e.get("primaryAccession") or e.get("accession"),
        "protein_name": (
            (e.get("proteinDescription", {}) or {}).get("recommendedName", {}) or {}
        ).get("fullName", {}).get("value")
        or e.get("protein_name"),
        "organism_name": (
            (e.get("organism") or {}).get("scientificName") or e.get("organism_name")
        ),
        "length": seqlen,
        "sequence": seq,
    }

# ---- 新增：按 accession 拉取 features 并抽取关键位点 ----
FEATURE_WHITELIST = {
    # UniProt JSON 中常见的 feature.type / featureType 取值（大小写以实际返回为准）
    "BINDING", "BINDING_SITE", "METAL", "METAL_BINDING", "ACT_SITE", "ACTIVE_SITE",
    "SITE", "MOTIF", "DOMAIN", "REGION"
}

def _pos_from_loc(loc: dict) -> tuple[int | None, int | None]:
    """从 location 结构提取起止坐标（1-based, inclusive）"""
    if not isinstance(loc, dict):
        return None, None
    start = loc.get("start", {}).get("value")
    end = loc.get("end", {}).get("value")
    try:
        return (int(start) if start is not None else None,
                int(end) if end is not None else None)
    except Exception:
        return None, None

def _desc_from_feature(f: dict) -> str | None:
    """拼接简要描述（type/ligand/description 注释）"""
    notes = []
    t = f.get("type") or f.get("featureType")
    if t: notes.append(str(t))
    lig = (f.get("ligand") or {}).get("name") if isinstance(f.get("ligand"), dict) else None
    if lig: notes.append(f"ligand={lig}")
    desc = f.get("description")
    if desc: notes.append(desc)
    if not notes: return None
    return "; ".join(notes)

def fetch_key_features_for_accession(acc: str,
                                     session: requests.Session,
                                     timeout: int = 30,
                                     verify_ssl: bool = False,
                                     debug: bool = False) -> dict:
    """
    通过 /uniprotkb/{acc}.json 拉取 features，带重试/抖动/证书开关
    """
    ENTRY_URL = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    FEATURE_WHITELIST = {
        "BINDING", "BINDING_SITE", "METAL", "METAL_BINDING",
        "ACT_SITE", "ACTIVE_SITE", "SITE", "MOTIF", "DOMAIN", "REGION"
    }

    _jitter()
    try:
        rj = _safe_get_json(session, ENTRY_URL, None, timeout, verify_ssl, debug)
        if not rj or not isinstance(rj, dict):
            raise RuntimeError("empty json")
    except Exception:
        # 失败时返回全 None，避免整体失败
        return {
            "binding_sites": None,
            "metal_binding": None,
            "active_sites": None,
            "motifs": None,
            "domains": None,
            "regions": None,
        }

    feats = rj.get("features", []) or []
    buckets = {
        "binding_sites": [],
        "metal_binding": [],
        "active_sites": [],
        "motifs": [],
        "domains": [],
        "regions": [],
    }

    def _pos_from_loc(loc: dict) -> tuple[int | None, int | None]:
        if not isinstance(loc, dict):
            return None, None
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        try:
            return (int(start) if start is not None else None,
                    int(end) if end is not None else None)
        except Exception:
            return None, None

    def _desc_from_feature(f: dict) -> str | None:
        notes = []
        t = f.get("type") or f.get("featureType")
        if t: notes.append(str(t))
        lig = (f.get("ligand") or {}).get("name") if isinstance(f.get("ligand"), dict) else None
        if lig: notes.append(f"ligand={lig}")
        desc = f.get("description")
        if desc: notes.append(desc)
        return "; ".join(notes) if notes else None

    for f in feats:
        ftype = (f.get("type") or f.get("featureType") or "").upper()
        if ftype not in FEATURE_WHITELIST:
            continue
        start, end = _pos_from_loc(f.get("location", {}))
        item = {
            "start": start,
            "end": end,
            "description": _desc_from_feature(f),
            "evidence": [ev.get("code") for ev in (f.get("evidences") or []) if isinstance(ev, dict)] or None,
        }
        if ftype in ("BINDING", "BINDING_SITE"):
            buckets["binding_sites"].append(item)
        elif ftype in ("METAL", "METAL_BINDING"):
            buckets["metal_binding"].append(item)
        elif ftype in ("ACT_SITE", "ACTIVE_SITE"):
            buckets["active_sites"].append(item)
        elif ftype == "MOTIF":
            buckets["motifs"].append(item)
        elif ftype == "DOMAIN":
            buckets["domains"].append(item)
        elif ftype == "REGION":
            buckets["regions"].append(item)

    for k in list(buckets.keys()):
        if not buckets[k]:
            buckets[k] = None
    return buckets


# ---- 主程序：整合关键位点 ----
def main():
    ap = argparse.ArgumentParser(description="Search UniProt by functional description and return sequences + key features as JSON.")
    ap.add_argument("--query", required=True, help="功能/注释文本（例如：Binds to the 23S rRNA）")
    ap.add_argument("--organism", default=None, help='物种过滤（例如："Escherichia coli" 或 "Homo sapiens"）')
    ap.add_argument("--size", type=int, default=5, help="返回条目最大数量（默认 5）")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP 超时时间（秒）")
    ap.add_argument("--unreviewed", action="store_true", help="包含未审阅条目（TrEMBL）。默认仅 Swiss-Prot。")
    ap.add_argument("--json", type=str, default=None,
                    help="如果提供路径，则将 [accession, sequence] 列表写入该文件（不包含其它元数据）")
    args = ap.parse_args()

    try:
        q = build_uniprot_query(
            args.query.strip(),
            args.organism.strip() if args.organism else None,
            reviewed_only=not args.unreviewed
        )

        session = make_session(verify_ssl=False)  # 如需严格校验证书可设 True

        raw_results = fetch_uniprot_results(
            q,
            size=max(1, args.size),
            timeout=args.timeout,
            session=session,
            verify_ssl=False,
            debug=False
        )

        normalized = []
        for e in raw_results:
            ne = normalize_entry(e)
            if ne and ne.get("sequence"):
                # 追加关键位点等特征信息
                feats = fetch_key_features_for_accession(
                    ne["accession"],
                    session=session,
                    timeout=args.timeout,
                    verify_ssl=False,
                    debug=False
                )
                ne["key_sites"] = feats
                normalized.append(ne)

        if not normalized:
            err = {
                "error": "No entries with sequences found for query.",
                "query": args.query,
                "uniprot_query": q,
                "size_requested": args.size,
            }
            print(json.dumps(err, ensure_ascii=False, indent=2))
            sys.exit(1)

        out = {
            "query": args.query,
            "uniprot_query": q,
            "count": len(normalized),
            "results": normalized,
        }

        # ---- 新增：写精简 JSON 到文件（只 accession + sequence） ----
        if args.json:
            minimal = [
                {
                    "accession": item.get("accession"),
                    "sequence": item.get("sequence"),
                }
                for item in normalized
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

        # 仍然把完整版打印到 stdout
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(0)

    except Exception as exc:
        err = {
            "error": f"{type(exc).__name__}: {str(exc)}",
            "query": args.query,
        }
        print(json.dumps(err, ensure_ascii=False, indent=2))
        sys.exit(2)


if __name__ == "__main__":
    main()
