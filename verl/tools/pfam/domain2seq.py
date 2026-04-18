#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
domain_text -> InterPro entry search (pfam/interpro/smart/prosite) -> InterPro proteins (accessions)
           -> UniProt (batch by accession) -> sequences (JSON for LLM)

用法示例：
  python domain2seq_keep_step1.py --text "Rhodanese" --size 10
  python domain2seq_keep_step1.py --text "Rhodanese" --size 20 --reviewed --debug
  python domain2seq_keep_step1.py --text "Rhodanese" --verify  # 如网络证书环境友好
"""
import os
import sys, json, time, random, argparse, ssl, warnings
from typing import Dict, List, Set, Tuple, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from urllib.parse import quote

# ---------- 全局常量 ----------
INTERPRO_BASE   = "https://www.ebi.ac.uk/interpro/api"
UNIPROT_SEARCH  = "https://rest.uniprot.org/uniprotkb/search"
UA_HEADERS = {
    "User-Agent": "desc2seq-agent (contact@example.com)",  # 建议换成你的邮箱/标识
    "Accept": "application/json",
    "Connection": "keep-alive",
}

# ---------- 默认静默 InsecureRequestWarning（关闭证书校验时更稳） ----------
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------- 稳定 session ----------
def make_session(verify_ssl: bool, retries: int = 3, backoff: float = 0.5) -> requests.Session:
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
        # 放宽 SSL 校验，避免 SSLError: EOF
        ssl._create_default_https_context = ssl._create_unverified_context
    return s

def jitter():
    time.sleep(0.5 + random.random() * 0.5)

def safe_get(session: requests.Session, url: str, params: Optional[dict], timeout: int,
             verify_ssl: bool, debug: bool):
    jitter()
    r = session.get(url, params=params, timeout=timeout, verify=verify_ssl)
    if debug:
        sys.stderr.write(f"[GET] {r.request.url}\n")
        sys.stderr.flush()
    r.raise_for_status()
    return r

# ========== 第1步（保持不变）：InterPro entry 搜索拿域 ID ==========
def interpro_search_ids_by_text(session: requests.Session, text: str, verify_ssl: bool,
                                timeout: int, debug: bool,
                                db_list=("pfam","interpro","smart","prosite"),
                                per_page: int = 100,
                                max_pages_per_db: int = 3) -> Dict[str, List[dict]]:
    """
    /entry/<db>/?search= 取域 ID；每库最多翻 max_pages_per_db 页
    返回: {db: [{"acc","name","integrated"}...]}
    """
    out: Dict[str, List[dict]] = {db: [] for db in db_list}
    for db in db_list:
        url = f"{INTERPRO_BASE}/entry/{quote(db)}/"
        
        params = {"search": text, "format": "json", "page_size": str(per_page), "size": str(per_page)}
        pages = 0
        while url and pages < max_pages_per_db:
            r = safe_get(session, url, params, timeout, verify_ssl, debug)
            data = r.json()
            results = data.get("results") or data.get("entries") or []
            for item in results:
                meta = item.get("metadata") or item.get("entry") or item
                if not isinstance(meta, dict):
                    continue
                acc  = (meta.get("accession") or meta.get("id") or "").upper().strip()
                if not acc:
                    continue
                name = (meta.get("name") or meta.get("short") or "").strip()
                integ= (meta.get("integrated") or "").upper().strip()
                out[db].append({"acc": acc, "name": name, "integrated": integ})
            url = data.get("next")
            params = None  # next 已带参数
            pages += 1
            if debug:
                sys.stderr.write(f"[entry/{db}] page={pages}, got={len(out[db])}\n")
                sys.stderr.flush()
    return out

# ========== 新的第2步：先用 InterPro 按域 ID 拉 accession，再用 UniProt 批量拉序列 ==========
def interpro_fetch_accessions_by_entry(session: requests.Session, db: str, acc: str, verify_ssl: bool,
                                       timeout: int, debug: bool,
                                       max_n: int = 400, page_size: int = 200, max_pages: int = 10) -> List[str]:
    """
    /protein/UniProt/entry/<db>/<acc>/ 返回匹配的 UniProt accessions（不依赖其直接给 sequence）
    """
    url = f"{INTERPRO_BASE}/protein/UniProt/entry/{quote(db)}/{quote(acc)}/"
    params = {"format": "json", "page_size": str(min(page_size, 200))}
    out, seen, got, pages = [], set(), 0, 0
    while url and got < max_n and pages < max_pages:
        r = safe_get(session, url, params, timeout, verify_ssl, debug)
        data = r.json()
        for item in (data.get("results") or []):
            meta = item.get("metadata") or {}
            a = (meta.get("accession") or "").upper()
            if a and a not in seen:
                seen.add(a); out.append(a); got += 1
                if got >= max_n: break
        url = data.get("next") if got < max_n else None
        params = None
        pages += 1
        if debug:
            sys.stderr.write(f"[protein/{db}:{acc}] page={pages}, acc_total={got}\n")
            sys.stderr.flush()
    return out

def uniprot_fetch_by_accessions(session: requests.Session, accessions: List[str], verify_ssl: bool,
                                timeout: int, debug: bool, reviewed_only: bool,
                                batch_size: int = 200) -> List[dict]:
    """
    逐个 accession 通过详情接口：
      https://rest.uniprot.org/uniprotkb/{ACC}?fields=accession,protein_name,cc_function,ft_binding,sequence
    拉取序列与关键信息；若 reviewed_only=True，仅保留 Swiss-Prot。
    """
    ENTRY_BASE = "https://rest.uniprot.org/uniprotkb"
    fields = "accession,protein_name,cc_function,ft_binding,sequence"
    results: List[dict] = []

    seen: Set[str] = set()
    for acc in accessions:
        a = (acc or "").strip()
        if not a or a in seen:
            continue
        seen.add(a)

        url = f"{ENTRY_BASE}/{a}"
        params = {"fields": fields}

        try:
            r = safe_get(session, url, params, timeout, verify_ssl, debug)
            print(r)
        except requests.HTTPError as e:
            # 对 404 或 410 直接跳过，其它错误抛出到外层的 except 统一处理
            status = getattr(e.response, "status_code", None)
            if status in (404, 410):
                if debug:
                    sys.stderr.write(f"[UniProt:{a}] not found (status {status})\n")
                    sys.stderr.flush()
                continue
            raise

        data = r.json() if r.content else {}
        print(data)
        if not isinstance(data, dict) or not data:
            if debug:
                sys.stderr.write(f"[UniProt:{a}] empty/invalid json\n")
                sys.stderr.flush()
            continue

        # reviewed 识别：entryType == "Swiss-Prot" 表示 reviewed
        reviewed_flag = bool(data.get("entryType") == "Swiss-Prot")
        if reviewed_only and not reviewed_flag:
            continue

        # accession
        primary_acc = data.get("primaryAccession") or data.get("accession") or a

        # title/protein name
        title = None
        try:
            title = (data.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("fullName", {})
                        .get("value"))
        except Exception:
            title = None
        if not title:
            # 回退
            title = data.get("protein_name") or data.get("uniProtkbId") or primary_acc
        if title and len(title) > 120:
            title = title[:117] + "..."

        # organism
        org = (data.get("organism") or {}).get("scientificName")

        # sequence & length
        seq, seqlen = None, None
        seq_field = data.get("sequence")
        if isinstance(seq_field, dict):
            seq = seq_field.get("value")
            seqlen = seq_field.get("length")
        elif isinstance(seq_field, str):
            seq = seq_field
            seqlen = len(seq) if seq else None
        if not seq:
            # 没有序列就跳过
            if debug:
                sys.stderr.write(f"[UniProt:{a}] no sequence\n")
                sys.stderr.flush()
            continue

        # cc_function（FUNCTION 注释，fields=cc_function 时会裁剪 comments）
        function_texts: List[str] = []
        for c in data.get("comments", []) or []:
            # 兼容可能的精简字段名/结构
            ctype = c.get("commentType") or c.get("type")
            if str(ctype).upper() == "FUNCTION":
                texts = c.get("texts") or c.get("text") or []
                if isinstance(texts, list):
                    for t in texts:
                        val = (t.get("value") if isinstance(t, dict) else str(t)).strip()
                        if val:
                            function_texts.append(val)
                else:
                    val = (texts.get("value") if isinstance(texts, dict) else str(texts)).strip()
                    if val:
                        function_texts.append(val)

        # ft_binding（BINDING 特征，fields=ft_binding 时会裁剪 features）
        binding_sites: List[dict] = []
        for f in data.get("features", []) or []:
            ftype = f.get("type") or f.get("featureType")
            if str(ftype).upper() == "BINDING":
                pos = None
                loc = f.get("location") or {}
                # 支持 position 或 begin/end
                if "position" in loc:
                    pos = loc["position"]
                else:
                    b, e = loc.get("begin"), loc.get("end")
                    if b and b == e:
                        pos = b
                desc = f.get("description") or f.get("ligand") or None
                binding_sites.append({
                    "position": pos,
                    "description": desc
                })

        results.append({
            "accession": primary_acc,
            "length": seqlen,
            "sequence": seq,
            "function": function_texts if function_texts else None,
            "binding_sites": binding_sites if binding_sites else None,
        })

    return results


def pick_top_ids(id_dict: Dict[str, List[dict]],
                 prefer=("pfam","interpro","smart","prosite"),
                 k_per_db: int = 3) -> List[Tuple[str,str]]:
    picks: List[Tuple[str,str]] = []
    for db in prefer:
        for it in id_dict.get(db, [])[:max(1, k_per_db)]:
            if it.get("acc"):
                picks.append((db, it["acc"]))
    return picks

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="InterPro step1 kept; step2 via accession->UniProt sequences")
    ap.add_argument("--text", required=True, help="域相关文本，如：Rhodanese / tyrosine kinase domain")
    ap.add_argument("--size", type=int, default=10, help="需要返回的序列条数（默认10）")
    ap.add_argument("--timeout", type=int, default=20, help="单次请求超时（默认20s）")
    ap.add_argument("--verify", action="store_true", help="开启证书校验（默认关闭以提高稳定性）")
    ap.add_argument("--reviewed", action="store_true", help="仅返回 Swiss-Prot（reviewed:true）")
    ap.add_argument("--k-per-db", type=int, default=3, help="每个数据库挑选的域ID上限（默认3）")
    ap.add_argument("--per-id-max", type=int, default=200, help="每个域ID最多抓取的 accession 数（默认200）")
    ap.add_argument("--debug", action="store_true", help="打印调试信息到 stderr")
    ap.add_argument("--json", type=str, default=None, help="输出结果保存到指定 JSON 文件（仅包含 accession 与 sequence）")

    args = ap.parse_args()

    try:
        session = make_session(verify_ssl=args.verify)

        # 1) 文本 -> 域ID（保持不变）
        id_dict = interpro_search_ids_by_text(
            session,
            args.text,
            verify_ssl=args.verify,
            timeout=args.timeout,
            debug=args.debug,
        )

        if not any(len(v) for v in id_dict.values()):
            print(json.dumps({
                "error": f"No domain IDs found for: {args.text}",
                "domain_text": args.text
            }, ensure_ascii=False, indent=2))
            sys.exit(1)

        # 选域ID（避免一次拉太多）
        candidates = pick_top_ids(id_dict, k_per_db=args.k_per_db)
        if not candidates:
            print(json.dumps({
                "error": "Domain IDs collected but empty after selection.",
                "domain_text": args.text,
                "ids": id_dict
            }, ensure_ascii=False, indent=2))
            sys.exit(2)

        # 2) 按域ID -> accession 池
        want = max(1, args.size)
        per_id_quota = max(50, min(args.per_id_max, want * 20))
        acc_pool: List[str] = []
        seen_acc: Set[str] = set()
        for db, acc in candidates:
            accs = interpro_fetch_accessions_by_entry(
                session,
                db,
                acc,
                verify_ssl=args.verify,
                timeout=args.timeout,
                debug=args.debug,
                max_n=per_id_quota,
                page_size=200,
                max_pages=10,
            )
            for a in accs:
                if a not in seen_acc:
                    seen_acc.add(a)
                    acc_pool.append(a)
            if len(acc_pool) >= want:  # 足够大就停
                acc_pool = acc_pool[:want]
                break

        if not acc_pool:
            print(json.dumps({
                "error": "No accessions collected from InterPro protein endpoint.",
                "domain_text": args.text,
                "picked_ids": candidates
            }, ensure_ascii=False, indent=2))
            sys.exit(3)

        # 3) UniProt 按 accession 批量拿序列
        seq_items = uniprot_fetch_by_accessions(
            session,
            acc_pool,
            verify_ssl=args.verify,
            timeout=args.timeout,
            debug=args.debug,
            reviewed_only=args.reviewed,
            batch_size=200,
        )
        if not seq_items:
            print(json.dumps({
                "error": "No sequences returned from UniProt by accession.",
                "domain_text": args.text,
                "picked_ids": candidates
            }, ensure_ascii=False, indent=2))
            sys.exit(4)

        # 去重并截断到 size
        merged, seen2 = [], set()
        for it in seq_items:
            acc = it.get("accession")
            if acc and acc not in seen2:
                seen2.add(acc)
                merged.append(it)
            if len(merged) >= want:
                break

        if not merged:
            print(json.dumps({
                "error": "Sequences fetched but empty after dedup/filter.",
                "domain_text": args.text
            }, ensure_ascii=False, indent=2))
            sys.exit(5)

        out = {
            "domain_text": args.text,
            # "picked_ids": candidates,  # 你之前注释掉了就保持一致
            "count": len(merged),
            "results": merged,
        }

        # ---- 新增：如果传了 --json，就把 accession + sequence 写到文件 ----
        if args.json:
            minimal = [
                {
                    "accession": it.get("accession"),
                    "sequence": it.get("sequence"),
                }
                for it in merged
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

        # 正常stdout输出完整版
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({
            "error": f"{type(e).__name__}: {e}",
            "domain_text": args.text
        }, ensure_ascii=False, indent=2))
        sys.exit(9)


if __name__ == "__main__":
    main()
