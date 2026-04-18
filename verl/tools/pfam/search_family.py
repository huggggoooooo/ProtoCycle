#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end local search for Pfam families by keywords.

Inputs: multiple keywords (AND 关系)，单个关键词内部可用 "|" 表示 OR，支持加引号的短语。
Data: requires Pfam-A.clans.tsv in local filesystem.
First run auto-builds a Whoosh index; subsequent runs are fast.

Usage examples:
  python search_family.py --pfam-clans ./Pfam-A.clans.tsv --kw kinase --kw "ATP|GTP" --topk 20
  python search_family.py --pfam-clans ./Pfam-A.clans.tsv --kw "UDP-glucose" --kw transferase

Output: JSON lines to stdout (one object with hits list).

Author: you :)
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

from whoosh import index, qparser, highlight
from whoosh.fields import Schema, ID, TEXT, STORED
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.scoring import BM25F

INDEX_DIR = "pfam_index"  # default index directory name


def build_index(pfam_clans_tsv: str, index_dir: str = INDEX_DIR) -> Tuple[bool, str]:
    """
    Build a Whoosh index from Pfam-A.clans.tsv

    Pfam-A.clans.tsv columns (tab-separated):
    Family ID | Family Name | Clan ID | Clan Name | Description
    Lines starting with '#' are comments.

    Returns: (built, index_dir)
    """
    if not os.path.exists(pfam_clans_tsv):
        raise FileNotFoundError(f"Pfam clans TSV not found: {pfam_clans_tsv}")

    os.makedirs(index_dir, exist_ok=True)

    # If already built and not empty, skip rebuild.
    maybe_ix = os.path.join(index_dir, "MAIN_WRITELOCK")
    # Better check: try opening index
    try:
        _ = index.open_dir(index_dir)
        return (False, index_dir)
    except Exception:
        pass

    # Define schema
    analyzer = StemmingAnalyzer()
    schema = Schema(
        family_id=ID(stored=True, unique=True),
        name=TEXT(stored=True, analyzer=analyzer),
        clan_id=ID(stored=True),
        clan_name=TEXT(stored=True, analyzer=analyzer),
        desc=TEXT(stored=True, analyzer=analyzer),
        # Keep original row for debugging if needed
        raw=STORED
    )

    if not index.exists_in(index_dir):
        ix = index.create_in(index_dir, schema)
    else:
        ix = index.open_dir(index_dir)

    writer = ix.writer(limitmb=512, procs=1, multisegment=True)

    total, added = 0, 0
    with open(pfam_clans_tsv, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            family_id, family_name, clan_id, clan_name, desc = parts[:5]
            writer.add_document(
                family_id=family_id,
                name=family_name,
                clan_id=clan_id,
                clan_name=clan_name,
                desc=desc,
                raw=line
            )
            added += 1

    writer.commit()
    return (True, index_dir)


def _compose_query_string(keywords: List[str]) -> str:
    """
    Compose a Whoosh query string from multiple keywords.

    - Multiple --kw are combined with AND (all must match).
    - Inside a single keyword you may use '|' to mean OR.
    - Quoted phrases are preserved by the parser.

    Examples:
      ["kinase", "ATP|GTP"] -> '(kinase) AND ((ATP) OR (GTP))'
    """
    clauses = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        # Split on '|' to support OR inside a term
        if "|" in kw:
            parts = [p.strip() for p in kw.split("|") if p.strip()]
            if parts:
                sub = " OR ".join(f'("{p}")' if " " in p else f'({p})' for p in parts)
                clauses.append(f"({sub})")
        else:
            clauses.append(f'("{kw}")' if " " in kw else f"({kw})")
    if not clauses:
        return "*"
    return " AND ".join(clauses)


def search_families(
    keywords: List[str],
    index_dir: str = INDEX_DIR,
    topk: int = 20
) -> List[dict]:
    """
    Perform multi-field search over name, desc, clan_name with BM25F.
    Returns list of {family_id, name, clan_id, clan_name, desc, score, highlights}
    """
    ix = index.open_dir(index_dir)

    fields = ["name", "desc", "clan_name"]
    parser = MultifieldParser(fields, schema=ix.schema, group=OrGroup.factory(0.0))
    query_str = _compose_query_string(keywords)
    q = parser.parse(query_str)

    results_out = []
    with ix.searcher(weighting=BM25F(B=0.75, K1=1.5)) as searcher:
        results = searcher.search(q, limit=topk)
        # Highlighter setup
        results.fragmenter = highlight.SentenceFragmenter()
        results.formatter = highlight.UppercaseFormatter()

        for hit in results:
            # Create combined highlights from name/desc/clan_name
            snippets = []
            try:
                snippets += hit.highlights("name", top=1, text=hit["name"]).split("\n")
            except Exception:
                pass
            try:
                snippets += hit.highlights("desc", top=2, text=hit["desc"]).split("\n")
            except Exception:
                pass
            try:
                snippets += hit.highlights("clan_name", top=1, text=hit["clan_name"]).split("\n")
            except Exception:
                pass
            # Clean snippets
            snippets = [s for s in (sn.strip() for sn in snippets) if s]

            results_out.append({
                "family_id": hit["family_id"],
                "name": hit["name"],
                "clan_id": hit["clan_id"],
                "clan_name": hit["clan_name"],
                "desc": hit["desc"],
                "score": float(hit.score),
                "highlights": snippets[:3]
            })

    return results_out


def main():
    ap = argparse.ArgumentParser(description="Local search for Pfam families (keywords → families).")
    ap.add_argument("--pfam-clans", default='./Pfam-A.clans.tsv')
    ap.add_argument("--index-dir", default=INDEX_DIR, help="Directory to store/load index (default: pfam_index)")
    ap.add_argument("--kw", action="append", required=True,
                    help="Keyword (can be repeated). Use '|' inside a keyword for OR. Example: --kw kinase --kw 'ATP|GTP'")
    ap.add_argument("--topk", type=int, default=20, help="Max hits to return (default: 20)")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    args = ap.parse_args()

    # Rebuild logic
    if args.rebuild and os.path.isdir(args.index_dir):
        for fn in os.listdir(args.index_dir):
            try:
                os.remove(os.path.join(args.index_dir, fn))
            except Exception:
                pass

    # Build (or open) index
    built, idx_dir = build_index(args.pfam_clans, index_dir=args.index_dir)
    if built:
        print(f"# Index built at {idx_dir}", file=sys.stderr)
    else:
        print(f"# Using existing index at {idx_dir}", file=sys.stderr)

    # Run search
    hits = search_families(args.kw, index_dir=idx_dir, topk=args.topk)

    # Output JSON to stdout
    out = {
        "query": args.kw,
        "topk": args.topk,
        "hits": hits
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
