#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
给定 accession，调用：
  https://rest.uniprot.org/uniprotkb/{ACC}?fields=accession,protein_name,cc_function,ft_binding,ft_motif,ft_domain,ft_region,sequence
解析返回的 features，输出包含 accession 与关键位点信息的 JSON，并保存到指定目录。
"""

import sys
import os
import json
import argparse
import requests
from typing import Optional, Tuple

BASE = "https://rest.uniprot.org/uniprotkb"

def _norm_ftype(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.replace(" ", "").replace("-", "").upper()

def _pos_from_loc(loc: dict) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(loc, dict):
        return None, None
    if "position" in loc and isinstance(loc["position"], dict):
        try:
            v = int(loc["position"].get("value"))
            return v, v
        except Exception:
            pass
    try:
        b = loc.get("start", {}).get("value") or loc.get("begin")
        e = loc.get("end", {}).get("value") or loc.get("end")
        return (int(b) if b else None, int(e) if e else None)
    except Exception:
        return None, None

def extract_key_sites(entry_json: dict) -> dict:
    feats = entry_json.get("features") or []
    buckets = {
        "binding_sites": [],
        "metal_binding": [],
        "active_sites": [],
        "motifs": [],
        "domains": [],
        "regions": [],
    }

    for f in feats:
        ftype = _norm_ftype(f.get("type") or f.get("featureType"))
        start, end = _pos_from_loc(f.get("location", {}))

        lig = f.get("ligand") if isinstance(f.get("ligand"), dict) else None
        ligand = {
            "name": (lig.get("name") or lig.get("label")) if lig else None,
            "id": lig.get("id") if lig else None
        } if lig else None
        lp = f.get("ligandPart") if isinstance(f.get("ligandPart"), dict) else None
        ligand_part = {
            "name": lp.get("name") if lp else None,
            "id": lp.get("id") if lp else None
        } if lp else None

        item = {
            "start": start,
            "end": end,
            "description": f.get("description") or None,
            "ligand": ligand,
            "ligand_part": ligand_part,
            "evidence": [ev.get("code") for ev in (f.get("evidences") or []) if isinstance(ev, dict)] or None,
        }

        if ftype in ("BINDINGSITE", "BINDING", "BINDING_SITE"):
            buckets["binding_sites"].append(item)
        elif ftype in ("METAL", "METALBINDING", "METAL_BINDING"):
            buckets["metal_binding"].append(item)
        elif ftype in ("ACT_SITE", "ACTIVESITE", "ACTIVE_SITE"):
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

def main():
    ap = argparse.ArgumentParser(description="Fetch UniProt features (binding_sites etc.) from fields API by accession.")
    ap.add_argument("--accession", required=True, help="UniProt accession，如 A0A010S514")
    ap.add_argument("--timeout", type=int, default=20, help="HTTP 超时（秒）")
    ap.add_argument("--out", default="out", help="输出目录（默认 out/）")
    args = ap.parse_args()

    acc = args.accession.strip()
    outdir = args.out.rstrip("/")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{acc}.json")

    url = f"{BASE}/{acc}"
    params = {
        "fields": "accession,protein_name,cc_function,ft_binding,ft_motif,ft_domain,ft_region,sequence"
    }

    try:
        r = requests.get(url, params=params, timeout=args.timeout)
        r.raise_for_status()
        data = r.json()

        function_texts = []
        for c in data.get("comments", []) or []:
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

        key_sites = extract_key_sites(data)

        seq_field = data.get("sequence", {})
        if isinstance(seq_field, dict):
            seq = seq_field.get("value")
            seqlen = seq_field.get("length")
        else:
            seq = None
            seqlen = None

        out = {
            "accession": data.get("primaryAccession") or acc,
            "protein_name": (
                (data.get("proteinDescription") or {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value")
            ) or data.get("protein_name"),
            "length": seqlen,
            "sequence": seq,
            "function": function_texts if function_texts else None,
            "key_sites": key_sites
        }

        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存到 {outfile}")

    except requests.HTTPError as e:
        msg = getattr(e.response, "text", str(e))[:500]
        print(json.dumps({"error": f"HTTPError: {msg}", "accession": acc}, ensure_ascii=False, indent=2))
        sys.exit(2)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}", "accession": acc}, ensure_ascii=False, indent=2))
        sys.exit(3)

if __name__ == "__main__":
    main()
