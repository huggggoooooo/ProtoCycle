"""Parse Pfam-A.clans.tsv -> family_info.jsonl (name/clan/desc per family)."""
import json
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
pfam_path = os.environ.get("PFAM_CLANS_TSV", os.path.join(_THIS_DIR, "Pfam-A.clans.tsv"))
output_json = os.path.join(_THIS_DIR, "family_info.jsonl")

with open(pfam_path, "r") as f_in, open(output_json, "w") as f_out:
    for line in f_in:
        if line.startswith("#") or not line.strip():
            continue
        cols = line.strip().split("\t")
        if len(cols) < 5:
            continue
        family_id, family_name, clan_id, clan_name, desc = cols[:5]
        record = {
            "family_id": family_id,
            "name": family_name,
            "clan_id": clan_id,
            "clan_name": clan_name,
            "desc": desc,
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Extracted family metadata to:", output_json)
