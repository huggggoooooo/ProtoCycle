# prosite_indexer.py
import re, json, os, sys

def prosite_pat_to_regex(ps: str) -> str:
    """
    把 PROSITE pattern 转换为 Python regex（宽松、可匹配AA序列）。
    规则要点（简化版，覆盖主流写法）：
      - 'x' -> '.'
      - 'x(3)' -> '.{3}', 'x(2,4)' -> '.{2,4}'
      - 'A(3)' -> 'A{3}'
      - '[ST]' 保留为字符类；'{P}' -> '[^P]'
      - '-' 分隔符去掉
      - '<' -> '^'， '>' -> '$'
    """
    s = ps.strip()
    s = s.replace("-", "")
    s = s.replace(".", "")  # 有些版本自带点作为分隔
    # 首尾锚
    s = s.replace("<", "^").replace(">", "$")
    # x 重复
    s = re.sub(r"x\((\d+)\)", r".{\1}", s)
    s = re.sub(r"x\((\d+),(\d+)\)", r".{\1,\2}", s)
    s = s.replace("x", ".")
    # 氨基酸重复 (A(3) -> A{3})
    s = re.sub(r"([A-Z])\((\d+)\)", r"\1{\2}", s)
    s = re.sub(r"([A-Z])\((\d+),(\d+)\)", r"\1{\2,\3}", s)
    # 非集 {P} -> [^P]
    s = re.sub(r"\{([A-Z]+)\}", lambda m: "[^" + m.group(1) + "]", s)
    # 防御性清洗：只保留 AA/正则标记
    ok = set("ACDEFGHIKLMNPQRSTVWY[]^$|{}.,()*+?\\-")
    s = "".join(ch for ch in s if ch.isalpha() or ch in ok)
    return s

def parse_prosite(dat_path: str):
    entries = []
    with open(dat_path, "r", encoding="utf-8", errors="ignore") as f:
        block = []
        for line in f:
            if line.startswith("//"):
                entries.append(block); block=[]
            else:
                block.append(line.rstrip("\n"))

    out = []
    for blk in entries:
        rec = {"ID":"", "AC":"", "DE":"", "PA":"", "CC":""}
        for ln in blk:
            if ln.startswith("ID   "): rec["ID"] = ln[5:].strip()
            elif ln.startswith("AC   "): rec["AC"] = ln[5:].strip().rstrip(";")
            elif ln.startswith("DE   "): rec["DE"] += ln[5:].strip()+" "
            elif ln.startswith("PA   "): rec["PA"] += ln[5:].strip()
            elif ln.startswith("CC   "): rec["CC"] += ln[5:].strip()+" "
        if rec["PA"]:
            pat = rec["PA"]
            # PROSITE 的 PA 行可能以 ';' 结束
            pat = pat.split(";")[0].strip()
            try:
                rx = prosite_pat_to_regex(pat)
                # 试编译
                re.compile(rx)
                rec["RX"] = rx
                out.append(rec)
            except Exception:
                continue
    return out

def main():
    dat = sys.argv[1] if len(sys.argv) > 1 else "/path/to/ProtoCycle/pfam/prosite.dat"
    if not os.path.exists(dat):
        print(f"[ERR] not found: {dat}")
        sys.exit(1)
    recs = parse_prosite(dat)
    js = os.path.join(os.path.dirname(dat), "prosite_index.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {js} with {len(recs)} entries")

if __name__ == "__main__":
    main()
