#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, subprocess, sys, tempfile, json, uuid, shutil
from typing import List, Tuple
from pathlib import Path

ALPHABET20 = set("ACDEFGHIKLMNPQRSTVWY")


def run(cmd: List[str], stdout_path=None, stderr_path=None, check=True):
    kw = {}
    if stdout_path:
        kw["stdout"] = open(stdout_path, "w")
    if stderr_path:
        kw["stderr"] = open(stderr_path, "w")
    return subprocess.run(cmd, check=check, **kw)


def extract_stockholm_for_family(pfam_seed: str, family_id: str, out_stockholm: str) -> bool:
    """
    从 Pfam-A.seed 抽取包含指定 family_id 的一个 Stockholm block。
    - 支持 ACC 含版本号（如 PF00069.27），只比对前缀 "PF00069"。
    - 仅当该 block 内出现匹配 ACC 时，才把完整 block（含 //）落盘。
    """
    fam_prefix = family_id.split(".")[0]  # 允许传 PF00069 或 PF00069.27
    in_block = False
    keep_block = False
    buf = []

    with open(pfam_seed, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # block start
            if line.startswith("# STOCKHOLM"):
                # 进入新块
                in_block = True
                keep_block = False
                buf = [line]
                continue

            if not in_block:
                # 还没进入任何块
                continue

            # 收集行
            buf.append(line)

            # 发现 ACC 行，判断是否命中这个 block
            if line.startswith("#=GF AC"):
                # line 例子: "#=GF AC   PF00069.27"
                parts = line.strip().split()
                if len(parts) >= 3:
                    acc_full = parts[2]  # PF00069.27
                    if acc_full.split(".")[0] == fam_prefix:
                        keep_block = True

            # block 结束
            if line.strip() == "//":
                if keep_block:
                    with open(out_stockholm, "w") as g:
                        g.writelines(buf)
                    return True
                # 否则丢弃这个块，继续找下一个
                in_block = False
                keep_block = False
                buf = []

    return False


def safe_mkdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def build_or_load_hmm(
    family_id: str,
    out_dir: str,
    pfam_hmm: str = None,
    pfam_seed: str = None,
    seed_strategy: str = "multi_species",
    msa_method: str = "seed"  # "seed" 使用 seed 的对齐，"mafft" 重新对齐
) -> Tuple[str, dict]:
    """
    返回 (hmm_id_path, meta)
    优先：hmmfetch 从 Pfam-A.hmm 直接取；否则用 Pfam-A.seed + hmmbuild。
    """
    safe_mkdir(out_dir)
    hmm_path = os.path.join(out_dir, f"{family_id}.hmm")
    meta = {"family_id": family_id, "route": None}

    # Route A: hmmfetch
    # Route A: hmmfetch
    if pfam_hmm and os.path.exists(pfam_hmm):
        # (A1) 先直接用 family_id 试
        try:
            with open(hmm_path, "w") as out:
                subprocess.run(["hmmfetch", pfam_hmm, family_id], check=True, stdout=out)
            meta["route"] = "hmmfetch"
            return hmm_path, meta
        except subprocess.CalledProcessError:
            pass  # 准备 (A2)

        # (A2) 若失败，尝试从 seed 里读出带版本号的 ACC 再 hmmfetch
        if pfam_seed and os.path.exists(pfam_seed):
            with tempfile.TemporaryDirectory() as td:
                sto_tmp = os.path.join(td, f"{family_id}.sto")
                ok = extract_stockholm_for_family(pfam_seed, family_id, sto_tmp)
                if ok:
                    acc_full = None
                    with open(sto_tmp, "r") as f:
                        for line in f:
                            if line.startswith("#=GF AC"):
                                parts = line.strip().split()
                                if len(parts) >= 3:
                                    acc_full = parts[2]  # e.g., PF00069.27
                                    break
                    if acc_full:
                        try:
                            with open(hmm_path, "w") as out:
                                subprocess.run(["hmmfetch", pfam_hmm, acc_full], check=True, stdout=out)
                            meta["route"] = "hmmfetch"
                            meta["acc_used"] = acc_full
                            return hmm_path, meta
                        except subprocess.CalledProcessError:
                            pass
        # 如果 A1/A2 都失败，落入 Route B


    # Route B: seed + hmmbuild
    if not pfam_seed or not os.path.exists(pfam_seed):
        raise FileNotFoundError("Neither valid Pfam-A.hmm nor Pfam-A.seed available to build HMM.")

    with tempfile.TemporaryDirectory() as td:
        sto = os.path.join(td, f"{family_id}.sto")
        ok = extract_stockholm_for_family(pfam_seed, family_id, sto)
        if not ok:
            raise RuntimeError(f"Family {family_id} not found in Pfam-A.seed")

        # 直接 hmmbuild：seed 是已对齐的 Stockholm
        try:
            run(["hmmbuild", "--amino", hmm_path, sto])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"hmmbuild failed: {e}")

    meta["route"] = "seed+hmmbuild"
    meta["seed_strategy"] = seed_strategy
    meta["msa_method"] = msa_method
    return hmm_path, meta


def filter_seq(seq: str, length_hint=None, max_run=1000) -> bool:
    if not set(seq) <= ALPHABET20:
        return False
    if length_hint:
        Lmin, Lmax = length_hint
        if not (Lmin <= len(seq) <= Lmax):
            return False
    # max run of the same AA
    r = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            r += 1
            if r > max_run:
                return False
        else:
            r = 1
    return True


def hmmemit_sample(
    hmm_id: str,
    num_samples: int = 8,
    length_hint: Tuple[int, int] = None,
    max_over_sample: int = 10,
    consensus: bool = False
) -> dict:
    """
    用 hmmemit 采样；如给出 length_hint，则过采样（num_samples * max_over_sample），再按长度/字母表过滤。
    返回 JSON 结构：{"samples":[{"id":..., "seq":...}, ...], "stats": {...}}
    """
    if not os.path.exists(hmm_id):
        raise FileNotFoundError(f"HMM not found: {hmm_id}")

    with tempfile.TemporaryDirectory() as td:
        outfa = os.path.join(td, "emit.fa")
        if consensus:
            run(["hmmemit", "-c", hmm_id], stdout_path=outfa)
        else:
            N = num_samples if not length_hint else max(num_samples * max_over_sample, num_samples)
            run(["hmmemit", "-N", str(N), hmm_id], stdout_path=outfa)

        # 解析 fasta
        samples = []
        current_id, current_seq = None, []
        with open(outfa) as f:
            for line in f:
                if line.startswith(">"):
                    if current_id is not None and current_seq:
                        seq = "".join(current_seq).strip()
                        samples.append((current_id, seq))
                    current_id = line[1:].strip()
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id is not None and current_seq:
                seq = "".join(current_seq).strip()
                samples.append((current_id, seq))

    # 过滤
    out = []
    for sid, s in samples:
        if filter_seq(s, length_hint=length_hint, max_run=1000):
            out.append({"id": sid, "seq": s})
        if not consensus and length_hint and len(out) >= num_samples:
            break

    stats = {
        "input_hmm": hmm_id,
        "consensus": consensus,
        "requested": num_samples,
        "generated": len(samples),
        "returned": len(out),
        "length_hint": list(length_hint) if length_hint else None
    }
    return {"samples": out, "stats": stats}


def cmd_build(args):
    hmm_path, meta = build_or_load_hmm(
        family_id=args.family_id,
        out_dir=args.out_dir,
        pfam_hmm=args.pfam_hmm,
        pfam_seed=args.pfam_seed,
        seed_strategy=args.seed_strategy,
        msa_method=args.msa_method
    )
    res = {"hmm_id": hmm_path, "meta": meta}
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_emit(args):
    res = hmmemit_sample(
        hmm_id=args.hmm_id,
        num_samples=args.num_samples,
        length_hint=(args.len_min, args.len_max) if args.len_min and args.len_max else None,
        max_over_sample=args.max_over,
        consensus=args.consensus
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="HMM step: build_or_load_hmm & hmmemit_sample")
    sub = ap.add_subparsers(required=True)

    ap_build = sub.add_parser("build_or_load_hmm", help="Create/load HMM for a family")
    ap_build.add_argument("--family-id", required=True, help="PFxxxxx, e.g. PF00069")
    ap_build.add_argument("--out-dir", required=True, help="Where to store the per-family HMM")
    ap_build.add_argument("--pfam-hmm", default='./Pfam-A.hmm')
    ap_build.add_argument("--pfam-seed", default='./Pfam-A.seed')
    ap_build.add_argument("--seed-strategy", default="multi_species")
    ap_build.add_argument("--msa-method", choices=["seed","mafft"], default="seed")
    ap_build.set_defaults(func=cmd_build)

    ap_emit = sub.add_parser("hmmemit_sample", help="Emit sequences from an HMM")
    ap_emit.add_argument("--hmm-id", required=True, help="Path to the HMM file produced above")
    ap_emit.add_argument("--num-samples", type=int, default=8)
    ap_emit.add_argument("--len-min", type=int, default=None)
    ap_emit.add_argument("--len-max", type=int, default=None)
    ap_emit.add_argument("--max-over", type=int, default=10, help="Oversample factor when using length_hint")
    ap_emit.add_argument("--consensus", action="store_true", help="Emit consensus instead of random samples")
    ap_emit.set_defaults(func=cmd_emit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
