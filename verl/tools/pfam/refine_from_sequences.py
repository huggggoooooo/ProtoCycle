#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, random, sys, math
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

ALPHABET20 = set("ACDEFGHIKLMNPQRSTVWY")

# ---------- IO ----------
def load_fasta(path:str):
    if not path: return []
    cur=None; buf=[]; out=[]
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None: out.append({"id":cur,"seq":"".join(buf)})
                cur=line[1:].strip(); buf=[]
            else:
                buf.append(line.strip())
    if cur is not None: out.append({"id":cur,"seq":"".join(buf)})
    return out

def save_json(obj, path=None):
    s=json.dumps(obj, ensure_ascii=False, indent=2)
    if path: open(path,"w").write(s)
    else: print(s)

# ---------- props ----------
def max_run_and_k2_entropy(seq:str, k:int=2):
    from collections import Counter
    # max run
    mr, r = 1, 1
    for i in range(1, len(seq)):
        if seq[i]==seq[i-1]:
            r += 1; mr = max(mr, r)
        else:
            r = 1
    # k2 entropy
    if len(seq) < k: return mr, 0.0
    km=[seq[i:i+k] for i in range(len(seq)-k+1)]
    c=Counter(km); tot=sum(c.values()); ent=0.0
    for v in c.values():
        p=v/tot; ent += -p*math.log(p+1e-9)
    return mr, ent

def basic_checks(seq:str, len_min:int, len_max:int, max_run:int):
    if not set(seq) <= ALPHABET20:
        return False, "alphabet"
    if not (len_min <= len(seq) <= len_max):
        return False, f"length:{len(seq)}"
    mr, _ = max_run_and_k2_entropy(seq)
    if mr > max_run:
        return False, f"max_run:{mr}"
    return True, ""

# ---------- locks ----------
def apply_locks(seq:str, locks:List[Dict])->List[bool]:
    keep=[False]*len(seq)
    for lk in (locks or []):
        if "start" in lk and "end" in lk:
            s=max(0,int(lk["start"])); e=min(len(seq), int(lk["end"]))
            for k in range(s,e): keep[k]=True
        elif "pattern" in lk:
            pat=str(lk["pattern"])
            i=seq.find(pat)
            while i!=-1:
                for k in range(i, i+len(pat)): keep[k]=True
                i=seq.find(pat, i+1)
    return keep

# ---------- MLM helpers ----------
def blocked_positions(L:int, block:int, stride:int, keep_mask:Optional[List[bool]]):
    i=0
    while i<L:
        idx=list(range(i, min(i+block, L)))
        if keep_mask: idx=[t for t in idx if not keep_mask[t]]
        if idx: yield idx
        i += stride

def ppl_like(seq:str, tok, model, device, step:int=128)->float:
    model.eval()
    hop=max(1, len(seq)//step)
    loss=0.0; n=0
    with torch.inference_mode():
        for i in range(0,len(seq),hop):
            chars=list(seq); chars[i]="[MASK]"
            txt=" ".join(chars)
            inp=tok(txt, return_tensors="pt").to(device)
            out=model(**inp).logits
            mask_id=tok.mask_token_id
            loc=(inp["input_ids"][0]==mask_id).nonzero(as_tuple=True)[0][0].item()
            target=seq[i]
            tid=tok.convert_tokens_to_ids(target) if len(target)==1 else None
            if tid is None: continue
            prob=torch.softmax(out[0,loc],dim=-1)[tid].float().clamp_min(1e-9)
            loss += (-prob.log()).item(); n+=1
    return loss/max(n,1)

def refine_once(seq:str, tok, model, device, top_k:int, block:int, stride:int, keep_mask:Optional[List[bool]]):
    model.eval()
    ids=list(seq); L=len(ids)
    with torch.inference_mode():
        for idxs in blocked_positions(L, block, stride, keep_mask):
            masked=list(ids)
            for i in idxs: masked[i]="[MASK]"
            txt=" ".join(masked)
            inp=tok(txt, return_tensors="pt").to(device)
            out=model(**inp).logits
            mlm_ids=inp["input_ids"][0]
            mask_id=tok.mask_token_id
            mask_locs=(mlm_ids==mask_id).nonzero(as_tuple=True)[0].tolist()
            for pos, write_pos in zip(mask_locs, idxs):
                logits=out[0,pos]
                probs=torch.softmax(logits,dim=-1)
                k=min(top_k, probs.size(0))
                topv, topi = torch.topk(probs, k=k)
                choice = topi[ random.randrange(k) ].item()
                tok_str = tok.convert_ids_to_tokens([choice])[0]
                aa = tok_str if (len(tok_str)==1 and tok_str in ALPHABET20) else random.choice(list(ALPHABET20))
                ids[write_pos]=aa
    return "".join(ids)

def load_model(model_path:str, dtype:str, device_map:str, quant:str):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # dtype
    if dtype.lower()=="bf16": torch_dtype=torch.bfloat16
    elif dtype.lower()=="fp16": torch_dtype=torch.float16
    else: torch_dtype=torch.float32
    # quant
    quant_kwargs={}
    if quant=="8bit":   quant_kwargs=dict(load_in_8bit=True)
    elif quant=="4bit": quant_kwargs=dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)
    dm=None if device_map=="none" else device_map
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=dm,
        low_cpu_mem_usage=True, trust_remote_code=True, **quant_kwargs
    )
    if dm is None:
        device="cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    return tok, model

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser(description="Refine protein sequences with local ESM (input: sequences).")
    # input
    ap.add_argument("--in_fasta", help="FASTA with one or more sequences")
    ap.add_argument("--seq", help="Single sequence string (will be named 'seq1')")
    # model
    ap.add_argument("--model-path", default='/path/to/ProtoCycle/esm/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc')
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--device-map", default="auto", choices=["auto","balanced","cpu","none"])
    ap.add_argument("--quant", default="none", choices=["none","8bit","4bit"])
    # refine
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--lock-json", default=None, help='[{"pattern":"HExH"},{"start":120,"end":130}]')
    ap.add_argument("--ppx-step", type=int, default=128)
    # props
    ap.add_argument("--len-min", type=int, default=1)
    ap.add_argument("--len-max", type=int, default=100000)
    ap.add_argument("--max-run", type=int, default=6)
    # output
    ap.add_argument("--out-json", help="Write results to file; default prints to stdout")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    # load input
    items=[]
    if args.in_fasta:
        items = load_fasta(args.in_fasta)
    if args.seq:
        items.append({"id":"seq1","seq":args.seq})
    if not items:
        print("No input sequences. Use --in_fasta or --seq.", file=sys.stderr); sys.exit(1)

    # model
    tok, model = load_model(args.model_path, args.dtype, args.device_map, args.quant)
    device = next(model.parameters()).device
    locks = json.loads(args.lock-json) if hasattr(args,"lock-json") and args.__dict__.get("lock-json") else None
    if locks is None and args.lock_json:
        locks = json.loads(args.lock_json)

    refined=[]
    props_ok=[]; props_viols=[]

    for it in items:
        s0 = it["seq"]
        # props check (input)
        ok, why = basic_checks(s0, args.len_min, args.len_max, args.max_run)
        keep_mask = apply_locks(s0, locks or [])
        # base ppl
        base_ppx = ppl_like(s0, tok, model, device, step=args.ppx_step)
        best = (s0, base_ppx)

        for _ in range(args.iters):
            cand = refine_once(best[0], tok, model, device, args.top_k, args.block, args.stride, keep_mask)
            ppx = ppl_like(cand, tok, model, device, step=args.ppx_step)
            if ppx < best[1]:
                best = (cand, ppx)

        # props check (output)
        ok2, why2 = basic_checks(best[0], args.len_min, args.len_max, args.max_run)
        refined.append({
            "id": it["id"],
            "parent_ppx": base_ppx,
            "seq": best[0],
            "ppx": best[1],
            "input_props_ok": ok,
            "input_props_reason": why,
            "output_props_ok": ok2,
            "output_props_reason": why2
        })
        if ok2: props_ok.append(it["id"])
        else: props_viols.append({"seq_id": it["id"], "rule": why2})

    # ranking by ppl
    ranking = sorted(refined, key=lambda x: x["ppx"])
    out = {
        "refined": refined,
        "ranking": [r["id"] for r in ranking],
        "params": {
            "iters": args.iters, "top_k": args.top_k,
            "block": args.block, "stride": args.stride,
            "ppx_step": args.ppx_step
        }
    }
    save_json(out, args.out_json)

if __name__ == "__main__":
    main()
