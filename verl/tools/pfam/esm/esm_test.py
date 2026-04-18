# -*- coding: utf-8 -*-
import os
import torch
from transformers import EsmTokenizer, EsmForMaskedLM

# ======== 配置：本地模型路径 + 测试序列 ========
LOCAL_DIR = "/path/to/ProtoCycle/esm/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"

SEQ = (
    "MQKAVVMDEQAIRRALTRIAHEIIERNKGIDGCVLVGIKTRGIYLARRLAERIEQIEGASVPVGELDITLYRDDLTVKTDDHEPLVKGTNVPFPVTERNVILVDDVLFTGRTVRAAMDAVMDLGRPARIQLAVLVDRGHRELPIRADFVGKNVPTSRSELIVVELSEVDGIDQVSIHEK"
)

# 最简单的“constraint”：只在这个区间做 inpaint，其它位置保持原序列
INPAINT_REGION = (40, 60)  # 1-based, inclusive

# ======== 离线模式（可选，确保不会联网） ========
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== 加载本地 tokenizer / model ========
tokenizer = EsmTokenizer.from_pretrained(LOCAL_DIR, local_files_only=True)
model = EsmForMaskedLM.from_pretrained(LOCAL_DIR, local_files_only=True).to(device)
model.eval()

MASK_ID = tokenizer.mask_token_id
CLS_ID  = tokenizer.cls_token_id
EOS_ID  = tokenizer.eos_token_id

# 只允许从 20 个标准氨基酸里选（排除特殊 token）
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA2ID = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AA20}
ID2AA = {v: k for k, v in AA2ID.items()}
AA_IDS = torch.tensor([AA2ID[a] for a in AA20], device=device)

def clamp_region(s, e, L):
    s = max(1, min(L, s))
    e = max(1, min(L, e))
    if s > e: s, e = e, s
    return s, e

def mask_region(seq, region):
    """返回 masked 序列字符串（仅在 region 用 <mask>）。region 为 1-based 闭区间"""
    s, e = clamp_region(region[0], region[1], len(seq))
    chars = list(seq)
    for i in range(s-1, e):
        chars[i] = tokenizer.mask_token  # "<mask>"
    return "".join(chars), (s, e)

def greedy_infill(masked_seq):
    """对 masked 序列做一次 forward，对每个 <mask> 位置用 argmax 直接填充（仅一轮）"""
    inputs = tokenizer(masked_seq, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0]  # (L_with_specials, vocab)

    # 将 logits 中每个 <mask> 位置，限制在 20AA 上做 argmax
    input_ids = inputs["input_ids"][0]  # 含 [CLS] seq [EOS]
    filled_ids = input_ids.clone()

    # 遍历序列 token（跳过首尾特殊符号）
    for pos in range(1, input_ids.size(0) - 1):
        if input_ids[pos].item() == MASK_ID:
            # 只在 20AA 集合上选 argmax
            aa_logits = logits[pos, AA_IDS]  # (20,)
            best_idx = torch.argmax(aa_logits).item()
            filled_ids[pos] = AA_IDS[best_idx]

    # 把 special token 去掉，转换回氨基酸序列
    kept = [tid for tid in filled_ids.tolist() if tid not in (CLS_ID, EOS_ID)]
    toks = tokenizer.convert_ids_to_tokens(kept)
    seq = "".join([t for t in toks if len(t) == 1])  # 过滤特殊符号
    return seq

def main():
    print(f"Original length: {len(SEQ)}")
    masked_seq, region = mask_region(SEQ, INPAINT_REGION)
    print(f"Masked region: {region[0]}–{region[1]}")
    out_seq = greedy_infill(masked_seq)
    print("\n=== ORIGINAL ===")
    print(SEQ)
    print("\n=== MASKED ===")
    print(masked_seq)
    print("\n=== FILLED (argmax) ===")
    print(out_seq)

if __name__ == "__main__":
    main()
