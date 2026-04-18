# protrek_rank_seqs.py
import argparse
import json
import sys
from typing import List, Dict, Optional
import torch

from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel


def _clean_seq(s: str) -> str:
    return "".join(s.strip().upper().split())


def _read_seqs_from_file(path: str) -> List[str]:
    """
    支持两种格式：
    1) 每行一条序列（忽略空行和以#开头的注释）
    2) 简易FASTA（以 '>' 开头的描述行 + 若干行序列）
    返回: list[str] 仅含氨基酸序列
    """
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        buf = []
        is_fasta = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                is_fasta = True
                if buf:
                    seqs.append(_clean_seq("".join(buf)))
                    buf = []
            else:
                if is_fasta:
                    if line.strip():
                        buf.append(line)
                else:
                    if line.strip() and not line.lstrip().startswith("#"):
                        seqs.append(_clean_seq(line))
        if buf:
            seqs.append(_clean_seq("".join(buf)))
    # 过滤空串
    return [s for s in seqs if s]


def _read_seqs_from_json(json_path: str) -> List[Dict[str, Optional[str]]]:
    """
    读取之前 pipeline 用 --json 导出的文件:
    [
      {"accession": "P12345", "sequence": "MKT..."},
      {"accession": "Q8XXX7", "sequence": "MAD..."},
      ...
    ]

    返回:
    [
      {"sequence": "MKT...", "accession": "P12345"},
      {"sequence": "MAD...", "accession": "Q8XXX7"},
      ...
    ]
    仅保留非空序列。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            seq_raw = item.get("sequence", "")
            seq_clean = _clean_seq(seq_raw)
            if not seq_clean:
                continue
            acc = item.get("accession", None)
            out.append({
                "sequence": seq_clean,
                "accession": acc
            })
    else:
        # 容错：如果不是list就直接忽略
        pass
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Rank multiple protein sequences by similarity to a given text using ProTrek."
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="Input text description."
    )

    # 新增：可以直接喂前面步骤保存下来的 minimal JSON
    parser.add_argument(
        "--json", type=str, default=None,
        help="Path to a JSON file from previous retrieval steps (list of {accession, sequence})."
    )

    # 仍保留原有接口：--seq 可多次给，或 --seqs_file 给一堆
    parser.add_argument(
        "--seq", action="append", default=[],
        help="Amino acid sequence (can repeat this flag)."
    )
    parser.add_argument(
        "--seqs_file", type=str, default=None,
        help="Path to a file containing sequences (one per line) or FASTA."
    )

    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (cuda or cpu)."
    )
    parser.add_argument(
        "--model_dir", type=str, default="/path/to/ProtoCycle/pfam/ProTrek/weights/ProTrek_35M",
        help="Path to ProTrek model weights directory."
    )
    parser.add_argument(
        "--topk", type=int, default=0,
        help="If >0, only output top-K results."
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="If set, write JSON to this path instead of stdout."
    )

    args = parser.parse_args()

    # ---------- 收集序列候选 ----------
    # 我们现在存成一个 list[dict]，每个元素至少有 "sequence"
    import time
    start = time.time()
    collected: List[Dict[str, Optional[str]]] = []

    # 1) 来自 --json
    if args.json:
        collected.extend(_read_seqs_from_json(args.json))

    # 2) 来自 --seqs_file
    if args.seqs_file:
        for s in _read_seqs_from_file(args.seqs_file):
            collected.append({"sequence": s, "accession": None})

    # 3) 来自重复 --seq
    for s in args.seq:
        collected.append({"sequence": _clean_seq(s), "accession": None})

    # ---------- 去重并保持顺序 ----------
    # 以氨基酸序列本身作为唯一性键，如果同一条序列出现多次，只保留第一个
    seen = set()
    uniq_items: List[Dict[str, Optional[str]]] = []
    for item in collected:
        seq = item.get("sequence", "")
        if not seq:
            continue
        if seq not in seen:
            seen.add(seq)
            uniq_items.append(item)

    if not uniq_items:
        print("No sequences provided. Use --json or --seq / --seqs_file.", file=sys.stderr)
        sys.exit(1)

    # 为后续打分准备纯序列列表
    uniq_seqs: List[str] = [it["sequence"] for it in uniq_items]

    # ---------- 加载 ProTrek 模型 ----------
    config = {
        "protein_config": f"{args.model_dir}/esm2_t12_35M_UR50D",
        "text_config": f"{args.model_dir}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": f"{args.model_dir}/foldseek_t12_35M",
        "from_checkpoint": f"{args.model_dir}/ProTrek_35M.pt",
    }
    print(config)
    device = args.device
    model = ProTrekTrimodalModel(**config).eval().to(device)

    with torch.no_grad():
        # 文本只算一次
        text_embedding = model.get_text_repr([args.text])  # [1, D]

        # 序列批量
        seq_embeddings = model.get_protein_repr(uniq_seqs)  # [N, D]

        # 计算相似度分数
        # scores[i] = (seq_i ⋅ text) / temperature
        scores = (seq_embeddings @ text_embedding.T / model.temperature).squeeze(-1)  # [N]

        scores_list = scores.cpu().tolist()

    # 把分数绑回 accession/sequence
    paired = []
    for item, sc in zip(uniq_items, scores_list):
        obj = {
            "sequence": item["sequence"],
            "score": float(sc)
        }
        if item.get("accession"):
            obj["accession"] = item["accession"]
        paired.append(obj)

    # 排序（score 降序）
    paired.sort(key=lambda x: x["score"], reverse=True)

    # top-k 截断
    if args.topk and args.topk > 0:
        paired = paired[:args.topk]

    out_json = json.dumps(paired, ensure_ascii=False, indent=2)
    end = time.time()
    print(f"time: {end-start}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json + "\n")
    else:
        print(out_json)


if __name__ == "__main__":
    main()
