#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute protein design metrics from a merged CSV:
- Plausibility: ESM Pseudo-Perplexity, Repeat%
- Foldability: pTM, pLDDT, PAE from Chai-1
- Language Alignment: ProTrek Score, EvoLlama Score
- Retrieval Acc (ProTrek-35M)

新增：
- has_tag:   是否认为这条样本有合法的 <answer> 序列（1/0），来自输入 CSV
- is_valid:  pred_seq 是否为严格 20AA 序列（1/0）

只有 has_tag==1 且 is_valid==1 时才真正计算这些指标；
否则所有指标记为 NaN。
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import shutil
import uuid

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertModel


# Path to a local chai-lab checkout. Override with env var CHAI_LAB_ROOT; otherwise
# defaults to <repo>/eval_tools/chai-lab (not shipped; clone separately).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CHAI_ROOT = os.environ.get("CHAI_LAB_ROOT", os.path.join(_THIS_DIR, "chai-lab"))

if LOCAL_CHAI_ROOT and os.path.isdir(LOCAL_CHAI_ROOT) and LOCAL_CHAI_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_CHAI_ROOT)


# -----------------------------
# Helpers to find columns
# -----------------------------

def pick_column(df: pd.DataFrame, candidates: List[str], desc: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Cannot find column for {desc}. Tried: {candidates}. Existing: {df.columns.tolist()}"
    )


# -----------------------------
# 20AA 合法性检查
# -----------------------------

VALID_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def is_valid_aa_sequence(seq: str) -> bool:
    """
    检查序列是否全部由 20 个标准氨基酸字母组成（不允许 X / B / Z 等）。
    空串或含非法字符 => False
    """
    if not isinstance(seq, str):
        return False
    s = "".join(seq.strip().upper().split())
    if not s:
        return False
    for ch in s:
        if ch not in VALID_AA_SET:
            return False
    return True


# -----------------------------
# Plausibility: ESM Pseudo-Perplexity
# -----------------------------

@torch.no_grad()
def esm_pseudo_perplexity(
    seq: str,
    tokenizer,
    model,
    device: torch.device,
) -> float:
    """
    Pseudo-perplexity for a single protein sequence.

    PPL = exp( - 1/L * sum_i log p(x_i | x_{-i}) ), i over aa tokens.
    """
    seq = "".join(seq.strip().upper().split())
    if not seq:
        return float("nan")

    # Tokenize once with specials
    encoded = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Find positions corresponding to AA tokens (exclude special tokens)
    # Simple heuristic: skip the first and last tokens (CLS + EOS).
    seq_len = input_ids.size(1)
    token_positions = list(range(1, seq_len - 1))
    mask_token_id = tokenizer.mask_token_id

    if mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id; ESM masked LM is required.")

    log_likelihood = 0.0
    n_tokens = 0

    for pos in token_positions:
        masked_ids = input_ids.clone()
        target_id = masked_ids[0, pos].item()
        masked_ids[0, pos] = mask_token_id

        outputs = model(input_ids=masked_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, pos]
        log_probs = torch.log_softmax(logits, dim=-1)
        log_likelihood += log_probs[target_id].item()
        n_tokens += 1

    if n_tokens == 0:
        return float("nan")

    avg_ll = log_likelihood / n_tokens
    ppl = math.exp(-avg_ll)
    return float(ppl)


# -----------------------------
# Plausibility: Repeat% (PDFBench-like)
# -----------------------------

def compute_repeat(sequence: str) -> float:
    """
    PDFBench 的 Repeat 实现：
    返回 [0, 1] 之间的比例（重复片段在整条序列中占的长度比例）。
    """
    # 清理一下：去空白 + 大写
    sequence = "".join(sequence.strip().upper().split())
    n = len(sequence)

    # Bound check
    if n == 0:
        return 0.0

    regions = []
    max_window_size = min(20, n // 2)  # limit the window size
    for window_size in range(1, max_window_size + 1):
        for i in range(n - window_size + 1):
            pattern = sequence[i : i + window_size]
            count = 1

            j = i + window_size
            while j <= n - window_size:
                next_segment = sequence[j : j + window_size]
                if next_segment == pattern:
                    count += 1
                    j += window_size
                else:
                    break

            # 只有连续重复 >=3 次才算
            if count >= 3:
                start = i
                end = i + window_size * count
                regions.append((start, end))

    if not regions:
        return 0.0

    # Sort and merge intervals
    sorted_regions = sorted(regions, key=lambda x: x[0])
    merged = []
    for start, end in sorted_regions:
        if not merged:
            merged.append([start, end])
        else:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])

    total_repeat = sum(end - start for start, end in merged)
    proportion = total_repeat / n
    return proportion


# -----------------------------
# Foldability: Chai-1 pTM / pLDDT / PAE
# -----------------------------

def run_chai1_for_sequence(
    seq: str,
    name: str,
    output_root: Path,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    用本地 chai-1 跑一条序列，返回 (ptm_mean, plddt_mean, pae_mean)。

    - plddt / pae: 直接用 run_inference 返回的 outputs.plddt / outputs.pae
    - ptm: 尝试从 scores.model_idx_2.npz 里读，如果没有相关 key 就返回 NaN
    - 运行结束后会删除对应的临时输出目录 output_root/<name>_<uuid>
    """

    # 先给三个值一个默认 NaN，避免中途异常导致未定义
    ptm_mean = float("nan")
    plddt_mean = float("nan")
    pae_mean = float("nan")

    # --------- 确保用的是本地 chai_lab_local --------- #
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_CHAI_ROOT2 = os.path.join(SCRIPT_DIR, "chai_lab_local")
    if LOCAL_CHAI_ROOT2 not in sys.path:
        sys.path.insert(0, LOCAL_CHAI_ROOT2)

    try:
        from chai_lab.chai1 import run_inference
    except ImportError:
        from chai_lab_local.chai1 import run_inference

    # --------- 跑 chai-1 结构预测 --------- #
    seq = "".join(seq.strip().upper().split())

    # 给每次调用一个唯一的临时目录名：<name>_<8位uuid>
    unique_tag = uuid.uuid4().hex[:8]
    tmp_dir_name = f"{name}_{unique_tag}"
    output_dir = output_root / tmp_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_text = f">protein|name={name}\n{seq}\n"
    fasta_path = output_dir / f"{name}.fasta"
    fasta_path.write_text(fasta_text)

    try:
        outputs = run_inference(
            fasta_file=fasta_path,
            output_dir=output_dir,
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            seed=42,
            device=device,
            use_esm_embeddings=True,
        )

        # 你之前已经验证过这两个是有的
        plddt_mean = float(outputs.plddt[0].mean())
        pae_mean = float(outputs.pae[0].mean())

        # --------- ptm: 从 scores.model_idx_2.npz 里尝试读取 --------- #
        score_path = output_dir / "scores.model_idx_2.npz"
        if score_path.exists():
            try:
                score = np.load(score_path, allow_pickle=True)
                # 尝试几种可能的 key
                candidate_keys = []
                for k in score.keys():
                    kl = k.lower()
                    if "ptm" in kl:  # 覆盖 ptm / iptm / pTM 等
                        candidate_keys.append(k)
                if candidate_keys:
                    v = score[candidate_keys[0]]
                    ptm_mean = float(np.mean(v))
            except Exception as e:
                print(
                    f"[Chai-1] Warning: failed to read ptm from {score_path}: {e}",
                    file=sys.stderr,
                )

    finally:
        # 无论成功与否，都尝试删除临时目录
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception as e:
            print(
                f"[Chai-1] Warning: failed to remove temp dir {output_dir}: {e}",
                file=sys.stderr,
            )

    return ptm_mean, plddt_mean, pae_mean


# -----------------------------
# Language Alignment: ProTrek Score via external script
# -----------------------------

def compute_protrek_score_via_script(
    text: str,
    seq: str,
    protrek_python: str,
    protrek_script: str,
    model_dir: str,
    device: str = "cuda",
) -> Optional[float]:
    """
    调用 ProTrek 脚本，并通过 --out 写到一个临时 JSON 文件中，再读取 score。
    读取完后会删除临时文件。
    """
    text = (text or "").strip()
    seq = "".join((seq or "").strip().upper().split())
    if not text or not seq:
        return None

    # 先创建一个临时文件名（不占用文件描述符）
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(tmp_fd)  # 我们只需要路径，先关掉 fd

    cmd = [
        protrek_python,
        protrek_script,
        "--text",
        text,
        "--seq",
        seq,
        "--device",
        device,
        "--model_dir",
        model_dir,
        "--topk",
        "1",
        "--out",
        tmp_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"[ProTrek] Error for one sample: {e}\nSTDERR:\n{e.stderr}\n"
        )
        # 清理临时文件
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None

    # 读取 tmp JSON
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.stderr.write(
            f"[ProTrek] Failed to read JSON from {tmp_path}: {e}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None
    finally:
        # 尝试删除临时文件
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # 解析 score
    try:
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "score" in first:
                return float(first["score"])
        if isinstance(data, dict) and "score" in data:
            return float(data["score"])
    except Exception as e:
        sys.stderr.write(f"[ProTrek] Error extracting score: {e}\nData: {data}\n")
        return None

    sys.stderr.write(f"[ProTrek] No 'score' field in data: {data}\n")
    return None


def _clean_seq_simple(s: str) -> str:
    """和 ProTrek 脚本里的 _clean_seq 保持一致：去空白 + 大写。"""
    return "".join((s or "").strip().upper().split())


def compute_retrieval_hit_with_protrek35(
    query_text: str,
    pos_seq: str,
    all_seqs: List[str],
    self_idx: int,
    num_candidates: int,
    python_exe: str,
    script_path: str,
    model_dir: str,
    device: str = "cuda",
) -> Optional[int]:
    """
    用 35M ProTrek 做一次 retrieval：
    返回 1 / 0 / None
    """
    query_text = (query_text or "").strip()
    pos_seq_clean = _clean_seq_simple(pos_seq)
    if not query_text or not pos_seq_clean:
        return None

    # 构建负样本序列池（排除自己 + 排除和正样本完全相同的）
    indices = list(range(len(all_seqs)))
    indices.remove(self_idx)
    neg_pool = []
    seen = {pos_seq_clean}
    random.shuffle(indices)
    for j in indices:
        s = _clean_seq_simple(all_seqs[j])
        if not s or s in seen:
            continue
        seen.add(s)
        neg_pool.append(s)
        if len(neg_pool) >= max(0, num_candidates - 1):
            break

    if not neg_pool:
        # 没有可用的负样本，就没法算
        return None

    candidates = [pos_seq_clean] + neg_pool

    # 写临时 seqs_file（每行一条序列）
    fd_seqs, path_seqs = tempfile.mkstemp(suffix=".txt")
    os.close(fd_seqs)
    with open(path_seqs, "w", encoding="utf-8") as f:
        for s in candidates:
            f.write(s + "\n")

    # 临时输出 JSON
    fd_out, path_out = tempfile.mkstemp(suffix=".json")
    os.close(fd_out)

    cmd = [
        python_exe,
        script_path,
        "--text",
        query_text,
        "--seqs_file",
        path_seqs,
        "--device",
        device,
        "--model_dir",
        model_dir,
        "--topk",
        "0",  # 不截断，拿全部
        "--out",
        path_out,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"[Retrieval] ProTrek35 call failed: {e}\nSTDERR:\n{e.stderr}\n"
        )
        try:
            os.remove(path_seqs)
            os.remove(path_out)
        except OSError:
            pass
        return None

    # 读取 JSON 结果
    try:
        with open(path_out, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.stderr.write(
            f"[Retrieval] Failed to read JSON {path_out}: {e}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n"
        )
        try:
            os.remove(path_seqs)
            os.remove(path_out)
        except OSError:
            pass
        return None
    finally:
        try:
            os.remove(path_seqs)
            os.remove(path_out)
        except OSError:
            pass

    # data 应该是按 score 降序排好的 list[dict]
    if not isinstance(data, list) or not data:
        sys.stderr.write(f"[Retrieval] Unexpected JSON format: {data}\n")
        return None

    top_seq = data[0].get("sequence", "")
    top_seq_clean = _clean_seq_simple(top_seq)
    # 只要 top 的序列等于正样本序列，就认为命中
    return 1 if top_seq_clean == pos_seq_clean else 0


def compute_evollama_score_via_script(
    gt_text: str,
    seq: str,
    python_exe: str,
    script_path: str,
    config_path: str,
    pubmedbert_path: str,
    question: Optional[str] = None,
    foldseek: Optional[str] = None,
) -> Optional[float]:
    """
    调用 Evolla 环境里的 compute_evollama_score.py，解析 stdout 中的 Score。
    """
    gt_text = (gt_text or "").strip()
    seq_clean = _clean_seq_simple(seq)
    if not gt_text or not seq_clean:
        return None

    cmd = [
        python_exe,
        script_path,
        "--config_path",
        config_path,
        "--gt_text",
        gt_text,
        "--seq",
        seq_clean,
        "--pubmedbert_path",
        pubmedbert_path,
    ]
    if question:
        cmd.extend(["--question", question])
    if foldseek:
        cmd.extend(["--foldseek", foldseek])

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"[EvoLlama] Error when calling script: {e}\nSTDERR:\n{e.stderr}\n"
        )
        return None

    # 解析 stdout 里的 "Score: xxx"
    score = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("Score:"):
            # e.g. "Score: 0.1234"
            part = line.split("Score:", 1)[1].strip()
            try:
                score = float(part.split()[0])
            except Exception:
                pass
            break

    if score is None:
        sys.stderr.write(
            "[EvoLlama] Could not parse score from output:\n"
            f"{proc.stdout}\nSTDERR:\n{proc.stderr}\n"
        )

    return score


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    # ESM settings
    parser.add_argument(
        "--esm_model_path",
        type=str,
        default=os.environ.get(
            "ESM_MODEL_PATH",
            "/path/to/esm2_t36_3B_UR50D",
        ),
    )
    parser.add_argument("--esm_device", type=str, default="cuda:0")

    # Chai-1 settings
    parser.add_argument("--chai_device", type=str, default="cuda:0")
    parser.add_argument(
        "--chai_output_root",
        type=str,
        default="./chai_outputs",
        help="Directory where Chai-1 outputs (cif, npz, etc.) will be stored.",
    )
    parser.add_argument(
        "--skip_chai",
        action="store_true",
        help="If set, skip foldability metrics (pTM/pLDDT/PAE).",
    )

    # ProTrek settings
    parser.add_argument(
        "--protrek_python",
        type=str,
        default=os.environ.get("PROTREK_ENV_PYTHON", "/path/to/miniconda3/envs/protrek/bin/python"),
    )
    parser.add_argument(
        "--protrek_script",
        type=str,
        default=os.path.abspath(os.path.join(
            _THIS_DIR, "..", "verl", "tools", "pfam", "ProTrek",
            "caculate_similarity_text_seq.py",
        )),
    )
    parser.add_argument(
        "--protrek_model_dir",
        type=str,
        default=os.environ.get("PROTREK_650M_DIR", "/path/to/ProTrek_650M"),
    )
    parser.add_argument(
        "--skip_protrek",
        action="store_true",
        help="If set, skip ProTrek and fill NaN.",
    )

    # Retrieval settings
    parser.add_argument(
        "--skip_retrieval",
        action="store_true",
        help="If set, skip Retrieval Accuracy computation.",
    )
    parser.add_argument(
        "--retrieval_script",
        type=str,
        default=os.path.abspath(os.path.join(
            _THIS_DIR, "..", "verl", "tools", "pfam", "ProTrek",
            "caculate_similarity_text_seq_35M.py",
        )),
        help="ProTrek script for Retrieval Accuracy (35M).",
    )
    parser.add_argument(
        "--retrieval_model_dir",
        type=str,
        default=os.environ.get("PROTREK_35M_DIR", "/path/to/ProTrek_35M"),
        help="Model dir for 35M ProTrek used in retrieval.",
    )
    parser.add_argument(
        "--retrieval_device",
        type=str,
        default="cuda",
        help="Device string passed to retrieval ProTrek script.",
    )
    parser.add_argument(
        "--retrieval_num_candidates",
        type=int,
        default=32,
        help="Total #candidates per query (1 positive + N-1 negative sequences).",
    )

    # EvoLlama / Evolla settings
    parser.add_argument(
        "--use_evollama",
        action="store_false",
        help="Enable EvoLlama(Evolla) score computation.",
    )
    parser.add_argument(
        "--evollama_config",
        type=str,
        default=os.environ.get("EVOLLAMA_CONFIG", "/path/to/Evolla/config/Evolla_10B.yaml"),
        help="Path to Evolla/EvoLlama config yaml (external, not shipped).",
    )
    parser.add_argument(
        "--evollama_python",
        type=str,
        default=os.environ.get("EVOLLAMA_PYTHON", "/path/to/miniconda3/envs/Evolla/bin/python"),
        help="Python executable in Evolla conda env.",
    )
    parser.add_argument(
        "--evollama_script",
        type=str,
        default=os.environ.get("EVOLLAMA_SCRIPT", "/path/to/Evolla/compute_evollama_score.py"),
        help="Path to compute_evollama_score.py (external, not shipped).",
    )
    parser.add_argument(
        "--evollama_pubmedbert_path",
        type=str,
        default=os.environ.get("EVOLLAMA_PUBMEDBERT", "/path/to/pubmedbert-base-embeddings"),
        help="Local path to pubmedbert-base-embeddings.",
    )
    parser.add_argument(
        "--evollama_device",
        type=str,
        default="cuda:0",
        help="Device for EvoLlama and text encoder.",
    )
    parser.add_argument(
        "--evollama_question",
        type=str,
        default=None,
        help="Question/prompt text passed to EvoLlama; if None, use default.",
    )
    parser.add_argument(
        "--evollama_foldseek",
        type=str,
        default=None,
        help="Optional foldseek string for EvoLlama, if your model uses it.",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    idx_col = pick_column(
        df,
        ["row_idx", "jsonl_index"],
        "row index",
    )

    seq_col = pick_column(
        df,
        ["pred_seq"],
        "predicted sequence",
    )

    req_col = pick_column(
        df,
        ["requirement_parquet"],
        "requirement text",
    )

    # answer-tag 标志列：优先 has_tag，其次 pred_in_answer_tag
    tag_col = "has_tag"
    if tag_col is None:
        print(
            "[Warning] No has_tag / pred_in_answer_tag column in input CSV; "
            "all rows will be treated as has_tag=1.",
            file=sys.stderr,
        )

    all_pred_seqs = df[seq_col].astype(str).tolist()

    # ---- Load ESM model ----
    esm_device = torch.device(args.esm_device)
    print(f"[ESM] Loading model from {args.esm_model_path} to {esm_device} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model_path)
    esm_model = AutoModelForMaskedLM.from_pretrained(args.esm_model_path)
    esm_model.to(esm_device)
    esm_model.eval()

    # ---- Prepare Chai / ProTrek ----
    chai_device = torch.device(args.chai_device)
    chai_root = Path(args.chai_output_root)
    chai_root.mkdir(parents=True, exist_ok=True)

    results = []

    for i, row in df.iterrows():
        row_id = int(row[idx_col])
        seq = str(row[seq_col])
        requirement = str(row[req_col])

        # 1) has_tag：输入 CSV 里的标志（如果没有，就当作 1）
        if tag_col is not None:
            raw_flag = row[tag_col]
            try:
                has_tag = int(raw_flag) != 0
            except Exception:
                has_tag = bool(raw_flag)
        else:
            has_tag = True
        has_tag_int = 1 if has_tag else 0

        # 2) is_valid：严格 20AA 检查
        valid_aa = is_valid_aa_sequence(seq)
        is_valid_int = 1 if valid_aa else 0

        print(
            f"[Row {row_id}] processing... (has_tag={has_tag_int}, is_valid={is_valid_int})",
            file=sys.stderr,
        )

        # 如果任一条件不满足，直接跳过所有评测
        if (not has_tag) or (not valid_aa):
            ppl = float("nan")
            repeat_pct = float("nan")
            ptm = float("nan")
            plddt = float("nan")
            pae = float("nan")
            protrek_score = float("nan")
            evollama_score = float("nan")
            retrieval_acc = float("nan")
        else:
            # Plausibility
            try:
                ppl = esm_pseudo_perplexity(seq, tokenizer, esm_model, esm_device)
            except Exception as e:
                sys.stderr.write(f"[Row {row_id}] ESM PPL failed: {e}\n")
                ppl = float("nan")

            repeat_pct = compute_repeat(seq) * 100.0

            # Foldability
            if not args.skip_chai:
                try:
                    ptm, plddt, pae = run_chai1_for_sequence(
                        seq=seq,
                        name=f"sample_{row_id}",
                        output_root=chai_root,
                        device=chai_device,
                    )
                except Exception as e:
                    sys.stderr.write(f"[Row {row_id}] Chai-1 failed: {e}\n")
                    ptm = plddt = pae = float("nan")
            else:
                ptm = plddt = pae = float("nan")

            # Language Alignment: ProTrek Score
            if not args.skip_protrek:
                protrek_score = compute_protrek_score_via_script(
                    text=requirement,
                    seq=seq,
                    protrek_python=args.protrek_python,
                    protrek_script=args.protrek_script,
                    model_dir=args.protrek_model_dir,
                    device="cuda",
                )
            else:
                protrek_score = float("nan")

            # EvoLlama Score
            if args.use_evollama:
                try:
                    score = compute_evollama_score_via_script(
                        gt_text=requirement,
                        seq=seq,
                        python_exe=args.evollama_python,
                        script_path=args.evollama_script,
                        config_path=args.evollama_config,
                        pubmedbert_path=args.evollama_pubmedbert_path,
                        question=args.evollama_question,
                        foldseek=args.evollama_foldseek,
                    )
                    evollama_score = float(score) if score is not None else float("nan")
                except Exception as e:
                    sys.stderr.write(f"[Row {row_id}] EvoLlama failed: {e}\n")
                    evollama_score = float("nan")
            else:
                evollama_score = float("nan")

            # Retrieval Accuracy
            if not args.skip_retrieval:
                try:
                    hit = compute_retrieval_hit_with_protrek35(
                        query_text=requirement,
                        pos_seq=seq,
                        all_seqs=all_pred_seqs,
                        self_idx=i,  # 注意这里用的是 df 的行号 i
                        num_candidates=args.retrieval_num_candidates,
                        python_exe=args.protrek_python,
                        script_path=args.retrieval_script,
                        model_dir=args.retrieval_model_dir,
                        device=args.retrieval_device,
                    )
                    retrieval_acc = float(hit) if hit is not None else float("nan")
                except Exception as e:
                    sys.stderr.write(f"[Row {row_id}] Retrieval failed: {e}\n")
                    retrieval_acc = float("nan")
            else:
                retrieval_acc = float("nan")

        results.append(
            {
                "idx": row_id,
                "esm_ppl": ppl,
                "repeat_pct": repeat_pct,
                "ptm_mean": ptm,
                "plddt_mean": plddt,
                "pae_mean": pae,
                "protrek_score": protrek_score,
                "evollama_score": evollama_score,
                "retrieval_acc": retrieval_acc,
                "has_tag": has_tag_int,
                "is_valid": is_valid_int,
            }
        )
        print(results[-1])

    out_df = pd.DataFrame(results)
    out_df.sort_values("idx", inplace=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
