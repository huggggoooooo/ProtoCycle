#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
import json
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertTokenizer,
    BertModel,
    T5Tokenizer,
)

# 讓本腳本可以 import ProteinDT package
import sys

DEFAULT_PARQUET = "/path/to/ProtoCycle/data/proteinllm/desc2seq_agent_eval_clever_100.parquet"
DEFAULT_OUTPUT_CSV = "/path/to/ProtoCycle/baseline_results/ProteinDT.csv"
DEFAULT_PROTEINDT_REPO_ROOT = "/path/to/ProtoCycle/ACL_rebuttal/ProteinDT"
DEFAULT_PRETRAINED_FOLDER = (
    "/path/to/ProtoCycle/ACL_rebuttal/ProteinDT/checkpoint/ProteinDT/ProteinDT/"
    "ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10"
)
DEFAULT_STEP_04_FOLDER = (
    "/path/to/ProtoCycle/ACL_rebuttal/ProteinDT/checkpoint/ProteinDT/ProteinDT/"
    "ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10/"
    "step_04_T5Decoder_T5Base_lr_1e-4_hidden_16_e_10_unconditional_0.1"
)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def extract_ground_truth_from_reward_model(val: Any) -> str:
    rm: Dict[str, Any] = {}
    if isinstance(val, dict):
        rm = val
    else:
        text = str(val)
        try:
            rm = json.loads(text)
        except Exception:
            try:
                rm = ast.literal_eval(text)
            except Exception:
                rm = {}

    if not isinstance(rm, dict):
        rm = {}

    gt = rm.get("ground_truth") or rm.get("ground_truth_seq") or rm.get("gt") or rm.get("sequence") or ""
    return str(gt)


def extract_requirement_from_messages(messages: Any) -> str:
    # 兼容 parquet 里的 ndarray / list / JSON string
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return ""

    if not isinstance(messages, list):
        return ""

    # 取第一個 user 訊息
    user_text = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            user_text = str(m.get("content", ""))
            break

    if not user_text:
        return ""

    # 若有固定 marker，抽純 requirement
    marker = "The following text is the design requirement you must satisfy for this conversation."
    idx = user_text.find(marker)
    if idx == -1:
        return user_text.strip()

    sub = user_text[idx + len(marker):].lstrip()
    stop_markers = [
        "\n\nOverall agent protocol for each sample",
        "\n\nOverall agent protocol",
        "\n\nYou must:",
        "\n\nDesign stages:",
    ]
    end = len(sub)
    for sm in stop_markers:
        j = sub.find(sm)
        if j != -1 and j < end:
            end = j
    return sub[:end].strip()


def clean_pred_seq(seq: str) -> str:
    # ProteinDT decoder 可能輸出帶空格，統一清理
    seq = re.sub(r"[^A-Za-z]", "", str(seq or "")).upper()
    return seq


class TextListDataset(Dataset):
    def __init__(self, text_list: List[str], text_tokenizer, text_max_sequence_len: int):
        self.text_list = text_list
        self.text_tokenizer = text_tokenizer
        self.text_max_sequence_len = text_max_sequence_len

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        enc = self.text_tokenizer(
            text,
            truncation=True,
            max_length=self.text_max_sequence_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "text_sequence": text,
            "text_sequence_input_ids": enc.input_ids.squeeze(0),
            "text_sequence_attention_mask": enc.attention_mask.squeeze(0),
        }


@torch.no_grad()
def run_generation(
    dataloader,
    device,
    args,
    text_model,
    text2latent_model,
    facilitator_model,
    decoder_model,
    protein_tokenizer,
    protein_model,
    protein2latent_model,
    protein_decoder_tokenizer,
):
    all_text, all_pred = [], []

    for batch in dataloader:
        text_sequence = batch["text_sequence"]
        text_ids = batch["text_sequence_input_ids"].to(device)
        text_mask = batch["text_sequence_attention_mask"].to(device)
        all_text.extend(text_sequence)

        B = len(text_sequence)

        # 1) text -> latent
        text_out = text_model(text_ids, text_mask)
        text_repr = text_out["pooler_output"]
        text_repr = text2latent_model(text_repr)

        # 2) facilitator
        if args.use_facilitator:
            # 注意 ProteinDT 原碼方法名就是 inerence（拼寫如此）
            condition_repr = facilitator_model.inerence(text_repr=text_repr)
        else:
            condition_repr = text_repr

        # 3) repeat sampling
        repeated = condition_repr.unsqueeze(1).expand(-1, args.num_repeat, -1).reshape(-1, args.condition_dim)

        # 4) decode
        if args.decoder_distribution == "T5Decoder":
            if args.ar_generation_mode == "01":
                temperature, top_k, top_p = 1.0, 40, 0.9
                repetition_penalty, do_sample, num_beams = 1.0, True, 1
            else:
                temperature, top_k, top_p = 1.0, 40, 0.9
                repetition_penalty, do_sample, num_beams = 1.0, False, 1

            pred_ids = decoder_model.inference(
                condition=repeated,
                protein_seq_attention_mask=None,
                max_seq_len=args.protein_max_sequence_len,
                temperature=temperature,
                k=top_k,
                p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                do_sample=do_sample,
                num_beams=num_beams,
            )
        else:
            prot_mask = torch.zeros((B * args.num_repeat, args.protein_max_sequence_len), device=device)
            for row in prot_mask:
                valid_len = np.random.randint(300, args.protein_max_sequence_len + 1)
                row[:valid_len] = 1
            prot_mask = prot_mask.bool()

            pred_logits = decoder_model.inference(
                condition=repeated,
                protein_seq_attention_mask=prot_mask,
                max_seq_len=args.protein_max_sequence_len,
                mode=args.sde_sampling_mode,
            )
            pred_ids = torch.argmax(pred_logits, dim=-1)

            # truncate at pad token
            for seq_ids in pred_ids:
                pad_idx = None
                for i, token_id in enumerate(seq_ids):
                    if token_id.item() == protein_decoder_tokenizer.pad_token_id:
                        pad_idx = i
                        break
                if pad_idx is not None:
                    seq_ids[pad_idx:] = protein_decoder_tokenizer.pad_token_id

        # 5) decode ids -> seq strings
        seq_list = protein_decoder_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        seq_list = [s.replace(" ", "") for s in seq_list]

        # 6) 用 ProteinCLIP latent 相似度選每個 text 的最佳 seq
        cleaned_for_encoder = []
        for s in seq_list:
            s = re.sub(r"[UZOB]", "X", s)
            s = " ".join(s)
            cleaned_for_encoder.append(s)

        p_enc = protein_tokenizer(
            cleaned_for_encoder,
            truncation=True,
            max_length=args.protein_max_sequence_len,
            padding="max_length",
            return_tensors="pt",
        )
        p_ids = p_enc.input_ids.to(device)
        p_mask = p_enc.attention_mask.to(device)

        p_out = protein_model(p_ids, p_mask)
        p_repr = protein2latent_model(p_out["pooler_output"])

        for i in range(B):
            st, ed = i * args.num_repeat, (i + 1) * args.num_repeat
            sim = torch.matmul(text_repr[i], p_repr[st:ed].transpose(0, 1))
            best_local = torch.argmax(sim).item()
            best_idx = st + best_local
            all_pred.append(seq_list[best_idx])

    return all_text, all_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)

    parser.add_argument("--proteindt_repo_root", type=str, default=DEFAULT_PROTEINDT_REPO_ROOT,
                        help="例如 /.../ACL_rebuttal/ProteinDT")
    parser.add_argument("--pretrained_folder", type=str, default=DEFAULT_PRETRAINED_FOLDER,
                        help="例如 .../ProtBERT_BFD-...-epoch-10")
    parser.add_argument("--step_04_folder", type=str, default=DEFAULT_STEP_04_FOLDER,
                        help="例如 .../step_04_T5Decoder_T5Base_lr_1e-4_hidden_16_e_10_unconditional_0.1")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_repeat", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None, help="可选，仅推理前 N 条样本用于快速测试")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD",
                        choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)

    parser.add_argument("--ssl_emb_dim", type=int, default=256)
    parser.add_argument("--condition_dim", type=int, default=256)

    parser.add_argument("--decoder_distribution", type=str, default="T5Decoder",
                        choices=["T5Decoder", "MultinomialDiffusion"])
    parser.add_argument("--score_network_type", type=str, default="T5Base")
    parser.add_argument("--ar_generation_mode", type=str, default="01", choices=["01", "02"])
    parser.add_argument("--sde_sampling_mode", type=str, default="simplified", choices=["simplified", "weighted"])

    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--beta_min", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=30)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)

    parser.add_argument("--use_facilitator", action="store_true")
    parser.add_argument("--no_use_facilitator", dest="use_facilitator", action="store_false")
    parser.set_defaults(use_facilitator=True)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Import only required modules directly.
    # Avoid importing ProteinDT.models.__init__, which pulls in BindingModel/CDConv
    # and hard-requires torch_scatter for tasks we do not use in text2protein inference.
    sys.path.insert(0, args.proteindt_repo_root)
    from ProteinDT.models.model_MultinomialDiffusionDecoder import MultinomialDiffusion
    from ProteinDT.models.model_TransformerDecoder import T5Decoder
    from ProteinDT.models.model_GaussianFacilitator import GaussianFacilitatorModel

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 讀 parquet
    df = pd.read_parquet(args.parquet)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()

    if "requirement" in df.columns:
        requirement_list = [str(x) for x in df["requirement"].tolist()]
    elif "messages" in df.columns:
        requirement_list = [extract_requirement_from_messages(x) for x in df["messages"].tolist()]
    else:
        raise ValueError("parquet 不含 requirement 或 messages 欄位，無法取得文字需求")

    ground_truth_list = []
    if "reward_model" in df.columns:
        ground_truth_list = [extract_ground_truth_from_reward_model(x) for x in df["reward_model"].tolist()]
    else:
        ground_truth_list = [""] * len(df)

    # 載模型
    if args.protein_backbone_model == "ProtBERT":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert")
        protein_dim = 1024
    else:
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        protein_dim = 1024

    text_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    text_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    text_dim = 768

    # 載 pretrain heads
    protein2latent = nn.Linear(protein_dim, args.ssl_emb_dim)
    text2latent = nn.Linear(text_dim, args.ssl_emb_dim)
    facilitator = GaussianFacilitatorModel(args.ssl_emb_dim)

    def _load_state(m, p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"checkpoint 不存在: {p}")
        state_dict = torch.load(p, map_location="cpu")
        # HF/transformers minor version mismatch may introduce this buffer key.
        # It is safe to ignore for inference.
        if isinstance(state_dict, dict) and "embeddings.position_ids" in state_dict:
            state_dict.pop("embeddings.position_ids", None)

        missing, unexpected = m.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys while loading {p}: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys while loading {p}: {unexpected}")

    _load_state(protein_model, os.path.join(args.pretrained_folder, "protein_model.pth"))
    _load_state(text_model, os.path.join(args.pretrained_folder, "text_model.pth"))
    _load_state(protein2latent, os.path.join(args.pretrained_folder, "protein2latent_model.pth"))
    _load_state(text2latent, os.path.join(args.pretrained_folder, "text2latent_model.pth"))
    _load_state(facilitator, os.path.join(args.pretrained_folder, "step_03_Gaussian_10", "facilitator_distribution_model.pth"))

    # decoder
    if args.decoder_distribution == "T5Decoder":
        protein_decoder_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        decoder = T5Decoder(
            hidden_dim=args.condition_dim,
            tokenizer=protein_decoder_tokenizer,
            T5_model=args.score_network_type,
        )
    else:
        protein_decoder_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        decoder = MultinomialDiffusion(
            hidden_dim=args.hidden_dim,
            condition_dim=args.condition_dim,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            num_diffusion_timesteps=args.num_diffusion_timesteps,
            mask_id=4,
            num_classes=protein_decoder_tokenizer.vocab_size,
            score_network_type=args.score_network_type,
        )

    _load_state(decoder, os.path.join(args.step_04_folder, "decoder_distribution_model.pth"))

    # to device
    for m in [protein_model, text_model, protein2latent, text2latent, facilitator, decoder]:
        m.to(device)
        m.eval()

    dataset = TextListDataset(requirement_list, text_tokenizer, args.text_max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    _, pred_list = run_generation(
        dataloader=dataloader,
        device=device,
        args=args,
        text_model=text_model,
        text2latent_model=text2latent,
        facilitator_model=facilitator,
        decoder_model=decoder,
        protein_tokenizer=protein_tokenizer,
        protein_model=protein_model,
        protein2latent_model=protein2latent,
        protein_decoder_tokenizer=protein_decoder_tokenizer,
    )

    rows = []
    for i, (req, gt, pred) in enumerate(zip(requirement_list, ground_truth_list, pred_list)):
        pred_seq = clean_pred_seq(pred)
        rows.append({
            "row_idx": i,
            "jsonl_index": i,
            "requirement_parquet": req,
            "requirement_from_jsonl": req,
            "ground_truth_seq": gt,
            "pred_seq": pred_seq,
            "has_tag": 1 if len(pred_seq) > 0 else 0,
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"[DONE] wrote {len(out_df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()