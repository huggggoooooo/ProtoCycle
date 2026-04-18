# ProtoCycle

**Agentic Protein Design with Reinforcement Learning**

ProtoCycle trains LLMs to perform multi-step, tool-augmented protein design via
GRPO-TCR (Group Relative Policy Optimization with Tool-Call Rewards). Given a
natural-language design requirement, the model iteratively invokes specialized
biology tools (scaffold retrieval, constraint building, ESM-based refinement,
ProTrek scoring) and outputs a final amino-acid sequence.

> This repository contains our protein-specific recipe, tools, and reward.
> It is built on top of
> [Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL) /
> [VeRL](https://github.com/volcengine/verl), which provides the underlying
> agentic RL training framework.

## Repository Layout

```
protocycle/
├── recipe/protein/               # Training & evaluation scripts
│   ├── reward.py                 # GRPO-TCR reward + custom dataset class
│   ├── protein_dataset.py        # Multi-turn SFT dataset adapter
│   ├── tool_config.yaml          # 10 tool schemas for agentic rollout
│   ├── grpo_tcr_qwen2_7b.sh      # RL (GRPO-TCR) main recipe
│   ├── qwen2_7b_sft.sh           # Cold-start SFT recipe
│   └── ...                       # Llama variants, 4-GPU variants, infer/eval
├── verl/
│   ├── tools/
│   │   ├── protein_tools.py      # Agentic tool entry (registered to VeRL)
│   │   └── pfam/                 # Biology tool implementations
│   │       ├── pipline_new.py    # Main AgentRuntime
│   │       ├── function2seq.py   # Stage-1: function -> scaffolds
│   │       ├── pathway2seq.py    # Stage-1: pathway -> scaffolds
│   │       ├── domain2seq.py     # Stage-1: domain -> scaffolds
│   │       ├── go2seq.py         # Stage-1: GO term -> scaffolds
│   │       ├── dna_binding2seq.py
│   │       ├── cofactor2constraints.py  # Stage-2
│   │       ├── motif2constraints.py     # Stage-2
│   │       ├── signal2constraints.py    # Stage-2
│   │       ├── esm/esm_constrain.py     # Stage-3: ESM inpainting
│   │       └── ProTrek/                 # Stage-3: ProTrek scoring (code only)
│   └── workers/
│       └── reward_manager/
│           └── protein.py        # ProteinRewardManager (VeRL plugin)
├── eval_tools/                   # Metric / scoring utilities
├── data/
│   └── proteinllm/
│       └── protein_eval_30.csv   # Sample evaluation set
├── LICENSE                       # Apache-2.0
└── README.md
```

## Agent Protocol (Three Stages)

During rollout the agent emits `<think>` / `<plan>` / `<tool_call>` per step,
and finally an `<answer>` containing one amino-acid sequence.

- **Stage 1 — Scaffold retrieval**: `function2seq`, `pathway2seq`,
  `domain2seq`, `go2seq`, `dna_binding2seq`
- **Stage 2 — Constraint injection**: `cofactor2constraints`,
  `motif2constraints`, `signal2constraints`
- **Stage 3 — Refinement & scoring**: `esm_inpaint`, `get_score`

The reward combines:
- Protocol compliance (`<think>` / `<plan>` / `<answer>` format)
- Tool-call quality (Stage-1 success, ProTrek score after each call)
- Outcome signal (global best ProTrek score)
- Efficiency (penalty for excessive turns)

See `recipe/protein/reward.py` for the exact scoring function.

## Installation

ProtoCycle depends on the VeRL / Open-AgentRL framework.

```bash
# 1) Clone and install the base framework
git clone https://github.com/Gen-Verse/Open-AgentRL.git
cd Open-AgentRL
conda create -n OpenAgentRL python=3.11 -y
conda activate OpenAgentRL
bash scripts/install_vllm_sglang_mcore.sh
pip install -e .[vllm]

# 2) Overlay this repository on top of Open-AgentRL
cd ..
git clone git@github.com:huggggoooooo/ProtoCycle.git
# Merge protein-specific code into Open-AgentRL
cp -r ProtoCycle/recipe/protein         Open-AgentRL/recipe/
cp    ProtoCycle/verl/tools/protein_tools.py         Open-AgentRL/verl/tools/
cp -r ProtoCycle/verl/tools/pfam                     Open-AgentRL/verl/tools/
cp    ProtoCycle/verl/workers/reward_manager/protein.py \
                                                     Open-AgentRL/verl/workers/reward_manager/
cp -r ProtoCycle/eval_tools                          Open-AgentRL/
cp -r ProtoCycle/data/proteinllm                     Open-AgentRL/data/
```

### External Assets

The biology tools require several large external resources. Download them
separately and place them under the paths below (or edit `tool_config.yaml` /
the Python constants to point elsewhere):

| Asset | Purpose | Source |
|-------|---------|--------|
| ProTrek 35M / 650M checkpoints | Stage-3 scoring | [ProTrek repo](https://github.com/westlake-repl/ProTrek) |
| ESM2 3B (`facebook/esm2_t36_3B_UR50D`) | Stage-3 inpainting | HuggingFace |
| Pfam-A.hmm, Pfam-A.seed | Profile HMM scans | [Pfam FTP](https://www.ebi.ac.uk/interpro/) |
| Foldseek binary | Structure search (optional) | [Foldseek](https://github.com/steineggerlab/foldseek) |
| PROSITE database (`prosite.dat`) | Motif lookup | [PROSITE](https://prosite.expasy.org/) |

### Model Checkpoints

Our trained model weights are hosted on Hugging Face:

| Model | Description | Link |
|-------|-------------|------|
| Qwen2.5-7B-Protein-SFT | Cold-start SFT checkpoint | _TODO: add link_ |
| ProtoCycle-7B | GRPO-TCR RL checkpoint | _TODO: add link_ |

## Training

Before running any script, edit the paths at the top of the shell file —
placeholders such as `/path/to/ProtoCycle`, `/path/to/models/...`,
`/path/to/miniconda3/envs/...` must be replaced with your actual locations.

### Cold-Start SFT

```bash
bash recipe/protein/qwen2_7b_sft.sh
```

After SFT, merge the FSDP checkpoint:

```bash
python3 -m verl.model_merger merge --backend fsdp \
    --local_dir $SAVE_PATH/global_step_XXX \
    --target_dir $SAVE_PATH/global_step_XXX/huggingface
```

### Agentic RL (GRPO-TCR)

```bash
bash recipe/protein/grpo_tcr_qwen2_7b.sh
```

We trained on one 8×A100 (80GB) node with batch size 64, 3 epochs.

## Evaluation

A small demo evaluation set is provided:

```bash
ls data/proteinllm/protein_eval_30.csv
```

Run inference then compute metrics:

```bash
bash recipe/protein/qwen2_7b_infer.sh
bash eval_tools/compute_metrics.sh
```

Scoring uses ProTrek similarity and, optionally, Chai-1 structure prediction
(`eval_tools/compute_chai_scores.py`).

## License

This repository is released under the **Apache License 2.0** — see
[LICENSE](LICENSE). This is consistent with the upstream VeRL / Open-AgentRL
projects that ProtoCycle builds upon.

## Acknowledgements

ProtoCycle stands on the shoulders of several excellent open-source projects.
We thank the authors and contributors of:

- [VeRL](https://github.com/volcengine/verl) (ByteDance Seed) — the underlying
  PPO/GRPO training framework.
- [Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL) (Gen-Verse) — the
  agentic RL recipe infrastructure we built on.
- [ProTrek](https://github.com/westlake-repl/ProTrek) — the scoring model used
  throughout Stage-3.
- [ESM](https://github.com/facebookresearch/esm) (Meta FAIR) — the protein
  language model used for inpainting-based refinement.
- [Chai-1](https://github.com/chaidiscovery/chai-lab) — optional downstream
  structure scoring.
- The [Pfam](https://www.ebi.ac.uk/interpro/) / [PROSITE](https://prosite.expasy.org/)
  / [ELM](http://elm.eu.org/) resources for family, motif, and linear motif
  annotations.

<!-- ## Citation

If you find this work useful, please cite:

```bibtex
@misc{protocycle2026,
  title  = {ProtoCycle: Agentic Protein Design with Reinforcement Learning},
  author = {TODO: add authors},
  year   = {2026},
  note   = {TODO: arXiv / venue info}
}
``` -->
