# ProtoCycle

**Agentic Protein Design with Reinforcement Learning**

ProtoCycle trains LLMs to perform multi-step, tool-augmented protein design via
GRPO-TCR (Group Relative Policy Optimization with Tool-Call Rewards). Given a
natural-language design requirement, the model iteratively invokes specialized
biology tools (scaffold retrieval, constraint building, ESM-based refinement,
ProTrek scoring) and outputs a final amino-acid sequence.

> This repository contains our protein-specific recipe, tools, reward and
> evaluation pipeline. It is built on top of
> [Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL) /
> [VeRL](https://github.com/volcengine/verl), which provides the underlying
> agentic RL training framework.

## Repository Layout

```
protocycle/
├── infer_tools.py                # Multi-turn tool-calling inference entry point
├── infer_tools.sh                # End-to-end inference + metrics pipeline
├── recipe/protein/               # Training scripts
│   ├── reward.py                 # GRPO-TCR reward + custom dataset class
│   ├── protein_dataset.py        # Multi-turn SFT dataset adapter
│   ├── tool_config.yaml          # 10 tool schemas for agentic rollout
│   ├── qwen2_7b_sft.sh           # Cold-start SFT recipe
│   └── grpo_tcr_qwen2_7b.sh      # Agentic RL (GRPO-TCR) recipe
├── verl/
│   ├── tools/
│   │   ├── protein_tools.py      # Agentic tool entry (registered to VeRL)
│   │   └── pfam/                 # Biology tool implementations
│   │       ├── pipline_new.py    # Main AgentRuntime
│   │       ├── function2seq.py   # Stage-1: function -> scaffolds
│   │       ├── pathway2seq.py
│   │       ├── domain2seq.py
│   │       ├── go2seq.py
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
│   ├── extract_answer.py
│   ├── compute_metrics.py
│   ├── compute_metrics_multi_gpu.py
│   ├── compute_chai_scores.py
│   ├── summarize_metrics.py
│   ├── compute_metrics.sh
│   └── example.fasta
├── data/
│   └── proteinllm/
│       └── protein_eval_30.csv   # Small sample evaluation set
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
cp -r  ProtoCycle/recipe/protein                        Open-AgentRL/recipe/
cp     ProtoCycle/verl/tools/protein_tools.py           Open-AgentRL/verl/tools/
cp -r  ProtoCycle/verl/tools/pfam                       Open-AgentRL/verl/tools/
cp     ProtoCycle/verl/workers/reward_manager/protein.py \
                                                        Open-AgentRL/verl/workers/reward_manager/
cp -r  ProtoCycle/eval_tools                            Open-AgentRL/
cp -r  ProtoCycle/data/proteinllm                       Open-AgentRL/data/
cp     ProtoCycle/infer_tools.py  ProtoCycle/infer_tools.sh  Open-AgentRL/
```

All paths inside this repo are **relative to the repository root** and are
resolved automatically by the scripts. External resources (conda envs, model
weights, databases) are referenced via environment variables; see the next
section.

## Configuration

All scripts read the following environment variables to locate external
resources. Export them once in your shell (or in a wrapper script):

| Variable | Meaning |
|----------|---------|
| `CONDA_ROOT` | Root of your miniconda install, e.g. `/home/user/miniconda3` |
| `MODEL_DIR` | Absolute path to a base or RL checkpoint |
| `MODEL_PATH` | Base-model HF snapshot (used by `qwen2_7b_sft.sh` / `grpo_tcr_qwen2_7b.sh`) |
| `PROTREK_ENV_PYTHON` | Python in the `protrek` conda env, e.g. `$CONDA_ROOT/envs/protrek/bin/python` |
| `PROTREK_35M_DIR`, `PROTREK_650M_DIR` | Local paths to the ProTrek checkpoints |
| `ESM_MODEL_PATH` | Local HF snapshot of `facebook/esm2_t36_3B_UR50D` |
| `CHAI_LAB_ROOT` | Path to a local `chai-lab` checkout (optional, only for Chai-1 metrics) |

Any remaining `/path/to/...` literals left in the code are external resources
— either download them and export the matching env var, or pass them as CLI
arguments (most scripts accept flags that override the defaults).

### External Assets

| Asset | Purpose | Source |
|-------|---------|--------|
| ProTrek 35M / 650M checkpoints | Stage-3 scoring | [ProTrek repo](https://github.com/westlake-repl/ProTrek) |
| ESM2 3B (`facebook/esm2_t36_3B_UR50D`) | Stage-3 inpainting | HuggingFace |
| Pfam-A.hmm, Pfam-A.seed | Profile HMM scans | [Pfam FTP](https://www.ebi.ac.uk/interpro/) |
| Foldseek binary | Structure search (optional) | [Foldseek](https://github.com/steineggerlab/foldseek) |
| PROSITE database (`prosite.dat`) | Motif lookup | [PROSITE](https://prosite.expasy.org/) |
| Chai-1 | Optional structure-prediction metrics | [chai-lab](https://github.com/chaidiscovery/chai-lab) |

### Model Checkpoints

Our trained model weights are hosted on Hugging Face:

| Model | Description | Link |
|-------|-------------|------|
| ProtoCycle-7B-SFT | Cold-start SFT checkpoint (Qwen2.5-7B base) | [Huggggooo/ProtoCycle-7B-SFT](https://huggingface.co/Huggggooo/ProtoCycle-7B-SFT) |
| ProtoCycle-7B | GRPO-TCR RL checkpoint | [Huggggooo/ProtoCycle-7B](https://huggingface.co/Huggggooo/ProtoCycle-7B) |

## Training

### Cold-Start SFT

```bash
export MODEL_PATH=/abs/path/to/Qwen2.5-7B-Instruct
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
export MODEL_PATH=$SAVE_PATH/global_step_XXX/huggingface   # SFT checkpoint
bash recipe/protein/grpo_tcr_qwen2_7b.sh
```

We trained on one 8×A100 (80GB) node with batch size 64, 3 epochs.

## Evaluation

A small sample evaluation set lives at `data/proteinllm/protein_eval_30.csv`.
The end-to-end evaluation pipeline has three stages:

1. **Inference** (`infer_tools.py`) — runs multi-turn tool-augmented rollout
   with the trained model and saves each trajectory to JSONL.
2. **Answer extraction** (`eval_tools/extract_answer.py`) — parses the final
   `<answer>` segment from each trajectory into a CSV.
3. **Metric computation** (`eval_tools/compute_metrics_multi_gpu.py`) —
   computes ESM pseudo-perplexity, pTM / pLDDT (Chai-1, optional),
   ProTrek similarity, and retrieval accuracy.

The three stages can be run individually or end-to-end through
`infer_tools.sh`:

```bash
export CONDA_ROOT=/abs/path/to/miniconda3
export MODEL_DIR=/abs/path/to/ProtoCycle-7B          # or any HF-format checkpoint
export MODEL_NAME=ProtoCycle-7B                      # used in output filenames
bash infer_tools.sh
```

This will create `baseline_results/${MODEL_NAME}.jsonl`,
`baseline_results/${MODEL_NAME}.csv`, and
`baseline_results/metrics/${MODEL_NAME}_metrics.csv`.

You can also invoke metrics independently:

```bash
bash eval_tools/compute_metrics.sh <input_csv> <output_metrics_csv>
```

Optional flags for `compute_metrics_multi_gpu.py` include `--skip_chai`,
`--skip_protrek`, `--skip_retrieval`, and `--use_evollama`.

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
