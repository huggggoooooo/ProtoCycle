#!/usr/bin/env bash
# Launches the ProTrek-35M scoring service. Requires CONDA_ROOT to point to your
# miniconda install that has a "protrek" env.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:?Please export CONDA_ROOT=/absolute/path/to/miniconda3}"
PROTREK_ENV="${PROTREK_ENV:-${CONDA_ROOT}/envs/protrek}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${PROTREK_ENV}"
cd "${SCRIPT_DIR}"
python protrek_35m_api.py
