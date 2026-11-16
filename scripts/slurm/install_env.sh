#!/usr/bin/env bash
# Run ONCE, from your already-activated Python environment (e.g. `conda activate
# <env>`), on a node with internet access (typically the cluster login node), before
# submitting any TSP generation jobs. Assumes `python` on PATH already has every
# dependency this repo needs (torch, transformers, accelerate, peft, pyarrow, etc. --
# see the `gpu`/`api` extras in pyproject.toml for the full package list); this script
# does NOT install/manage a Python environment itself. It just verifies that
# environment, then pre-downloads+verifies the two pinned model checkpoints (via
# warm_model_cache.sh), so the GPU compute nodes running the Slurm array in
# tsp_generate_array.sbatch can work fully offline (they run with HF_HUB_OFFLINE=1).
#
# Every Slurm job submitted via submit_tsp_generation.sh must activate the SAME
# environment. The simplest way: submit from this same activated shell -- sbatch
# inherits PATH/CONDA_* via --export=ALL (already the default in submit_tsp_generation.sh
# and both .sbatch files). If your cluster's job startup resets PATH regardless (some do,
# via login-shell init scripts), set CONDA_ENV=<env name> when submitting; both .sbatch
# files will explicitly `conda activate "$CONDA_ENV"` if it's set.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

echo "Using python: $(command -v python)"
python --version

# pyproject.toml requires >=3.10 (pervasive use of zip(..., strict=True), added in 3.10); fail
# fast with a clear message here instead of a confusing traceback deep in an import chain.
python -c "
import sys
if sys.version_info < (3, 10):
    sys.exit(
        f'This repo requires Python >=3.10 (pyproject.toml); the active environment has '
        f'{sys.version.split()[0]}. Activate a different conda env (ideally 3.12, matching '
        f'.python-version) before running this script.'
    )
"

export PYTHONHASHSEED=0
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
# Override with a shared/project storage path before running this script, e.g.:
#   export HF_HOME=/storage/huggingface
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo "Verifying torch actually imports ..."
python -c "import torch; print('torch', torch.__version__, '- import OK')"

echo "Pre-downloading and verifying pinned checkpoints into HF_HOME=$HF_HOME ..."
HF_HOME="$HF_HOME" bash "$ROOT/scripts/slurm/warm_model_cache.sh"

echo "Validating release inputs (no GPU/weights touched by this check) ..."
python -m dev.schema_llm.explore.tsp_generate --root "$ROOT" --check-only

cat <<EOF

Environment ready:
  python:  $(command -v python)
  HF_HOME: $HF_HOME

Export HF_HOME=$HF_HOME (or pass it through --export=) in every Slurm job so compute
nodes reuse this cache instead of re-downloading weights. Make sure every Slurm job
activates this SAME python environment (see the note at the top of this script).
EOF
