#!/usr/bin/env bash
# Run ONCE on a node with internet access (typically the cluster login node) before
# submitting any TSP generation jobs. It builds the locked `uv` environment under
# .venv/ and pre-downloads+verifies the two pinned model checkpoints (via
# warm_model_cache.sh), so the GPU compute nodes running the Slurm array in
# tsp_generate_array.sbatch can work fully offline (they run with HF_HUB_OFFLINE=1).
#
# Assumes $ROOT (this repo checkout) lives on storage shared between the login node and
# compute nodes (e.g. NFS/Lustre home or project space) -- `uv sync` is NOT re-run per
# job; every array task reuses this same .venv via `uv run --no-sync`.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# The repo pins 3.12 via .python-version (torch's locked version has no wheel for very
# new interpreters -- e.g. 3.15 -- so `requires-python = ">=3.11"`'s floor alone isn't
# enough). Make sure it's actually installed rather than relying on uv finding one.
echo "Ensuring Python 3.12 is available ..."
uv python install 3.12

export PYTHONHASHSEED=0
# Override with a shared/project storage path before running this script, e.g.:
#   export HF_HOME=/storage/huggingface
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo "uv version: $(uv --version)"
echo "Syncing locked environment (extras: gpu, api) into $ROOT/.venv ..."
# Requires internet access: pulls torch/transformers/accelerate/peft from PyPI and the
# baukit dependency from GitHub (git+https://github.com/davidbau/baukit).
uv sync --locked --extra gpu --extra api

echo "Pre-downloading and verifying pinned checkpoints into HF_HOME=$HF_HOME ..."
HF_HOME="$HF_HOME" bash "$ROOT/scripts/slurm/warm_model_cache.sh"

echo "Validating release inputs (no GPU/weights touched by this check) ..."
uv run --no-sync python -m dev.schema_llm.explore.tsp_generate --root "$ROOT" --check-only

cat <<EOF

Environment ready:
  venv:    $ROOT/.venv
  HF_HOME: $HF_HOME

Export HF_HOME=$HF_HOME (or pass it through --export=) in every Slurm job so compute
nodes reuse this cache instead of re-downloading weights.
EOF
