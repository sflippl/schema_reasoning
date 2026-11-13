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
#
# `--locked` enforces the exact frozen release versions (the common/tested case). It can
# fail on older HPC glibc (e.g. torch wheels built only for manylinux_2_28 / glibc>=2.28,
# while an older cluster node reports manylinux_2_17) -- a genuine platform limitation of
# the frozen lockfile, not something a Python version pin controls. Fall back to letting
# uv re-resolve (still respecting pyproject.toml's constraints, just not the exact locked
# versions) so this platform gets wheels it can actually install.
if ! uv sync --locked --extra gpu --extra api; then
  echo
  echo "Locked sync failed (commonly a wheel/platform mismatch -- see the error above for" >&2
  echo "a 'manylinux_2_NN'/glibc hint). Falling back to an unlocked resolve so uv can pick" >&2
  echo "versions with wheels for THIS platform; this updates uv.lock locally and may differ" >&2
  echo "from the frozen release versions." >&2
  uv sync --extra gpu --extra api
fi

echo "Verifying torch actually imports ..."
uv run --no-sync python -c "import torch; print('torch', torch.__version__, '- import OK')"

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
