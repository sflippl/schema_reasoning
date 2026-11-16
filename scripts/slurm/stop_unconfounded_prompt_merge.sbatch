#!/usr/bin/env bash
# Assemble the item shards after stop_unconfounded_prompt_array.sbatch completes.
#SBATCH --job-name=stop-prompt-merge
#SBATCH --output=logs/stop-prompt-merge_%j.out
#SBATCH --error=logs/stop-prompt-merge_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail

ROOT="${STOP_ROOT:?set STOP_ROOT to the repo checkout path}"
OUT="${STOP_PROMPT_OUT:?set STOP_PROMPT_OUT to the shared output directory}"
BUDGET="${STOP_PROMPT_BUDGET:--1}"
RUN_OFFSETS="${STOP_PROMPT_RUN_OFFSETS:-0 30}"

cd "$ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

# shellcheck disable=SC2086
python -m dev.schema_llm.explore.stop_steering.prompt_injection \
  --root "$ROOT" \
  --out "$OUT" \
  --budget "$BUDGET" \
  --run-offsets $RUN_OFFSETS

echo "Pack ready: $OUT/pack"
