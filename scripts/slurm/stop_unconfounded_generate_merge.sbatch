#!/usr/bin/env bash
# Final merge step: submit with --dependency=afterok:<array job id> (done automatically by
# submit_stop_unconfounded.sh). Runs the injector with NO --task-indices, so it iterates the
# full population; every shard was already written by stop_unconfounded_generate_array.sbatch,
# so this only validates/loads cached rows (no model weights loaded) and builds OUT/pack/.
#SBATCH --job-name=stop-unconf-merge
#SBATCH --output=logs/stop-unconf-merge_%j.out
#SBATCH --error=logs/stop-unconf-merge_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail

ROOT="${STOP_ROOT:?set STOP_ROOT to the repo checkout path}"
OUT="${STOP_OUT:?set STOP_OUT to the shared output directory}"
ARMS="${STOP_ARMS:-stop shuffled}"
DOSES="${STOP_DOSES:-0.25 0.5 1.0}"
BUDGET="${STOP_BUDGET:-2048}"

cd "$ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# shellcheck disable=SC2086
python -m dev.schema_llm.explore.stop_steering.unconfounded_generate \
  --root "$ROOT" \
  --out "$OUT" \
  --arms $ARMS \
  --doses $DOSES \
  --budget "$BUDGET" \
  --device cuda:0

echo "Pack ready: $OUT/pack"
