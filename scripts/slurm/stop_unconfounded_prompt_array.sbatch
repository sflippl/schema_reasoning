#!/usr/bin/env bash
# Each array task runs all prompt-insertion conditions and both T_solved offsets for its
# contiguous block of the feasible, judge-solved D1 population. A separate merge job assembles
# the completed item shards.
#SBATCH --array=0-19
#SBATCH --job-name=stop-prompt
#SBATCH --output=logs/stop-prompt_%A_%a.out
#SBATCH --error=logs/stop-prompt_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

ROOT="${STOP_ROOT:?set STOP_ROOT to the repo checkout path}"
OUT="${STOP_PROMPT_OUT:?set STOP_PROMPT_OUT to the shared output directory}"
BUDGET="${STOP_PROMPT_BUDGET:--1}"
RUN_OFFSETS="${STOP_PROMPT_RUN_OFFSETS:-0 30}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array task}"
ARRAY_SIZE="${SLURM_ARRAY_TASK_COUNT:-${STOP_PROMPT_ARRAY_SIZE:?set STOP_PROMPT_ARRAY_SIZE if unavailable}}"

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

TOTAL_ITEMS=$(python -m dev.schema_llm.explore.stop_steering.prompt_injection \
  --root "$ROOT" --run-offsets $RUN_OFFSETS --population-size)
CHUNK=$(((TOTAL_ITEMS + ARRAY_SIZE - 1) / ARRAY_SIZE))
START=$((ARRAY_INDEX * CHUNK))
if [ "$START" -ge "$TOTAL_ITEMS" ]; then
  echo "Array task $ARRAY_INDEX has no items (array size $ARRAY_SIZE > $TOTAL_ITEMS); nothing to do."
  exit 0
fi
END=$((START + CHUNK - 1))
if [ "$END" -ge "$TOTAL_ITEMS" ]; then
  END=$((TOTAL_ITEMS - 1))
fi
TASK_INDICES="$START-$END"
if [ "$BUDGET" -le 0 ]; then
  BUDGET_LABEL="remaining-to-8192-cap"
else
  BUDGET_LABEL="$BUDGET generated tokens"
fi
echo "Array task $ARRAY_INDEX/$ARRAY_SIZE: items $TASK_INDICES of $TOTAL_ITEMS; offsets=[$RUN_OFFSETS], budget=$BUDGET_LABEL."

# shellcheck disable=SC2086
python -m dev.schema_llm.explore.stop_steering.prompt_injection \
  --root "$ROOT" \
  --out "$OUT" \
  --budget "$BUDGET" \
  --run-offsets $RUN_OFFSETS \
  --device cuda:0 \
  --task-indices "$TASK_INDICES"
