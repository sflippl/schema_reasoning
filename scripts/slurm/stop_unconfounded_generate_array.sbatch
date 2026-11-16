#!/usr/bin/env bash
# Slurm array job: each array task injects into an explicit, contiguous block of the 84-item
# injectable population (``unconfounded_data.load_population``) -- sharding only ever splits
# items, never arms/doses; every array task still runs the FULL arm/dose plan (control + every
# requested arm x dose) for its own items -- and writes content-addressed shard files under
# $STOP_OUT/shards/. It does not build the final pack -- run
# stop_unconfounded_generate_merge.sbatch once every array task has finished for that.
#
# Requires stop_unconfounded_capture.sbatch to have completed first if STOP_ARMS includes
# "shuffled" (the default) -- that job builds the shuffled null this one reads.
#
# --array is the actual sharding knob: it must be a contiguous 0-based range (0-19 below means
# 20 array tasks). Edit it directly for your desired job count, or override from the command
# line / submit_stop_unconfounded.sh with `sbatch --array=0-N`, which takes precedence over
# this line. Each array task's $SLURM_ARRAY_TASK_ID picks a block of
# `ceil(population_size / array_size)` CONSECUTIVE population-row positions below (e.g. 84
# items / 20 array tasks = a literal, contiguous "--task-indices 0-4", "5-9", ... per job -- no
# interleaving/striding).
#SBATCH --array=0-19
#SBATCH --job-name=stop-unconf-gen
#SBATCH --output=logs/stop-unconf-gen_%A_%a.out
#SBATCH --error=logs/stop-unconf-gen_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

ROOT="${STOP_ROOT:?set STOP_ROOT to the repo checkout path}"
OUT="${STOP_OUT:?set STOP_OUT to the shared output directory}"
ARMS="${STOP_ARMS:-stop shuffled}"
DOSES="${STOP_DOSES:-0.25 0.5 1.0}"
BUDGET="${STOP_BUDGET:-2048}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array task (sbatch --array=0-N)}"
ARRAY_SIZE="${SLURM_ARRAY_TASK_COUNT:-${STOP_ARRAY_SIZE:?set STOP_ARRAY_SIZE if your Slurm version does not export SLURM_ARRAY_TASK_COUNT}}"

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

TOTAL_ITEMS=$(python -m dev.schema_llm.explore.stop_steering.unconfounded_generate \
  --root "$ROOT" --population-size)
CHUNK=$(((TOTAL_ITEMS + ARRAY_SIZE - 1) / ARRAY_SIZE))
START=$((ARRAY_INDEX * CHUNK))
if [ "$START" -ge "$TOTAL_ITEMS" ]; then
  echo "Array task $ARRAY_INDEX has no items to inject (array size $ARRAY_SIZE > $TOTAL_ITEMS items); nothing to do."
  exit 0
fi
END=$((START + CHUNK - 1))
if [ "$END" -ge "$TOTAL_ITEMS" ]; then
  END=$((TOTAL_ITEMS - 1))
fi
TASK_INDICES="$START-$END"
echo "Array task $ARRAY_INDEX/$ARRAY_SIZE injecting population positions $TASK_INDICES (of $TOTAL_ITEMS), arms=[$ARMS] doses=[$DOSES]."

# shellcheck disable=SC2086
python -m dev.schema_llm.explore.stop_steering.unconfounded_generate \
  --root "$ROOT" \
  --out "$OUT" \
  --arms $ARMS \
  --doses $DOSES \
  --budget "$BUDGET" \
  --device cuda:0 \
  --task-indices "$TASK_INDICES"
