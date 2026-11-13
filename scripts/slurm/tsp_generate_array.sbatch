#!/usr/bin/env bash
# Slurm array job: each array task generates an explicit, contiguous block of TSP tasks
# for BOTH pinned models (phi4mr, qwen34b) -- sharding only ever splits tasks, never
# models -- and writes content-addressed shard files under $TSP_OUT/shards/. It does not
# build the final pack -- run tsp_generate_merge.sbatch (or
# scripts/generate_tsp_responses.sh with no --task-indices) once every array task has
# finished for that.
#
# --array is the actual sharding knob: it must be a contiguous 0-based range (0-19 below
# means 20 array tasks). Edit it directly for your desired job count, or override from
# the command line / submit_tsp_generation.sh with `sbatch --array=0-N`, which takes
# precedence over this line. Each array task's $SLURM_ARRAY_TASK_ID picks a block of
# `ceil(total_tasks / array_size)` CONSECUTIVE task-file positions below (e.g. 100 tasks /
# 20 array tasks = a literal, contiguous "--task-indices 0-4", "5-9", ... per job -- no
# interleaving/striding, exactly a plain list of indices per job).
#SBATCH --array=0-19
#SBATCH --job-name=tsp-gen
#SBATCH --output=logs/tsp-gen_%A_%a.out
#SBATCH --error=logs/tsp-gen_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

ROOT="${TSP_ROOT:?set TSP_ROOT to the repo checkout path}"
OUT="${TSP_OUT:?set TSP_OUT to the shared output directory}"
TASKS="${TSP_TASKS:?set TSP_TASKS to the tasks JSONL path}"
MAX_NEW_TOKENS="${TSP_MAX_NEW_TOKENS:-32768}"
SEED="${TSP_SEED:-0}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array task (sbatch --array=0-N)}"
ARRAY_SIZE="${SLURM_ARRAY_TASK_COUNT:-${TSP_ARRAY_SIZE:?set TSP_ARRAY_SIZE if your Slurm version does not export SLURM_ARRAY_TASK_COUNT}}"

cd "$ROOT"

# Optional safety net: if your cluster's job startup doesn't inherit PATH/CONDA_* from
# the submitting shell (submit_tsp_generation.sh exports --export=ALL, which usually
# covers this), set CONDA_ENV=<env name> when submitting to explicitly reactivate it here.
if [ -n "${CONDA_ENV:-}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

# Non-empty lines, matching tsp_generate.load_tasks's own `if line.strip()` filter.
TOTAL_TASKS=$(grep -c '.' "$TASKS")
CHUNK=$(((TOTAL_TASKS + ARRAY_SIZE - 1) / ARRAY_SIZE))
START=$((ARRAY_INDEX * CHUNK))
if [ "$START" -ge "$TOTAL_TASKS" ]; then
  echo "Array task $ARRAY_INDEX has no tasks to generate (array size $ARRAY_SIZE > $TOTAL_TASKS tasks); nothing to do."
  exit 0
fi
END=$((START + CHUNK - 1))
if [ "$END" -ge "$TOTAL_TASKS" ]; then
  END=$((TOTAL_TASKS - 1))
fi
TASK_INDICES="$START-$END"
echo "Array task $ARRAY_INDEX/$ARRAY_SIZE generating task-file positions $TASK_INDICES (of $TOTAL_TASKS) for both models."

export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
# Must match whatever install_env.sh actually populated. Defaults the same way that
# script does, so this works even if HF_HOME was never exported/passed through --export.
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
# Enforce offline: run scripts/slurm/install_env.sh (or warm_model_cache.sh) first. With
# this set, a cold/incomplete cache fails immediately with a clear error instead of each
# of the (potentially 20+) concurrent array tasks silently racing to hit the Hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Assumes the active python environment (see install_env.sh / CONDA_ENV above) already
# has every dependency this repo needs -- no environment install/sync happens here.
python -m dev.schema_llm.explore.tsp_generate \
  --root "$ROOT" \
  --tasks "$TASKS" \
  --out "$OUT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED" \
  --device cuda:0 \
  --task-indices "$TASK_INDICES"
