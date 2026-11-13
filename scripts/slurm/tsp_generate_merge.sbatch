#!/usr/bin/env bash
# Final merge step: submit with --dependency=afterok:<array job id> (done automatically by
# submit_tsp_generation.sh). Runs the generator with NO sharding args, so it iterates the
# full task list; every shard was already written by tsp_generate_array.sbatch, so this
# only validates/loads cached rows (no model weights loaded) and builds OUT/pack/.
#SBATCH --job-name=tsp-gen-merge
#SBATCH --output=logs/tsp-gen-merge_%j.out
#SBATCH --error=logs/tsp-gen-merge_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail

ROOT="${TSP_ROOT:?set TSP_ROOT to the repo checkout path}"
OUT="${TSP_OUT:?set TSP_OUT to the shared output directory}"
TASKS="${TSP_TASKS:?set TSP_TASKS to the tasks JSONL path}"
MAX_NEW_TOKENS="${TSP_MAX_NEW_TOKENS:-32768}"
SEED="${TSP_SEED:-0}"

cd "$ROOT"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
# Must match whatever install_env.sh actually populated. Defaults the same way that
# script does, so this works even if HF_HOME was never exported/passed through --export.
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
# Enforce offline (see tsp_generate_array.sbatch for why): the merge job typically only
# reads already-cached shards and never even touches a model, but if it does need to
# regenerate a missing one, it must not silently hit the network either.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run --no-sync python -m dev.schema_llm.explore.tsp_generate \
  --root "$ROOT" \
  --tasks "$TASKS" \
  --out "$OUT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED" \
  --device cuda:0

echo "Pack ready: $OUT/pack"
