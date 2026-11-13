#!/usr/bin/env bash
# Submits a TSP generation array (each array task gets a contiguous block of tasks, e.g.
# 100 tasks / 20 array tasks = 5 tasks x 2 models = 10 responses per job), then a merge
# job that depends on it (--dependency=afterok) and assembles the final pack once every
# array task is done.
#
# Prerequisites: run scripts/slurm/install_env.sh once first (builds .venv, caches both
# pinned checkpoints under HF_HOME).
#
# Required cluster settings -- edit these or export them before running:
#   SLURM_ACCOUNT, SLURM_PARTITION   (sbatch --account/--partition for your cluster)
#   HF_HOME                          (shared cache populated by install_env.sh)
#
# Optional overrides: ARRAY_SIZE (default 20), TSP_OUT, TSP_TASKS, TSP_MAX_NEW_TOKENS,
# TSP_SEED.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SLURM_DIR="$ROOT/scripts/slurm"

: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to your Slurm billing account}"
: "${SLURM_PARTITION:?set SLURM_PARTITION to your GPU partition}"
: "${HF_HOME:?set HF_HOME to the shared cache created by install_env.sh}"

ARRAY_SIZE="${ARRAY_SIZE:-20}"
TSP_OUT="${TSP_OUT:-$ROOT/results/tsp-generation}"
TSP_TASKS="${TSP_TASKS:-$ROOT/data/tsp_generation/tasks_100.jsonl}"
TSP_MAX_NEW_TOKENS="${TSP_MAX_NEW_TOKENS:-32768}"
TSP_SEED="${TSP_SEED:-0}"

mkdir -p "$TSP_OUT" "$SLURM_DIR/logs"

export_vars="ALL,TSP_ROOT=$ROOT,TSP_OUT=$TSP_OUT,TSP_TASKS=$TSP_TASKS"
export_vars+=",TSP_MAX_NEW_TOKENS=$TSP_MAX_NEW_TOKENS,TSP_SEED=$TSP_SEED,HF_HOME=$HF_HOME"

ARRAY_JOB=$(sbatch --parsable \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
  --array="0-$((ARRAY_SIZE - 1))" \
  --chdir="$SLURM_DIR" \
  --export="${export_vars}" \
  "$SLURM_DIR/tsp_generate_array.sbatch")
echo "submitted array job $ARRAY_JOB (0-$((ARRAY_SIZE - 1)), $ARRAY_SIZE array tasks)"

MERGE_JOB=$(sbatch --parsable \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
  --dependency="afterok:$ARRAY_JOB" \
  --chdir="$SLURM_DIR" \
  --export="$export_vars" \
  "$SLURM_DIR/tsp_generate_merge.sbatch")
echo "submitted merge job $MERGE_JOB (runs after $ARRAY_JOB succeeds)"

cat <<EOF

Track progress with:
  squeue -u \$USER
  tail -f $SLURM_DIR/logs/tsp-gen_${ARRAY_JOB}_*.out
  tail -f $SLURM_DIR/logs/tsp-gen-merge_${MERGE_JOB}.out

Final pack (200 responses: 100 phi4mr + 100 qwen34b) lands at: $TSP_OUT/pack
EOF
