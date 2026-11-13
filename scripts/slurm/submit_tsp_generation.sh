#!/usr/bin/env bash
# Submits a TSP generation array (each array task gets a contiguous block of tasks, e.g.
# 100 tasks / 20 array tasks = 5 tasks x 2 models = 10 responses per job), then a merge
# job that depends on it (--dependency=afterok) and assembles the final pack once every
# array task is done.
#
# Prerequisites: run scripts/slurm/install_env.sh once first, FROM YOUR ACTIVATED PYTHON
# ENVIRONMENT (e.g. `conda activate <env>`) -- it verifies that environment and caches
# both pinned checkpoints under HF_HOME. Run THIS script from that same activated shell
# too: --export=ALL below already carries its PATH/CONDA_* into every Slurm job.
#
# Nothing is strictly required -- every setting below has a default. Edit/export before
# running only to override them:
#   HF_HOME                          (default: $ROOT/.cache/huggingface, matching
#                                      install_env.sh's default -- override to point at
#                                      shared/project storage instead of the repo clone)
#   SLURM_ACCOUNT, SLURM_PARTITION   (sbatch --account/--partition; omitted entirely if
#                                      unset, so your cluster's defaults apply)
#   CONDA_ENV                        (only needed if your cluster's job startup does NOT
#                                      inherit PATH/CONDA_* from --export=ALL -- set to
#                                      the env name and both .sbatch files will
#                                      explicitly `conda activate` it)
#   ARRAY_SIZE (default 20), TSP_OUT, TSP_TASKS, TSP_MAX_NEW_TOKENS, TSP_SEED
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SLURM_DIR="$ROOT/scripts/slurm"

# Must match whatever install_env.sh actually populated -- same default as that script.
HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

# Only pass --account/--partition through to sbatch if you actually set them.
SBATCH_ACCOUNT_ARGS=()
[ -n "${SLURM_ACCOUNT:-}" ] && SBATCH_ACCOUNT_ARGS+=(--account="$SLURM_ACCOUNT")
SBATCH_PARTITION_ARGS=()
[ -n "${SLURM_PARTITION:-}" ] && SBATCH_PARTITION_ARGS+=(--partition="$SLURM_PARTITION")

ARRAY_SIZE="${ARRAY_SIZE:-20}"
TSP_OUT="${TSP_OUT:-$ROOT/results/tsp-generation}"
TSP_TASKS="${TSP_TASKS:-$ROOT/data/tsp_generation/tasks_100.jsonl}"
TSP_MAX_NEW_TOKENS="${TSP_MAX_NEW_TOKENS:-32768}"
TSP_SEED="${TSP_SEED:-0}"

mkdir -p "$TSP_OUT" "$SLURM_DIR/logs"

export_vars="ALL,TSP_ROOT=$ROOT,TSP_OUT=$TSP_OUT,TSP_TASKS=$TSP_TASKS"
export_vars+=",TSP_MAX_NEW_TOKENS=$TSP_MAX_NEW_TOKENS,TSP_SEED=$TSP_SEED,HF_HOME=$HF_HOME"

ARRAY_JOB=$(sbatch --parsable \
  "${SBATCH_ACCOUNT_ARGS[@]}" "${SBATCH_PARTITION_ARGS[@]}" \
  --array="0-$((ARRAY_SIZE - 1))" \
  --chdir="$SLURM_DIR" \
  --export="${export_vars}" \
  "$SLURM_DIR/tsp_generate_array.sbatch")
echo "submitted array job $ARRAY_JOB (0-$((ARRAY_SIZE - 1)), $ARRAY_SIZE array tasks)"

MERGE_JOB=$(sbatch --parsable \
  "${SBATCH_ACCOUNT_ARGS[@]}" "${SBATCH_PARTITION_ARGS[@]}" \
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
