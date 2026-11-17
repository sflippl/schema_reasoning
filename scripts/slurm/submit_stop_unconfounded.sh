#!/usr/bin/env bash
# Submit the core, primitive-centered STOP-vector experiment:
#   1. stop_unconfounded_generate_array.sbatch -- paired control and vector doses at offsets 0/30
#   2. stop_unconfounded_generate_merge.sbatch -- assembles the final pack
#
# Prerequisites: run from your already-activated Python environment (e.g. `conda activate
# <env>`) that already has this repo's deps -- same expectation as
# scripts/slurm/install_env.sh/submit_tsp_generation.sh. Submit from that same activated
# shell: --export=ALL below carries its PATH/CONDA_* into every Slurm job.
#
# Nothing below is required -- every setting has a default. Edit/export before running only
# to override them:
#   HF_HOME                          (default: $ROOT/.cache/huggingface, matching
#                                      install_env.sh's default)
#   SLURM_ACCOUNT, SLURM_PARTITION   (sbatch --account/--partition; omitted entirely if
#                                      unset, so your cluster's defaults apply)
#   CONDA_ENV                        (only needed if your cluster's job startup does NOT
#                                      inherit PATH/CONDA_* from --export=ALL)
#   ARRAY_SIZE (default 20), STOP_OUT, STOP_ARMS (default "stop"),
#   STOP_DOSES (default "0.25 0.5"), STOP_BUDGET (default -1 = remaining capacity to 8192),
#   STOP_RUN_OFFSETS (default "0 30" -- runs after T_solved to inject at; every item is
#     injected at EVERY requested offset, so conditions are paired per item)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SLURM_DIR="$ROOT/scripts/slurm"

HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

SBATCH_ACCOUNT_ARGS=()
[ -n "${SLURM_ACCOUNT:-}" ] && SBATCH_ACCOUNT_ARGS=(--account="$SLURM_ACCOUNT")
SBATCH_PARTITION_ARGS=()
[ -n "${SLURM_PARTITION:-}" ] && SBATCH_PARTITION_ARGS=(--partition="$SLURM_PARTITION")

ARRAY_SIZE="${ARRAY_SIZE:-20}"
STOP_OUT="${STOP_OUT:-$ROOT/results/stop-unconfounded-injection-v2}"
STOP_ARMS="${STOP_ARMS:-stop}"
STOP_DOSES="${STOP_DOSES:-0.25 0.5}"
STOP_BUDGET="${STOP_BUDGET:--1}"
STOP_RUN_OFFSETS="${STOP_RUN_OFFSETS:-0 30}"

mkdir -p "$STOP_OUT" "$SLURM_DIR/logs"

export_vars="ALL,STOP_ROOT=$ROOT,STOP_OUT=$STOP_OUT,STOP_ARMS=$STOP_ARMS"
export_vars+=",STOP_DOSES=$STOP_DOSES,STOP_BUDGET=$STOP_BUDGET"
export_vars+=",STOP_RUN_OFFSETS=$STOP_RUN_OFFSETS,HF_HOME=$HF_HOME"

ARRAY_JOB=$(sbatch --parsable \
  "${SBATCH_ACCOUNT_ARGS[@]+"${SBATCH_ACCOUNT_ARGS[@]}"}" \
  "${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"}" \
  --array="0-$((ARRAY_SIZE - 1))" \
  --chdir="$SLURM_DIR" \
  --export="${export_vars}" \
  "$SLURM_DIR/stop_unconfounded_generate_array.sbatch")
echo "submitted array job $ARRAY_JOB (0-$((ARRAY_SIZE - 1)), $ARRAY_SIZE array tasks)"

MERGE_JOB=$(sbatch --parsable \
  "${SBATCH_ACCOUNT_ARGS[@]+"${SBATCH_ACCOUNT_ARGS[@]}"}" \
  "${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"}" \
  --dependency="afterok:$ARRAY_JOB" \
  --chdir="$SLURM_DIR" \
  --export="$export_vars" \
  "$SLURM_DIR/stop_unconfounded_generate_merge.sbatch")
echo "submitted merge job $MERGE_JOB (runs after $ARRAY_JOB succeeds)"

cat <<EOF

Track progress with:
  squeue -u \$USER
  tail -f $SLURM_DIR/logs/stop-unconf-gen_${ARRAY_JOB}_*.out
  tail -f $SLURM_DIR/logs/stop-unconf-merge_${MERGE_JOB}.out

Final pack (84 items x run_offsets x (1 control + arms x doses)) lands at: $STOP_OUT/pack
EOF
