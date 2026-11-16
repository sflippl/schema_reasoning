#!/usr/bin/env bash
# Submits the 3-stage unconfounded STOP-injection experiment:
#   1. stop_unconfounded_capture.sbatch   -- builds the shuffled-label null (skip with
#                                             SKIP_CAPTURE=1 if you already have it, or if
#                                             STOP_ARMS excludes "shuffled")
#   2. stop_unconfounded_generate_array.sbatch -- injects into the 84-item population,
#                                             sharded across ARRAY_SIZE array tasks
#                                             (--dependency=afterok:<capture job>)
#   3. stop_unconfounded_generate_merge.sbatch -- assembles the final pack
#                                             (--dependency=afterok:<array job>)
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
#   ARRAY_SIZE (default 20), STOP_OUT, STOP_ARMS (default "stop shuffled"),
#   STOP_DOSES (default "0.25 0.5 1.0"), STOP_BUDGET (default 2048), SKIP_CAPTURE
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SLURM_DIR="$ROOT/scripts/slurm"

HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

SBATCH_ACCOUNT_ARGS=()
[ -n "${SLURM_ACCOUNT:-}" ] && SBATCH_ACCOUNT_ARGS=(--account="$SLURM_ACCOUNT")
SBATCH_PARTITION_ARGS=()
[ -n "${SLURM_PARTITION:-}" ] && SBATCH_PARTITION_ARGS=(--partition="$SLURM_PARTITION")

ARRAY_SIZE="${ARRAY_SIZE:-20}"
STOP_OUT="${STOP_OUT:-$ROOT/results/stop-unconfounded-injection}"
STOP_ARMS="${STOP_ARMS:-stop shuffled}"
STOP_DOSES="${STOP_DOSES:-0.25 0.5 1.0}"
STOP_BUDGET="${STOP_BUDGET:-2048}"

mkdir -p "$STOP_OUT" "$SLURM_DIR/logs"

export_vars="ALL,STOP_ROOT=$ROOT,STOP_OUT=$STOP_OUT,STOP_ARMS=$STOP_ARMS"
export_vars+=",STOP_DOSES=$STOP_DOSES,STOP_BUDGET=$STOP_BUDGET,HF_HOME=$HF_HOME"

ARRAY_DEPENDENCY=()
if [ "${SKIP_CAPTURE:-0}" != "1" ]; then
  CAPTURE_JOB=$(sbatch --parsable \
    "${SBATCH_ACCOUNT_ARGS[@]+"${SBATCH_ACCOUNT_ARGS[@]}"}" \
    "${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"}" \
    --chdir="$SLURM_DIR" \
    --export="ALL,STOP_ROOT=$ROOT,HF_HOME=$HF_HOME" \
    "$SLURM_DIR/stop_unconfounded_capture.sbatch")
  echo "submitted capture job $CAPTURE_JOB (builds the shuffled null)"
  ARRAY_DEPENDENCY=(--dependency="afterok:$CAPTURE_JOB")
else
  echo "SKIP_CAPTURE=1: not submitting the capture job (shuffled null must already exist if STOP_ARMS includes 'shuffled')"
fi

ARRAY_JOB=$(sbatch --parsable \
  "${SBATCH_ACCOUNT_ARGS[@]+"${SBATCH_ACCOUNT_ARGS[@]}"}" \
  "${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"}" \
  "${ARRAY_DEPENDENCY[@]+"${ARRAY_DEPENDENCY[@]}"}" \
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

Final pack (84 items x (1 control + arms x doses)) lands at: $STOP_OUT/pack
EOF
