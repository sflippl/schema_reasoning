#!/usr/bin/env bash
# Submit the prompt-only alternative to the unconfounded STOP-vector experiment:
#   1. array job: replay each item's prefix through T_solved and +30 runs, append the literal
#      text condition, then generate a paired greedy continuation up to the remaining 8192-token
#      completion cap (override with STOP_PROMPT_BUDGET>0 for a fixed generated-token limit);
#   2. merge job: assemble all item shards into results.parquet and a manifest.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SLURM_DIR="$ROOT/scripts/slurm"

HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
ARRAY_SIZE="${STOP_PROMPT_ARRAY_SIZE:-20}"
STOP_PROMPT_OUT="${STOP_PROMPT_OUT:-$ROOT/results/stop-unconfounded-prompt-injection-v2}"
STOP_PROMPT_BUDGET="${STOP_PROMPT_BUDGET:--1}"
STOP_PROMPT_RUN_OFFSETS="${STOP_PROMPT_RUN_OFFSETS:-0 30}"
[[ "$ARRAY_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "STOP_PROMPT_ARRAY_SIZE must be a positive integer." >&2
  exit 2
}

SBATCH_ACCOUNT_ARGS=()
[ -n "${SLURM_ACCOUNT:-}" ] && SBATCH_ACCOUNT_ARGS=(--account="$SLURM_ACCOUNT")
SBATCH_PARTITION_ARGS=()
[ -n "${SLURM_PARTITION:-}" ] && SBATCH_PARTITION_ARGS=(--partition="$SLURM_PARTITION")

mkdir -p "$STOP_PROMPT_OUT" "$SLURM_DIR/logs"
export_vars="ALL,STOP_ROOT=$ROOT,STOP_PROMPT_OUT=$STOP_PROMPT_OUT"
export_vars+=",STOP_PROMPT_BUDGET=$STOP_PROMPT_BUDGET"
export_vars+=",STOP_PROMPT_RUN_OFFSETS=$STOP_PROMPT_RUN_OFFSETS"
export_vars+=",STOP_PROMPT_ARRAY_SIZE=$ARRAY_SIZE,HF_HOME=$HF_HOME"

ARRAY_JOB=$(sbatch --parsable \
  "${SBATCH_ACCOUNT_ARGS[@]+"${SBATCH_ACCOUNT_ARGS[@]}"}" \
  "${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"}" \
  --array="0-$((ARRAY_SIZE - 1))" \
  --chdir="$SLURM_DIR" \
  --export="$export_vars" \
  "$SLURM_DIR/stop_unconfounded_prompt_array.sbatch")
echo "submitted prompt-injection array $ARRAY_JOB (0-$((ARRAY_SIZE - 1)), $ARRAY_SIZE tasks)"

MERGE_JOB=$(sbatch --parsable \
  "${SBATCH_ACCOUNT_ARGS[@]+"${SBATCH_ACCOUNT_ARGS[@]}"}" \
  "${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"}" \
  --dependency="afterok:$ARRAY_JOB" \
  --chdir="$SLURM_DIR" \
  --export="$export_vars" \
  "$SLURM_DIR/stop_unconfounded_prompt_merge.sbatch")
echo "submitted merge job $MERGE_JOB (runs after $ARRAY_JOB succeeds)"

cat <<EOF

Track progress with:
  squeue -u \$USER
  tail -f $SLURM_DIR/logs/stop-prompt_${ARRAY_JOB}_*.out
  tail -f $SLURM_DIR/logs/stop-prompt-merge_${MERGE_JOB}.out

Generated responses and task records: $STOP_PROMPT_OUT/pack
EOF
