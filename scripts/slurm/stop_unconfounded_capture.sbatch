#!/usr/bin/env bash
# Stage 0 (single GPU job, no array): forward-passes the 89 reaches_solved unconfounded-corpus
# traces once each (teacher-forced, no generation) to capture real layer-16 residual states at
# the same S/B run-end positions the supplied STOP-approach direction was built from, then
# builds the shuffled-label null from them (no shuffled vector was supplied -- see
# unconfounded_capture.py's module docstring for why this needs a capture pass at all).
#
# Run this ONCE before stop_unconfounded_generate_array.sbatch if you want the "shuffled" arm;
# it writes data/stop_solved_unconfounded/derived/shuffled_direction_phi4mr_unconfounded.npz,
# which that array job reads. submit_stop_unconfounded.sh submits this first automatically
# (--dependency=afterok chains the array job to it).
#SBATCH --job-name=stop-unconf-capture
#SBATCH --output=logs/stop-unconf-capture_%j.out
#SBATCH --error=logs/stop-unconf-capture_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail

ROOT="${STOP_ROOT:?set STOP_ROOT to the repo checkout path}"

cd "$ROOT"

# Optional safety net: see tsp_generate_array.sbatch for why this may be needed.
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

python -m dev.schema_llm.explore.stop_steering.unconfounded_capture \
  --root "$ROOT" \
  --device cuda:0
