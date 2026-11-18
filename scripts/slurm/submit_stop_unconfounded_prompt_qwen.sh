#!/usr/bin/env bash
# Submit the Qwen prompt-insertion array and dependent merge using the shared job scripts.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export SCHEMA_HEAD_FV_MODEL=qwen34b
export STOP_PROMPT_OUT="${STOP_PROMPT_OUT:-$ROOT/results/stop-unconfounded-prompt-injection-qwen34b-v2}"

exec bash "$ROOT/scripts/slurm/submit_stop_unconfounded_prompt.sh"
