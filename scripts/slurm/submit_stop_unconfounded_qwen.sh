#!/usr/bin/env bash
# Submit the Qwen STOP-vector array and dependent merge using the shared Phi job scripts.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export SCHEMA_HEAD_FV_MODEL=qwen34b
export STOP_OUT="${STOP_OUT:-$ROOT/results/stop-unconfounded-injection-qwen34b-v2}"

exec bash "$ROOT/scripts/slurm/submit_stop_unconfounded.sh"
