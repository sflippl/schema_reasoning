#!/usr/bin/env bash
# Run ONCE, from your already-activated Python environment (install_env.sh calls this
# automatically; rerun it standalone any time you want to re-verify the cache, e.g. after
# clearing HF_HOME or moving to a new shared filesystem).
#
# Downloads AND fully exercises both pinned checkpoints through the SAME load path
# tsp_generate.py uses (dev.schema_llm.core.model.load_model: tokenizer, chat template,
# config, weights), on CPU -- no GPU reservation needed just to populate a cache. This is
# stronger than a raw file-download: it also catches cache/revision/tokenizer mismatches
# before committing to a 20-way array.
#
# The point: once this has completed, tsp_generate_array.sbatch and
# tsp_generate_merge.sbatch run with HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1, so NONE of
# the (potentially 20+) concurrent generation jobs ever hits the network -- there's no
# window where multiple jobs could race to download the same multi-GB checkpoint, and a
# genuinely incomplete cache fails immediately and loudly instead of hanging/retrying.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

echo "Warming HF_HOME=$HF_HOME by loading both pinned checkpoints on CPU ..."
python - <<'PY'
from dev.schema_llm.core.model import load_model
from dev.schema_llm.explore.combined_primitives import MODELS

for key, (checkpoint, revision) in MODELS.items():
    print(f"[{key}] loading {checkpoint}@{revision} (device_map=cpu) ...", flush=True)
    lm = load_model(checkpoint, revision=revision, device_map="cpu")
    del lm
    print(f"[{key}] OK -- cached and loadable.", flush=True)
print("Both pinned checkpoints are fully downloaded and verified loadable from HF_HOME.")
PY

echo "Model cache warmed and verified at HF_HOME=$HF_HOME"
