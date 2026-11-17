"""Shared loaders for the "unconfounded" STOP-direction replication corpus.

This is a SEPARATE corpus from the one ``runframe.py``/``pilot.py``/``inject.py`` were written
against (``results/schema_llm/export/20260824-rlreasoning_v1``, bound via ``configure()``).
The new corpus is D1 of ``20260915-binding_dyn_d1_v3w2`` (300 phi4mr items; see
``data/stop_solved_unconfounded/README.md``), already present in this repo as
``data/vstop/core/D1-tokens.parquet`` and (for gold answers) ``data/primitives/tasks.parquet``
(``corpus == "d1"``). Its STOP-approach direction/metadata were computed externally and shipped
as derived artifacts under ``data/stop_solved_unconfounded/``.

Despite the different corpus, the run/position bookkeeping is IDENTICAL to the original
pipeline's (same ``metacluster_L16``-run definition, same trailing-marker trim), so this module
reuses ``runframe.Trace``/``runs_of``/``clean``/``MARK`` directly rather than reimplementing
them. Verified byte-exact against the supplied metadata: reconstructing item_0005's run boundaries
from ``D1-tokens.parquet`` alone reproduces ``solved_position_abs`` (1192) and ``end_position_abs``
(1590) from ``t_solved_per_trace_phi4mr_unconfounded.csv`` exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import runframe

MODEL = "phi4mr"
RUN_OFFSET = 30  # inject at the run end this many runs after T_solved (protocol section 4)
CORPUS_DIR = Path("data/stop_solved_unconfounded")
VECTORS_PATH = Path("data/stop_vectors/phi4mr/stop_vector_phi4mr.npz")
POPULATION_CSV = CORPUS_DIR / "t_solved_per_trace_phi4mr_unconfounded.csv"
TOKENS_PATH = Path("data/vstop/core/D1-tokens.parquet")
GOLD_TASKS_PATH = Path("data/primitives/tasks.parquet")
GOLD_CORPUS = "d1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_vectors(root: Path) -> dict:
    """The supplied direction vectors/dose units, as plain Python/NumPy values."""
    with np.load(root / VECTORS_PATH) as data:
        return {k: data[k] for k in data.files}


def load_population(root: Path, min_runs_after_solved: int = RUN_OFFSET) -> pd.DataFrame:
    """Judge-all-right, solved traces with at least ``min_runs_after_solved`` runs to spare.

    Pass the default (``RUN_OFFSET`` = 30) for the INJECTION population: traces with
    ``runs_after_solved <= 30`` cannot be injected at all under this protocol, since the
    model's own reasoning already ended before the prescribed injection point would occur (84
    of 89 ``reaches_solved`` items clear that bar). Pass ``0`` for the CAPTURE population used to
    build the shuffled-label null -- that only needs ``reaches_solved`` (89 items), not injection
    feasibility.
    """
    table = pd.read_csv(root / POPULATION_CSV)
    require((table["model"] == MODEL).all(), "unexpected model in population csv")
    table = table[table["reaches_solved"] & (table["runs_after_solved"] > min_runs_after_solved)]
    require(len(table), "empty population")
    return table.sort_values("item_id").reset_index(drop=True)


def load_tokens(root: Path) -> pd.DataFrame:
    """D1's phi4mr tokens, with the columns ``inject.py``'s resume/prompt-length helpers need."""
    columns = ["item_id", "task_family", "model", "position", "token_id", "token", "segment",
               "metacluster_L16"]
    tokens = pd.read_parquet(root / TOKENS_PATH, columns=columns)
    tokens = tokens[(tokens.model == MODEL) & (tokens.task_family == runframe.TASK)]
    require(len(tokens), "empty D1 phi4mr token slice")
    return tokens


def build_trace(tokens: pd.DataFrame, item: str) -> runframe.Trace:
    """Port of ``runframe.build_traces``'s per-item logic for ONE already-terminated item.

    Every item this module is used for comes from ``load_population`` (``reaches_solved``, hence
    ``terminated`` per the t_solved csv), so this always uses the think-only segment path -- the
    ``["think", "answer"]`` fallback ``build_traces`` uses for non-terminating items does not
    apply here and is deliberately not ported.
    """
    d = tokens[(tokens.item_id == item) & (tokens.segment == "think")].sort_values("position")
    ok = ~pd.isna(d["metacluster_L16"].values)
    z = d["metacluster_L16"].values[ok].astype(int)
    pos = d["position"].values[ok].astype(int)
    raw = [str(x) for x in d["token"].values[ok]]
    text = [runframe.clean(x) for x in raw]
    n = len(text)
    while n > 0 and text[n - 1] in runframe.MARK:
        n -= 1
    require(n >= runframe.MIN_RUNS, f"{item}: too few think tokens after trim")
    return runframe.Trace(model=MODEL, item=item, z=z[:n], pos=pos[:n], tokens=text[:n],
                          raw=raw[:n], stripped=len(text) - n, terminating=True)


def injection_prefix_len(trace: runframe.Trace, solved_run_index: int,
                         run_offset: int = RUN_OFFSET) -> int:
    """Absolute token position (``resume_generate``'s ``prefix_len``) to inject at.

    "The run end ``run_offset`` runs after ``solved_run_index``" (protocol section 4), computed
    the same way ``pilot.injection_point`` computes the original corpus's T_solved offset.
    """
    _, starts, durations = runframe.runs_of(trace.z)
    ends = starts + durations - 1
    target = solved_run_index + run_offset
    require(0 <= target < len(ends), "run_offset outside the run list")
    return int(trace.pos[ends[target]]) + 1


def load_task_records(root: Path) -> dict[str, dict]:
    """Full D1 task records keyed by the bare ``item_NNNN`` id.

    ``data/primitives/tasks.parquet`` namespaces D1 items as ``d1__item_NNNN``; this strips that
    prefix so callers can key by the same bare id used in ``D1-tokens.parquet``/the population csv.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(root / GOLD_TASKS_PATH)
    out = {}
    for row in table.to_pylist():
        if row.get("corpus") != GOLD_CORPUS:
            continue
        item_id = row["item_id"]
        bare = item_id.removeprefix(f"{GOLD_CORPUS}__")
        out[bare] = row
    require(out, "empty gold lookup")
    return out


def load_gold(root: Path) -> dict[str, list]:
    """D1's person-level gold role records, keyed by bare item id."""
    return {
        item: json.loads(task["gold"])["roles"]
        for item, task in load_task_records(root).items()
    }
