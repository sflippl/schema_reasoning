"""Shared loaders for the "unconfounded" STOP-direction replication corpus.

This is a SEPARATE corpus from the one ``runframe.py``/``pilot.py``/``inject.py`` were written
against (``results/schema_llm/export/20260824-rlreasoning_v1``, bound via ``configure()``).
The new corpus is D1 of ``20260915-binding_dyn_d1_v3w2`` (300 items per model; see
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

from ..head_fv_models import active as active_model
from . import runframe

RUN_OFFSET = 30  # inject at the run end this many runs after T_solved (protocol section 4)
CORPUS_DIR = Path("data/stop_solved_unconfounded")
TOKENS_PATH = Path("data/vstop/core/D1-tokens.parquet")
GOLD_TASKS_PATH = Path("data/primitives/tasks.parquet")
GOLD_CORPUS = "d1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def population_path() -> Path:
    return CORPUS_DIR / f"t_solved_per_trace_{active_model().key}_unconfounded.csv"


def vectors_path() -> Path:
    model = active_model().key
    return Path(f"data/stop_vectors/{model}/stop_vector_{model}.npz")


def load_vectors(root: Path) -> dict:
    """The supplied direction vectors/dose units, as plain Python/NumPy values."""
    with np.load(root / vectors_path()) as data:
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
    model = active_model().key
    table = pd.read_csv(root / population_path())
    required = {"model", "item_id", "reaches_solved", "solved_token_index",
                "solved_run_index", "solved_position_abs", "end_token_index",
                "end_run_index", "end_position_abs", "runs_after_solved",
                "think_tokens", "think_runs", "terminated"}
    require(required <= set(table.columns), f"population csv missing columns: {sorted(required - set(table.columns))}")
    require(not table.empty and (table["model"] == model).all(),
            f"unexpected model or empty population csv for {model}")
    require(table["item_id"].notna().all() and table["item_id"].is_unique,
            "population csv has missing or duplicate item IDs")
    require(table["reaches_solved"].isin([True, False]).all()
            and table["terminated"].isin([True, False]).all(),
            "population csv contains invalid boolean flags")
    table = table[table["reaches_solved"] & (table["runs_after_solved"] > min_runs_after_solved)]
    require(len(table) and table["terminated"].all(), "empty or non-terminating population")
    require(table[list(required - {"model", "item_id"})].notna().all().all(),
            "eligible population contains missing values")
    return table.sort_values("item_id").reset_index(drop=True)


def load_tokens(root: Path) -> pd.DataFrame:
    """D1 tokens for the selected model, with the columns needed by the resume harness."""
    columns = ["item_id", "task_family", "model", "position", "token_id", "token", "segment",
               "metacluster_L16"]
    tokens = pd.read_parquet(root / TOKENS_PATH, columns=columns)
    model = active_model().key
    tokens = tokens[(tokens.model == model) & (tokens.task_family == runframe.TASK)]
    require(len(tokens), f"empty D1 {model} token slice")
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
    return runframe.Trace(model=active_model().key, item=item, z=z[:n], pos=pos[:n], tokens=text[:n],
                          raw=raw[:n], stripped=len(text) - n, terminating=True)


def validate_trace(trace: runframe.Trace, target) -> None:
    """Check that the supplied solved coordinates describe this model's actual trace."""
    _, starts, durations = runframe.runs_of(trace.z)
    ends = starts + durations - 1
    token = int(target["solved_token_index"])
    run = int(target["solved_run_index"])
    item = trace.item
    require(len(trace.pos) == int(target["think_tokens"]) and len(starts) == int(target["think_runs"]),
            f"{item}: trace dimensions disagree with population csv")
    require(0 <= token < len(trace.pos) and 0 <= run < len(starts),
            f"{item}: solved coordinates are out of range")
    require(starts[run] <= token <= ends[run]
            and trace.pos[token] == int(target["solved_position_abs"]),
            f"{item}: solved coordinates disagree with token data")
    require(trace.pos[-1] == int(target["end_position_abs"])
            and int(target["end_token_index"]) == len(trace.pos) - 1
            and int(target["end_run_index"]) == len(starts) - 1
            and int(target["runs_after_solved"]) == len(starts) - 1 - run,
            f"{item}: end coordinates disagree with token data")


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
