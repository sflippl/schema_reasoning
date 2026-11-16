"""Inject the raw (uncentred) unconfounded STOP-approach direction, plus its shuffled null.

Parallel to ``tsp_generate.py`` in shape (resumable ``--task-indices`` shards over an item
population, a separate merge step building the final pack), but injecting into the unconfounded
corpus's own 84-item population (``unconfounded_data.load_population``) instead of generating new
TSP responses. Reuses the existing resume/injection harness (``inject.resume_generate``) and
scoring (``pilot.tail_stats``) unchanged -- only the vector, dose unit, population and injection
position are specific to this corpus.

Vector: ``u_raw_residual`` (the uncentred variant -- no within-primitive mean subtraction), dose
``alpha * sd_raw_along_u``, matching the supplied metadata's own convention for that variant.
The ``shuffled`` arm uses ``unconfounded_capture.build``'s reconstructed null (run that first;
see its module docstring for why a real shuffled-label null needs a capture pass at all). An
optional ``solved`` specificity-control arm is also available via ``--arms`` but is NOT run by
default, since its dose unit (no ``sd`` was supplied for it) is a documented assumption rather
than something the data specifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import combined_primitives as cp
from ..tsp_generate import parse_task_indices
from . import inject, pilot
from . import unconfounded_data as ud

ARMS = ("stop", "shuffled")  # "solved" is available but not run by default; see module docstring
DOSES = (0.25, 0.5, 1.0)
LAYER = 16
SCHEDULE = "sustained"
BUDGET = 2048
SHUFFLED_NPZ = Path("data/stop_solved_unconfounded/derived/shuffled_direction_phi4mr_unconfounded.npz")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shard_path(out: Path, item: str) -> Path:
    name = hashlib.sha256(item.encode()).hexdigest()
    return out / "shards" / f"{name}.json"


def arm_vector_and_dose_unit(root: Path, arm: str, supplied: dict) -> tuple[np.ndarray, float]:
    """``(unit vector, dose unit)`` for one arm -- everything downstream multiplies by dose."""
    if arm == "stop":
        return supplied["u_raw_residual"], float(supplied["sd_raw_along_u"])
    if arm == "shuffled":
        path = root / SHUFFLED_NPZ
        require(path.exists(),
                f"missing {path}; run unconfounded_capture.py first to build the shuffled null")
        with np.load(path) as data:
            return data["u_shuffled_residual"], float(data["sd_shuffled_along_u"])
    if arm == "solved":
        # No sd was supplied for u_solved_residual (it isn't a within-primitive/raw pair like
        # the stop direction is). Reusing sd_raw_along_u puts it on the same dose SCALE as the
        # stop arm it's meant to contrast with, but this is a judgement call, not a supplied
        # value -- flagged here and in the module docstring, not snuck in silently.
        return supplied["u_solved_residual"], float(supplied["sd_raw_along_u"])
    raise ValueError(f"unknown arm: {arm}")


def run(
    root: Path,
    out: Path,
    device: str,
    arms: tuple[str, ...] = ARMS,
    doses: tuple[float, ...] = DOSES,
    budget: int = BUDGET,
    layer: int = LAYER,
    schedule: str = SCHEDULE,
    smoke: bool = False,
    task_indices: list[int] | None = None,
) -> Path | None:
    """Generate this job's explicit subset of items, then build the pack iff it's everything.

    Mirrors ``tsp_generate.run``: ``task_indices`` selects which population rows THIS process
    injects into; it is not part of ``config``, so partial runs over different subsets of the
    SAME population/settings keep validating against each other. The pack is only built when
    ``task_indices`` is ``None`` (the default, and the final merge invocation).
    """
    from ...core.model import load_model
    from ..head_fv_models import active as active_model
    from ..reproduction import _runtime

    population = ud.load_population(root)
    if smoke:
        population = population.iloc[:2]
    if task_indices is None:
        work = population
    else:
        bad = [i for i in task_indices if not 0 <= i < len(population)]
        require(not bad, f"item index out of range [0, {len(population)}): {bad}")
        work = population.iloc[task_indices]

    supplied = ud.load_vectors(root)
    arm_specs = {arm: arm_vector_and_dose_unit(root, arm, supplied) for arm in arms}
    plan = [("control", 0.0)] + [(arm, dose) for arm in arms for dose in doses]

    config = {
        "corpus": "unconfounded-d1",
        "release": cp._sha(root / "MANIFEST.json"),
        "population_sha256": cp._sha(root / ud.POPULATION_CSV),
        "vectors_sha256": cp._sha(root / ud.VECTORS_PATH),
        "shuffled_sha256": cp._sha(root / SHUFFLED_NPZ) if "shuffled" in arms else None,
        "runtime": _runtime(),
        "device": device,
        "dtype": "bfloat16",
        "layer": layer,
        "schedule": schedule,
        "budget": budget,
        "arms": list(arms),
        "doses": list(doses),
        "run_offset": ud.RUN_OFFSET,
        "smoke": smoke,
        "vector_variant": "raw (no within-primitive mean subtraction)",
    }
    state = out / "generation.json"
    if state.exists():
        if cp._json(state) != config:
            raise ValueError("generation inputs/settings changed; choose a new output folder")
    else:
        out.mkdir(parents=True, exist_ok=True)
        cp._write_json(state, config)

    tokens = ud.load_tokens(root)
    gold = ud.load_gold(root)
    prompt_lens = inject.prompt_lengths(tokens)
    spec = active_model()
    eos = set(spec.eos_ids)

    lm = None
    rows = []
    for _, target in work.iterrows():
        item = target["item_id"]
        path = shard_path(out, item)
        if path.exists():
            item_rows = cp._json(path)
        else:
            if lm is None:
                lm = load_model(spec.model, revision=spec.revision, device_map=device)
            trace = ud.build_trace(tokens, item)
            prefix_len = ud.injection_prefix_len(trace, int(target["solved_run_index"]))
            ids = inject.load_ids(tokens, item)
            prefix_text = lm.tokenizer.decode(ids[:prefix_len], skip_special_tokens=False)
            item_rows = []
            for arm, dose in plan:
                vector = None if arm == "control" else arm_specs[arm][0]
                dose_unit = 0.0 if arm == "control" else arm_specs[arm][1]
                alpha = dose * dose_unit
                result = inject.resume_generate(
                    lm, ids, prefix_len, vector=vector, alpha=alpha, layer=layer,
                    schedule=schedule, budget=budget, eos_ids=eos,
                    prompt_len=int(prompt_lens[item]))
                stats = pilot.tail_stats(lm.tokenizer, result["generated"], prefix_text,
                                         gold[item])
                item_rows.append(dict(
                    item_id=item, arm=arm, dose=dose, alpha=alpha, layer=layer,
                    schedule=schedule, budget=budget, prefix_len=prefix_len,
                    solved_run_index=int(target["solved_run_index"]),
                    runs_after_solved=float(target["runs_after_solved"]),
                    stop_reason=result["stop_reason"], n_generated=len(result["generated"]),
                    generated_ids=list(result["generated"]),
                    terminated=result["stop_reason"] == "eos",
                    applied=result["applied"], mean_arrived_norm=result["mean_arrived_norm"],
                    **stats,
                ))
                print(f"{item} {arm:9s} dose {dose:>4} -> {result['stop_reason']:6s} "
                      f"{len(result['generated']):4d} tok think_closed={stats['closed_think']}",
                      flush=True)
            cp._write_json(path, item_rows)
        rows.extend(item_rows)
    if lm is not None:
        import torch

        del lm
        torch.cuda.empty_cache()

    if task_indices is not None:
        print(f"item indices {task_indices} complete: {len(rows)} rows written/verified; "
              "pack not built (rerun with no --task-indices once every subset has finished).",
              flush=True)
        return None
    return write_pack(root, out, rows, config)


def write_pack(root: Path, out: Path, rows: list[dict], config: dict) -> Path:
    pack = out / "pack"
    pack.mkdir(exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, pack / "results.parquet")
    manifest = {
        "format": "stop-unconfounded-injection-v1",
        "n_rows": len(rows),
        "n_items": len({r["item_id"] for r in rows}),
        "config": config,
        "code_sha256": {
            f"code/dev/schema_llm/explore/stop_steering/{f}": cp._sha(
                root / "code/dev/schema_llm/explore/stop_steering" / f)
            for f in ("inject.py", "pilot.py", "gate.py", "runframe.py", "unconfounded_data.py")
        },
    }
    cp._write_json(pack / "manifest.json", manifest)
    cp._write_json(
        out / "complete.json",
        {"pack_sha256": cp._sha(pack / "manifest.json"), "rows": len(rows)},
    )
    print(json.dumps({"verified": str(pack), "rows": len(rows),
                      "n_items": manifest["n_items"]}, indent=2))
    return pack


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[5])
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=("stop", "shuffled", "solved"))
    p.add_argument("--doses", nargs="+", type=float, default=list(DOSES))
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--schedule", default=SCHEDULE, choices=("sustained", "once"))
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--population-size", action="store_true",
                   help="print the injectable population size and exit (for Slurm array sizing)")
    p.add_argument(
        "--task-indices",
        help="only inject these 0-based positions in the population, e.g. '0-4,10,15-19'; "
        "skips building the pack. Omit for the full run (default).",
    )
    args = p.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    from ..reproduction import verify

    verify(root)
    population = ud.load_population(root)
    if args.population_size:
        print(len(population))
        return
    task_indices = None
    if args.task_indices:
        try:
            task_indices = parse_task_indices(args.task_indices, len(population))
        except ValueError as e:
            p.error(str(e))
    if args.check_only:
        ud.load_tokens(root)
        ud.load_vectors(root)
        ud.load_gold(root)
        if "shuffled" in args.arms:
            require((root / SHUFFLED_NPZ).exists(),
                    f"missing {SHUFFLED_NPZ}; run unconfounded_capture.py first")
        print(f"Unconfounded injection inputs verified: {len(population)} items, "
              f"{len(args.arms) * len(args.doses) + 1} arm/dose combos each "
              "(incl. control); no weights/GPU used.")
        return
    import torch

    if not torch.cuda.is_available():
        raise ValueError(
            "injection requires NVIDIA CUDA; use --check-only for input validation"
        )
    out = (args.out or root / "results/stop-unconfounded-injection").resolve()
    pack = run(
        root, out, args.device, tuple(args.arms), tuple(args.doses),
        min(args.budget, 64) if args.smoke else args.budget,
        args.layer, args.schedule, args.smoke, task_indices,
    )
    if pack is None:
        print(f"Item indices {task_indices} done; rerun with no --task-indices once every "
              "subset finishes to build the pack.")
    else:
        print(f"Pack ready: {pack}")


if __name__ == "__main__":
    main()
