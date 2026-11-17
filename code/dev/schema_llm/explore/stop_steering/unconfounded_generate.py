"""Inject the primitive-centered core STOP direction (and optionally its supplied shuffled null).

Parallel to ``tsp_generate.py`` in shape (resumable ``--task-indices`` shards over an item
population, a separate merge step building the final pack), but injecting into the unconfounded
corpus's own 84-item population (``unconfounded_data.load_population``) instead of generating new
TSP responses. Reuses the existing resume/injection harness (``inject.resume_generate``) and continuation
diagnostics (``pilot.tail_stats``) -- only the vector, dose unit, population and injection position
are specific to this corpus.

Vector: ``u_stop`` from ``data/stop_vectors/phi4mr/stop_vector_phi4mr.npz``. This is the core
direction with primitive/metacluster means subtracted; its dose unit is the paired ``dose_sd``.
At each generated token, inject ``alpha * dose_sd * u_stop``. A no-injection resumed control is
included for every location.
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

ARMS = ("stop",)
DOSES = (0.25, 0.5)
LAYER = 16
SCHEDULE = "sustained"
CORPUS_CAP = 8192
BUDGET = -1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shard_path(out: Path, item: str) -> Path:
    name = hashlib.sha256(item.encode()).hexdigest()
    return out / "shards" / f"{name}.json"


def arm_vector_and_dose_unit(root: Path, arm: str, supplied: dict) -> tuple[np.ndarray, float]:
    """Return a unit vector and its supplied residual-scale dose unit."""
    if arm == "stop":
        vector = supplied["u_stop"]
        dose_unit = float(supplied["dose_sd"])
        require(np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5),
                "core STOP vector u_stop is not unit norm")
        require(supplied["model"].item() == "phi4mr", "STOP vector model is not phi4mr")
        require(supplied["layer"].item() == f"L{LAYER}",
                f"STOP vector layer must be L{LAYER}")
        return vector, dose_unit
    if arm == "shuffled":
        vector = supplied["u_stop_shuffled"]
        dose_unit = float(supplied["dose_sd_shuffled"])
        require(np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5),
                "shuffled STOP vector is not unit norm")
        return vector, dose_unit
    raise ValueError(f"unknown arm: {arm}")


def completion_budget(response_prefix_tokens: int, budget: int = BUDGET) -> int:
    if budget > 0:
        return budget
    require(response_prefix_tokens < CORPUS_CAP,
            f"response prefix already reaches the {CORPUS_CAP}-token cap")
    return CORPUS_CAP - response_prefix_tokens


RUN_OFFSETS = (0, 30)  # T_solved itself, and 30 runs after it (the supplied protocol's default)


def run(
    root: Path,
    out: Path,
    device: str,
    arms: tuple[str, ...] = ARMS,
    doses: tuple[float, ...] = DOSES,
    budget: int = BUDGET,
    layer: int = LAYER,
    schedule: str = SCHEDULE,
    run_offsets: tuple[int, ...] = RUN_OFFSETS,
    smoke: bool = False,
    task_indices: list[int] | None = None,
) -> Path | None:
    """Generate this job's explicit subset of items, then build the pack iff it's everything.

    Mirrors ``tsp_generate.run``: ``task_indices`` selects which population rows THIS process
    injects into; it is not part of ``config``, so partial runs over different subsets of the
    SAME population/settings keep validating against each other. The pack is only built when
    ``task_indices`` is ``None`` (the default, and the final merge invocation).

    ``run_offsets`` injects at EVERY requested run-offset after T_solved for the SAME items
    (default both 0 -- immediately at T_solved -- and 30, the supplied protocol's own choice),
    so the two conditions are directly paired per item rather than needing separate runs/output
    directories. The population is filtered for feasibility at ``max(run_offsets)`` (the most
    restrictive offset), so every item in it is guaranteed usable at every smaller offset too.
    """
    from ...core.model import load_model
    from ..head_fv_models import active as active_model
    from ..reproduction import _runtime

    population = ud.load_population(root, min_runs_after_solved=max(run_offsets))
    require(layer == LAYER, f"the supplied core STOP vector is for layer {LAYER}, not {layer}")
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
        "runtime": _runtime(),
        "device": device,
        "dtype": "bfloat16",
        "layer": layer,
        "schedule": schedule,
        "budget": budget,
        "completion_cap": CORPUS_CAP if budget <= 0 else None,
        "arms": list(arms),
        "doses": list(doses),
        "run_offsets": list(run_offsets),
        "smoke": smoke,
        "vector_file": str(ud.VECTORS_PATH),
        "vector_key": "u_stop",
        "vector_variant": "core (primitive/metacluster-centered)",
        "dose_unit_key": "dose_sd",
        "dose_unit": arm_specs["stop"][1] if "stop" in arm_specs else None,
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
    task_records = ud.load_task_records(root)
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
            ids = inject.load_ids(tokens, item)
            task = task_records[item]
            task_prompt = lm.tokenizer.decode(
                ids[:int(prompt_lens[item])], skip_special_tokens=False
            )
            item_rows = []
            for run_offset in run_offsets:
                prefix_len = ud.injection_prefix_len(
                    trace, int(target["solved_run_index"]), run_offset=run_offset)
                prefix_text = lm.tokenizer.decode(ids[:prefix_len], skip_special_tokens=False)
                completion_prefix_tokens = prefix_len - int(prompt_lens[item])
                budget_tokens = completion_budget(completion_prefix_tokens, budget)
                for arm, dose in plan:
                    vector = None if arm == "control" else arm_specs[arm][0]
                    dose_unit = 0.0 if arm == "control" else arm_specs[arm][1]
                    injection_scale = dose * dose_unit
                    result = inject.resume_generate(
                        lm, ids, prefix_len, vector=vector, alpha=injection_scale, layer=layer,
                        schedule=schedule, budget=budget_tokens, eos_ids=eos,
                        prompt_len=int(prompt_lens[item]))
                    stats = pilot.tail_stats(lm.tokenizer, result["generated"], prefix_text,
                                             gold[item])
                    # The legacy pair-completeness metric expects `canonical`, absent from D1 gold.
                    stats.pop("pairs_complete", None)
                    response_ids = (
                        ids[int(prompt_lens[item]):prefix_len].tolist()
                        + list(result["generated"])
                    )
                    response_text = lm.tokenizer.decode(
                        response_ids, skip_special_tokens=False
                    )
                    item_rows.append(dict(
                        item_id=item, task_record=task, task_prompt=task_prompt,
                        run_offset=run_offset, arm=arm, dose=dose, alpha=dose,
                        dose_unit=dose_unit, injection_scale=injection_scale,
                        layer=layer, schedule=schedule, budget=budget_tokens,
                        budget_setting=budget, prefix_len=prefix_len,
                        completion_prefix_tokens=completion_prefix_tokens,
                        corpus_cap=CORPUS_CAP if budget <= 0 else None,
                        solved_run_index=int(target["solved_run_index"]),
                        runs_after_solved=float(target["runs_after_solved"]),
                        stop_reason=result["stop_reason"], n_generated=len(result["generated"]),
                        generated_ids=list(result["generated"]),
                        response_text=response_text,
                        terminated=result["stop_reason"] == "eos",
                        applied=result["applied"], mean_arrived_norm=result["mean_arrived_norm"],
                        **stats,
                    ))
                    print(f"{item} +{run_offset:>3}runs {arm:9s} alpha {dose:>4} -> "
                          f"{result['stop_reason']:6s} {len(result['generated']):4d} tok "
                          f"think_closed={stats['closed_think']}", flush=True)
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
    task_rows = {}
    result_rows = []
    for row in rows:
        require("task_record" in row and "task_prompt" in row and "response_text" in row,
                "shards do not include raw responses and task records; choose a fresh output "
                "directory and regenerate")
        row = dict(row)
        item = row["item_id"]
        record = row.pop("task_record")
        prompt = row.pop("task_prompt")
        task_row = {
            **record,
            "source_item_id": record["item_id"],
            "item_id": item,
            "task_prompt": prompt,
        }
        if item in task_rows:
            require(task_rows[item] == task_row,
                    f"{item}: inconsistent task record or prompt across responses")
        else:
            task_rows[item] = task_row
        result_rows.append(row)
    pq.write_table(pa.Table.from_pylist(result_rows), pack / "results.parquet")
    pq.write_table(
        pa.Table.from_pylist([task_rows[item] for item in sorted(task_rows)]),
        pack / "tasks.parquet",
    )
    manifest = {
        "format": "stop-unconfounded-injection-v2",
        "n_rows": len(rows),
        "n_items": len({r["item_id"] for r in rows}),
        "config": config,
        "code_sha256": {
            f"code/dev/schema_llm/explore/stop_steering/{f}": cp._sha(
                root / "code/dev/schema_llm/explore/stop_steering" / f)
            for f in ("inject.py", "pilot.py", "gate.py", "runframe.py",
                      "unconfounded_data.py", "unconfounded_generate.py")
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
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=("stop", "shuffled"))
    p.add_argument("--doses", nargs="+", type=float, default=list(DOSES))
    p.add_argument(
        "--budget", type=int, default=BUDGET,
        help=f"generated-token budget; <=0 uses the remaining capacity to the "
             f"{CORPUS_CAP}-token completion cap",
    )
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--schedule", default=SCHEDULE, choices=("sustained", "once"))
    p.add_argument(
        "--run-offsets", nargs="+", type=int, default=list(RUN_OFFSETS),
        help="runs after T_solved to inject at (0 = immediately at T_solved); every item gets "
        "injected at EVERY requested offset, so conditions are paired per item. The population "
        "is filtered for feasibility at the largest requested offset.",
    )
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
    require(args.layer == LAYER,
            f"the supplied core STOP vector is for layer {LAYER}, not {args.layer}")
    root = args.root.resolve()
    os.chdir(root)
    from ..reproduction import verify

    verify(root)
    population = ud.load_population(root, min_runs_after_solved=max(args.run_offsets))
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
        supplied = ud.load_vectors(root)
        for arm in args.arms:
            arm_vector_and_dose_unit(root, arm, supplied)
        ud.load_task_records(root)
        print(f"Unconfounded injection inputs verified: {len(population)} items, "
              f"{len(args.run_offsets)} run-offsets x "
              f"{len(args.arms) * len(args.doses) + 1} arm/dose combos each "
              "(incl. control); no weights/GPU used.")
        return
    import torch

    if not torch.cuda.is_available():
        raise ValueError(
            "injection requires NVIDIA CUDA; use --check-only for input validation"
        )
    out = (args.out or root / "results/stop-unconfounded-injection-v2").resolve()
    pack = run(
        root, out, args.device,
        arms=tuple(args.arms), doses=tuple(args.doses),
        budget=min(args.budget, 64) if args.smoke else args.budget,
        layer=args.layer, schedule=args.schedule,
        run_offsets=tuple(args.run_offsets), smoke=args.smoke, task_indices=task_indices,
    )
    if pack is None:
        print(f"Item indices {task_indices} done; rerun with no --task-indices once every "
              "subset finishes to build the pack.")
    else:
        print(f"Pack ready: {pack}")


if __name__ == "__main__":
    main()
