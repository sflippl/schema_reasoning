"""Compare literal prompt-text insertions with an unmodified resumed continuation.

The prompt-only conditions use the unconfounded D1 Phi-4-mini-reasoning population: judge-solved,
terminated traces with more than 30 runs remaining after T_solved. Each item is resumed at the
run end at T_solved and 30 runs later. At each location, the control, literal ``</think>`` and
literal ``Final answer:`` conditions share the same stored prefix and greedy generation budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .. import combined_primitives as cp
from ..tsp_generate import parse_task_indices
from . import unconfounded_data as ud

RUN_OFFSETS = (0, 30)
CORPUS_CAP = 8192
BUDGET = -1
INSERTIONS = (
    ("control", ""),
    ("think_close", "</think>"),
    ("final_answer", "Final answer:"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shard_path(out: Path, item: str) -> Path:
    name = hashlib.sha256(item.encode()).hexdigest()
    return out / "shards" / f"{name}.json"


def encode_insertion(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    require(tokenizer.decode(token_ids, skip_special_tokens=False) == text,
            f"insertion does not round-trip through the model tokenizer: {text!r}")
    return [int(token_id) for token_id in token_ids]


def _config(root: Path, device: str, budget: int, run_offsets: tuple[int, ...],
            population_size: int) -> dict:
    from ..reproduction import _runtime

    return {
        "format": "stop-unconfounded-prompt-injection-v1",
        "corpus": "unconfounded-d1",
        "release": cp._sha(root / "MANIFEST.json"),
        "population_sha256": cp._sha(root / ud.POPULATION_CSV),
        "runtime": _runtime(),
        "device": device,
        "dtype": "bfloat16",
        "budget": budget,
        "corpus_cap": CORPUS_CAP if budget <= 0 else None,
        "population_size": population_size,
        "run_offsets": list(run_offsets),
        "generation": "greedy",
        "insertions": [{"condition": name, "text": text} for name, text in INSERTIONS],
    }


def run(
    root: Path,
    out: Path,
    device: str = "cuda:0",
    budget: int = BUDGET,
    run_offsets: tuple[int, ...] = RUN_OFFSETS,
    task_indices: list[int] | None = None,
) -> Path | None:
    """Generate an item shard, or merge completed shards when ``task_indices`` is omitted."""
    from ..head_fv_models import active as active_model

    population = ud.load_population(root, min_runs_after_solved=max(run_offsets))
    if task_indices is not None:
        bad = [i for i in task_indices if not 0 <= i < len(population)]
        require(not bad, f"item index out of range [0, {len(population)}): {bad}")
        work = population.iloc[task_indices]
    else:
        work = population

    config = _config(root, device, budget, run_offsets, len(population))
    state = out / "generation.json"
    if state.exists():
        if cp._json(state) != config:
            raise ValueError("generation inputs/settings changed; choose a new output folder")
    else:
        out.mkdir(parents=True, exist_ok=True)
        cp._write_json(state, config)

    if task_indices is None:
        missing = [item for item in work.item_id if not shard_path(out, item).exists()]
        require(not missing, f"missing {len(missing)} item shards; first missing: {missing[:5]}")
        rows = [row for item in work.item_id for row in cp._json(shard_path(out, item))]
        expected = {
            (item, offset, condition)
            for item in work.item_id
            for offset in run_offsets
            for condition, _ in INSERTIONS
        }
        actual = [(row["item_id"], row["run_offset"], row["condition"]) for row in rows]
        require(len(actual) == len(expected) and set(actual) == expected,
                "item shards do not contain exactly one row for every item/offset/condition")
        return write_pack(root, out, rows, config)

    from . import inject, pilot

    tokens = ud.load_tokens(root)
    gold = ud.load_gold(root)
    prompt_lens = inject.prompt_lengths(tokens)
    spec = active_model()
    eos = set(spec.eos_ids)
    from ...core.model import load_model

    lm = None
    for _, target in work.iterrows():
        item = target["item_id"]
        path = shard_path(out, item)
        if path.exists():
            continue
        if lm is None:
            lm = load_model(spec.model, revision=spec.revision, device_map=device)
        trace = ud.build_trace(tokens, item)
        ids = inject.load_ids(tokens, item)
        item_rows = []
        encoded = [(name, text, encode_insertion(lm.tokenizer, text))
                   for name, text in INSERTIONS]
        for run_offset in run_offsets:
            prefix_len = ud.injection_prefix_len(
                trace, int(target["solved_run_index"]), run_offset=run_offset)
            prefix_text = lm.tokenizer.decode(ids[:prefix_len], skip_special_tokens=False)
            for condition, text, inserted_ids in encoded:
                completion_prefix_tokens = prefix_len - int(prompt_lens[item])
                budget_tokens = (
                    max(CORPUS_CAP - completion_prefix_tokens - len(inserted_ids), 1)
                    if budget <= 0 else budget
                )
                result = inject.resume_generate(
                    lm, ids, prefix_len, budget=budget_tokens, eos_ids=eos,
                    prompt_len=int(prompt_lens[item]), inserted_ids=inserted_ids,
                )
                tail_ids = inserted_ids + result["generated"]
                stats = pilot.tail_stats(lm.tokenizer, tail_ids, prefix_text, gold[item])
                item_rows.append(dict(
                    item_id=item,
                    run_offset=run_offset,
                    condition=condition,
                    insertion_text=text,
                    inserted_token_ids=inserted_ids,
                    n_inserted_tokens=len(inserted_ids),
                    completion_prefix_tokens=completion_prefix_tokens,
                    budget=budget_tokens,
                    corpus_cap=CORPUS_CAP if budget <= 0 else None,
                    prefix_len=prefix_len,
                    solved_run_index=int(target["solved_run_index"]),
                    runs_after_solved=float(target["runs_after_solved"]),
                    stop_reason=result["stop_reason"],
                    n_generated=len(result["generated"]),
                    generated_ids=result["generated"],
                    terminated=result["stop_reason"] == "eos",
                    **stats,
                ))
                print(f"{item} +{run_offset:>3}runs {condition:12s} -> "
                      f"{result['stop_reason']:6s} {len(result['generated']):4d} tok "
                      f"think_closed={stats['closed_think']}", flush=True)
        cp._write_json(path, item_rows)

    if lm is not None:
        import torch

        del lm
        torch.cuda.empty_cache()
    print(f"item indices {task_indices} complete; merge after all array tasks finish.", flush=True)
    return None


def write_pack(root: Path, out: Path, rows: list[dict], config: dict) -> Path:
    pack = out / "pack"
    pack.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), pack / "results.parquet")
    manifest = {
        "format": "stop-unconfounded-prompt-injection-v1",
        "n_rows": len(rows),
        "n_items": len({row["item_id"] for row in rows}),
        "config": config,
        "code_sha256": {
            f"code/dev/schema_llm/explore/stop_steering/{name}": cp._sha(
                root / "code/dev/schema_llm/explore/stop_steering" / name
            )
            for name in ("inject.py", "prompt_injection.py", "pilot.py", "unconfounded_data.py")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--budget", type=int, default=BUDGET,
        help=f"generated-token budget per continuation; <=0 uses the remaining "
             f"{CORPUS_CAP}-token corpus completion cap after the prefix and inserted text",
    )
    parser.add_argument("--run-offsets", nargs="+", type=int, default=list(RUN_OFFSETS))
    parser.add_argument("--population-size", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--task-indices",
                        help="0-based item positions, e.g. '0-4,10'; omit only to merge shards")
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    from ..reproduction import verify

    verify(root)
    require(args.run_offsets and min(args.run_offsets) >= 0,
            "run offsets must be non-negative and include at least one value")
    population = ud.load_population(root, min_runs_after_solved=max(args.run_offsets))
    if args.population_size:
        print(len(population))
        return
    task_indices = None
    if args.task_indices:
        try:
            task_indices = parse_task_indices(args.task_indices, len(population))
        except ValueError as error:
            parser.error(str(error))
    if args.check_only:
        ud.load_tokens(root)
        ud.load_gold(root)
        print(f"Prompt-injection inputs verified: {len(population)} items, "
              f"{len(args.run_offsets)} offsets x {len(INSERTIONS)} conditions each; "
              "no weights/GPU used.")
        return
    if task_indices is not None:
        import torch

        if not torch.cuda.is_available():
            raise ValueError("prompt-injection generation requires NVIDIA CUDA")
    out = (args.out or root / "results/stop-unconfounded-prompt-injection").resolve()
    run(root, out, device=args.device, budget=args.budget,
        run_offsets=tuple(args.run_offsets), task_indices=task_indices)


if __name__ == "__main__":
    main()
