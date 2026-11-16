"""Build the shuffled-label null for the unconfounded corpus's STOP-approach direction.

The supplied ``stop_approach_direction_phi4mr_unconfounded.npz`` ships ``u_raw_residual`` (the
real STOP-approach direction, uncentred/"raw") and a ``solved`` specificity control, but no
shuffled-label null -- that requires raw layer-16 residual states at the same S/B positions,
which weren't shipped, only the final aggregated vectors. This module reconstructs those states
by forward-passing each trace once (teacher-forced, no generation) and capturing the layer-16
block-output residual at every run end, then:

  1. Recomputes the REAL ``stop`` contrast the same way (same S/B masks, same features) and
     checks it against the supplied ``u_raw_residual``/``contrast_raw_residual`` -- a parity gate
     for this whole reconstruction, before trusting anything built from it.
  2. Builds the shuffled-label contrast (``gate.shuffled_labels`` + ``gate.contrast``'s own
     difference-of-means, the same construction the original pipeline's null used) as the
     actual deliverable.

Uses the 89 ``reaches_solved`` items (``unconfounded_data.load_population(..., 0)``), not the
original analysis's full 297 terminated traces: there's no cheap way to determine termination for
the ~211 D1 items outside the t_solved csv without the original response-level export, and the
89-item judge-all-right/solved set is directly relevant to the traces this experiment injects
into anyway. This is a smaller S/B pool than the original (888 S / 6216 B), so exact numeric
agreement isn't expected -- the parity check below is about DIRECTION, not magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...core.model import load_model
from ..head_fv_models import active as active_model
from . import gate, inject, runframe
from . import unconfounded_data as ud

SEED = 20260919  # matches inject.py's own SEED, for consistency across this codebase
LAYER = 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capture_trace_states(lm, layer: int, ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Layer-``layer`` block-output residual at each of ``positions``, one bulk forward pass."""
    module = lm.model.get_submodule(lm.layer_hook_names([layer])[0])
    captured: dict = {}

    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().float().cpu()

    head = torch.from_numpy(np.ascontiguousarray(ids))[None].to(lm.device)
    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            lm.model(input_ids=head, attention_mask=torch.ones_like(head),
                     position_ids=torch.arange(len(ids), device=lm.device)[None],
                     use_cache=False)
    finally:
        handle.remove()
    hidden = captured["hidden"][0].numpy()
    return hidden[positions]


def build(root: Path, out: Path, device="cuda:0", layer: int = LAYER) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    population = ud.load_population(root, min_runs_after_solved=0)
    tokens = ud.load_tokens(root)
    supplied = ud.load_vectors(root)

    spec = active_model()
    require(spec.key == "phi4mr", "unconfounded corpus is phi4mr-only")
    lm = load_model(spec.model, revision=spec.revision, device_map=device)

    frames, state_parts = [], []
    for item in population["item_id"]:
        trace = ud.build_trace(tokens, item)
        frame = runframe.add_stop_masks(runframe.seg_frame(trace, fold=0))
        ids = inject.load_ids(tokens, item)
        end = int(frame["row_end"].max()) + 1
        states = capture_trace_states(lm, layer, ids[:end], frame["row_end"].values)
        require(len(states) == len(frame), f"{item}: state/frame row mismatch")
        frames.append(frame)
        state_parts.append(states)
        print(f"captured {item}: {len(frame)} runs", flush=True)
    del lm
    torch.cuda.empty_cache()

    big_frame = pd.concat(frames, ignore_index=True)
    states = np.concatenate(state_parts, axis=0).astype(np.float32)
    require(len(big_frame) == len(states), "frame/state row mismatch after concat")

    positive = big_frame["last3"].values
    negative = big_frame["ref_stop"].values
    raw_diff = states[positive].mean(0) - states[negative].mean(0)
    u_mine = gate.unit(raw_diff)
    cos_vs_supplied = float(u_mine @ supplied["u_raw_residual"])

    rng = np.random.default_rng(SEED)
    shuffled_pos, shuffled_neg = gate.shuffled_labels(big_frame, positive, negative, rng)
    shuffled_diff = states[shuffled_pos].mean(0) - states[shuffled_neg].mean(0)
    u_shuffled = gate.unit(shuffled_diff)
    pool = positive | negative
    sd_shuffled_along_u = float(np.std(states[pool] @ u_shuffled))

    np.savez(out / "shuffled_direction_phi4mr_unconfounded.npz",
            u_shuffled_residual=u_shuffled.astype(np.float32),
            contrast_shuffled_residual=shuffled_diff.astype(np.float32),
            sd_shuffled_along_u=sd_shuffled_along_u)
    report = dict(
        n_items=int(len(population)), n_runs=int(len(big_frame)),
        n_S=int(positive.sum()), n_B=int(negative.sum()),
        cos_raw_reconstruction_vs_supplied=cos_vs_supplied,
        raw_diff_norm_mine=float(np.linalg.norm(raw_diff)),
        shuffled_diff_norm=float(np.linalg.norm(shuffled_diff)),
        sd_shuffled_along_u=sd_shuffled_along_u,
        seed=SEED, layer=layer,
    )
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    require(cos_vs_supplied > 0.5,
            f"reconstructed raw contrast diverges from the supplied vector "
            f"(cos={cos_vs_supplied:.3f}); do not trust the shuffled null below this gate")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[5])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "data/stop_solved_unconfounded/derived").resolve()
    if args.check_only:
        population = ud.load_population(root, min_runs_after_solved=0)
        ud.load_tokens(root)
        ud.load_vectors(root)
        print(f"Unconfounded capture inputs verified: {len(population)} reaches_solved items; "
              "no weights/GPU used.")
        return
    build(root, out, device=args.device, layer=args.layer)


if __name__ == "__main__":
    main()
