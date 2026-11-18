# STOP-approach direction and T_solved for Phi-4-mini-reasoning, layer 16 — corpus unconfounded

    VALID: counterbalanced corpus 20260915-binding_dyn_d1_v3w2.

    Definitions (STOP_STEERING_PROTOCOL.md section 3; paper Appendix D): a run is a maximal stretch of think-segment tokens with one layer-16
    metacluster label (median about 2 tokens); S = run ends 0-2 runs before the last reasoning run (the run before the model's own </think>);
    B = run ends 20-40 runs before it; terminated Phi traces only; each run-end state minus the mean state of its own metacluster (over all
    run ends of the terminated traces, 99.5th-percentile norm cut); u = unit(mean_S - mean_B).

    Files
      stop_approach_direction_phi4mr_unconfounded.npz
        u_within_primitive_residual        unit STOP-approach direction in the layer-16 residual stream (3072 dims)   <- inject this
        contrast_within_primitive_residual the unnormalised contrast (norm 11.87)
        sd_within_primitive_along_u        one SD of the centred S∪B states along u: 6.28 residual units (the dose unit)
        u_within_primitive_residual_delta12, contrast_..._delta12, sd_..._delta12   variant with S = runs 1-2 before the end (the last run excluded; cos with the main direction 0.618)
        u_raw_residual, contrast_raw_residual, sd_raw_along_u   the same without within-primitive centring (cos with the centred one 0.840)
        u_solved_residual, contrast_solved_residual   companion control: run ends in the 30 runs after T_solved minus the 30 runs before (cos with u_STOP 0.135)
        metacluster_means, metacluster_ids            the per-metacluster means that were subtracted
      stop_approach_direction_phi4mr_unconfounded.csv      the three unit directions as columns
      t_solved_per_trace_phi4mr_unconfounded.csv           T_solved per judge-all-right trace: solved_token_index / solved_run_index (0-based in the
                                                      think segment, marker tokens stripped), solved_position_abs (absolute token position in the
                                                      prompt+response sequence, as in tokens.parquet), end indices, tokens_after_solved, terminated
      summary.csv

    Dose: add alpha * sd_within_primitive_along_u * u_within_primitive_residual to the layer-16 residual at every generated token from
    T_solved + 30 runs (sustained schedule, protocol section 4); alpha = 1 is one within-primitive SD. Validation here: held-out AUC of the
    projection on u for S against B, five passage folds, 0.882 (label-permutation null 0.501); solved direction 0.626.
    Cosine with the direction computed on the 2026-08-24 identify export: 0.296 (within-primitive), 0.386 (raw).

## Current core-vector injection

The current Slurm runner uses `data/stop_vectors/phi4mr/stop_vector_phi4mr.npz`, specifically
`u_stop` and its matching `dose_sd` (5.889 residual units), not the older
`u_raw_residual` vector described above. The supplied `u_stop` has unit norm and is exactly aligned
with the saved `contrast`; its provenance identifies the D1 metacluster-L16 token export. This is
the primitive-centered/core vector, with primitive means removed. The runner checks the vector's
model and layer before generation.

The default experiment is Phi-4-mini-reasoning only, at layer 16, with a sustained injection at
the two paired run offsets: 0 (at `T_solved`) and 30 runs after `T_solved`. Each location includes
an unmodified resumed control and core-vector conditions at `alpha=0.25` and `alpha=0.5`; the
actual injected residual scale is `alpha * dose_sd`. It uses the 84 eligible judge-solved traces
and, by default, generates up to the remaining capacity of the original 8,192-token completion
cap. This is 84 × 2 × 3 = 504 continuations.

The array and merge jobs produce `pack/results.parquet` with the full unmodified model response in
`response_text` and `pack/tasks.parquet` with the matching prompt and D1 task record/gold. No
TRAPI or accuracy-judge stage is required. The default destination is
`results/stop-unconfounded-injection-v2/`.

## Prompt-text alternatives

`scripts/slurm/submit_stop_unconfounded_prompt.sh` submits the prompt-only counterpart without
loading or waiting for the STOP vector. It uses the 84 judge-solved, terminated traces with more
than 30 reasoning runs after `T_solved`, and pairs each item across two insertion locations: the
run end at `T_solved` (offset 0) and the run end 30 runs later.

At each location, the three conditions are:

* **control:** no inserted text;
* **think-close:** append the literal `</think>` to the resumed context;
* **final-answer:** append the literal `Final answer:` to the resumed context.

The stored prompt and reasoning prefix are replayed unchanged up to the selected run end. The
insertion is then teacher-forced into the model context, and the model continues greedily up to
the remaining portion of the original 8,192-token completion cap. The remaining generation budget
is computed per item, offset, and condition as 8,192 minus the already-generated response prefix
and inserted text tokens; the no-insertion control uses only the response prefix in that
calculation. This keeps the full prefix, insertion, and generated continuation within the cap.
Thus the default experiment produces 84 items × 2 locations × 3 conditions = 504 paired
continuations. The per-row `budget` records the resulting generation limit. The array job writes
item shards; its dependent merge job produces `pack/results.parquet` and a manifest under
`results/stop-unconfounded-prompt-injection-v2/` by default, keeping the raw-response pack separate
from any prior output. `pack/results.parquet` contains each
unmodified response string in `response_text`, including the stored reasoning prefix, literal
insertion, and newly generated continuation. `pack/tasks.parquet` contains one row per item with
the exact tokenized task prompt and the original D1 task record, including the gold role key and
task-design metadata. These files are provided for external/manual evaluation; no accuracy judge
is run by the Slurm workflow. The legacy `pairs_complete` diagnostic is omitted because it expects
a different gold schema and would be misleading for D1.

The **control** condition is the paired no-insertion baseline at each location; there is no
additional, separately generated baseline arm.

## Qwen3-4B

With the same activated environment and Slurm setup, submit the Qwen STOP-vector and prompt
experiments separately:

```bash
bash scripts/slurm/submit_stop_unconfounded_qwen.sh
bash scripts/slurm/submit_stop_unconfounded_prompt_qwen.sh
```

Each submits an array and a dependent merge. These wrappers set `SCHEMA_HEAD_FV_MODEL=qwen34b`,
use the Qwen-specific T_solved table and layer-16 STOP vector, and write to
`results/stop-unconfounded-injection-qwen34b-v2/` and
`results/stop-unconfounded-prompt-injection-qwen34b-v2/`, respectively. The standard
`STOP_OUT`, `STOP_PROMPT_OUT`, dose, offset, budget, account, and partition overrides still apply.
The default offsets yield 118 eligible Qwen items and 708 rows per experiment. This is a
model-specific judge-solved population, not the same 84 Phi traces; the populations should not
be treated as item-paired without an explicit join.

The release manifest changes with this model support. Existing vector-run output directories
created against the previous manifest cannot be resumed or re-merged with the updated code:
their saved `generation.json` pins the previous release hash. Keep those packs for analysis,
and use a fresh output directory for a new vector run.

To check the supplied Qwen table and its run coordinates against the token data before submitting,
run:

```bash
SCHEMA_HEAD_FV_MODEL=qwen34b PYTHONPATH=code \
  python -m dev.schema_llm.explore.stop_steering.unconfounded_generate --check-only
SCHEMA_HEAD_FV_MODEL=qwen34b PYTHONPATH=code \
  python -m dev.schema_llm.explore.stop_steering.prompt_injection --check-only
```
    