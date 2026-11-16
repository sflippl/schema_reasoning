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
`results/stop-unconfounded-prompt-injection/` by default.
    