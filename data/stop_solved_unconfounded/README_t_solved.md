# T_solved for Phi-4-mini-reasoning — corpus unconfounded

VALID: counterbalanced corpus (gender evenly spread over roles, shared name pool).

t_solved_per_trace_phi4mr_unconfounded.csv — one row per trace on which the judge accepted every role (105 of 300 phi responses).
  solved_token_index / solved_run_index   first think-segment token / run at which every gold person-role pair has been stated
                                          (name within 15 tokens of an accepted word of the person's gold role; passage quotes excluded;
                                          monotone). 0-based within the think segment, '</think>' marker tokens stripped. A run is a maximal
                                          stretch of tokens with one layer-16 metacluster label (2.60 tokens per run here).
  solved_position_abs                     the same token as an absolute position in tokens.parquet (prompt + response); use this to inject
  end_token_index / end_run_index / end_position_abs   last reasoning token (the run before '</think>', or the last token before the cap)
  tokens_after_solved / runs_after_solved  how long the model kept reasoning after the problem was solved
  terminated                              True = emitted '</think>'; False = hit the generation cap
Validation on the answer segment: the extractor recovers 0.951 of the pairs the judge accepted.
Injection position per the protocol: the run end 30 runs after solved_run_index (sustained from there), or solved_position_abs + 30 * 2.6 tokens.
