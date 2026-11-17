# D2: fine clusters, meta-clusters and labels (Phi-4-mini-reasoning and Qwen3-4B)

These files use the same format as Sam's D1 example files (`binding_dynamics-phi4miniR_Qwen34B-v2-*`). All IDs come from the
per-dataset clustering in our 2026-11-13 run, **before any merge**.

## Files

| File | Contents |
|---|---|
| `d2-phi4miniR_Qwen34B-response_sequences.parquet` | One row per response: 297 items × 2 models = 594 rows. The same 95 columns and column types as Sam's D1 `response_sequences` file, plus 4 new columns (see below). |
| `d2-phi4miniR_Qwen34B-tokens.parquet` | One row per token, 1,628,642 rows. The same layout as the D1 tokens file we sent: `item_id`, `task_family`, `model`, `sample_id`, `position`, `response_position`, `prompt_len`, `token_id`, `token`, `segment`, `metacluster_L8`, `metacluster_L16`, `kmeans_L8`, `kmeans_L16`. |
| `d2-meta-cluster-labels-L8-L16.parquet` | Label, description, top tokens and token count for each meta-cluster (model × layer 8/16 × ID 0–7). |
| `d2-fine-cluster-labels-L8-L16.parquet` | The same for each fine cluster (ID 0–49), plus the meta-cluster it belongs to. |

**Sequence columns in the responses file.**
- Sam's four `*_reasoning_primitive_sequence_*` columns are present but **empty**. In his D1 file those columns hold runs of
  his old *merged* IDs, and our merged themes were built using spot-error data, so we don't fill them.
- They are replaced by four new columns: `phi4mr_metacluster_sequence_L8`, `phi4mr_metacluster_sequence_L16`,
  `qwen34b_metacluster_sequence_L8` and `qwen34b_metacluster_sequence_L16`.
- Each new column holds the run-collapsed sequence of **new meta-cluster IDs** over think and answer tokens. Consecutive
  repeats are collapsed. Each column is filled only on that model's rows, following Sam's convention.

**Ungraded controls.** Use `scoring_exclude` (or `arm`) in the responses file. 99 responses per model are ungraded control
passages: the unresolved, late_un and pairs_un arms.

## How to read the IDs

Cluster numbers are names within one fit (one dataset × one model × one layer), not meanings:
- Look up what a number means in the label tables, matching on `model`, `layer` and the ID.
- D2's meta-cluster 3 is unrelated to D1's or TSP's 3, or to earlier IDs.
- Within a fit, meta-clusters are numbered by where in a response their fine clusters typically occur, earliest first,
  but only loosely.

## Verified (2026-11-17)

- **Response metadata.** Po-Chen's primitives pack, joined the same way, reproduces all 91 non-sequence columns of Sam's
  D1 file for all 600 rows; only storage types differ, and we cast to Sam's types.
- **Response text.** All 594 D2 response texts equal the run export the token IDs came from.
- **Ungraded controls.** `scoring_exclude` is True for exactly the 99 control-arm responses per model.
- **Coverage.** The tokens file assigns IDs on all 1,177,149 think and answer tokens and on no prompt tokens.
- **Labels match the tokens.** The label tables' cluster-to-meta-cluster mapping and token counts match the tokens file, for
  both models and both layers.

## Caveats

- **D2 has 297 items, not 300.** Three passages are missing from Po-Chen's source data, with no recorded cause.
- **Fit on D2 only.** Clusters were fit on D2 only, separately per model.
- **No clustering object.** Cluster centres and normalization parameters weren't saved, so new tokens can't be assigned.
- **Labelling prompt.** Labels were written by `gpt-5.6-sol_2026-07-09`, using the same prompt for every dataset.
