# New D1 meta-clusters for Sam (Phi-4-mini-reasoning and Qwen3-4B)

File: `binding_dynamics-phi4miniR_Qwen34B-v2-tokens-NEW-metaclusters-20261113run.parquet`
(1,580,359 rows; SHA-256 `ee5d1b1d8d9882386460d7c827d5fe136675ef00085c1a33a363eee836fc665e`).

This has the same rows, in the same order, as Sam's `binding_dynamics-phi4miniR_Qwen34B-v2-tokens.parquet`, which covers 300 D1 items × 2 models.
`metacluster_L8` and `metacluster_L16` now hold the meta-clusters from our 2026-11-13 primitive run.

## Columns

| Column | Meaning |
|---|---|
| `item_id` … `segment` | Copied unchanged from Sam's file. |
| `metacluster_L8`, `metacluster_L16` | New meta-cluster ID, 0–7, at layer 8 or 16. NaN on prompt tokens, which is the same coverage as Sam's file. |
| `kmeans_L8`, `kmeans_L16` | New fine-cluster ID (0–49) that each token's meta-cluster was built from. These are extra columns. |

Sam's `merged_primitive_L8` and `merged_primitive_L16` columns are **removed** on purpose. His old merged IDs
don't belong with the new meta-clusters, and our merged themes were built using the
spot-error (core_spot) data. We can add them on request.

## Verified (checked against the files on 2026-11-16)

- **Same responses.** All 600 response texts (300 items × 2 models) are identical between Sam's
  `response_sequences.parquet` and our run's export, and so are the completion-token counts.
- **Same tokens.** All 1,580,359 token rows match one-for-one on item, model and position, with identical
  `token_id`, `prompt_len`, `segment` and `response_position`.
- **Same coverage.** New meta-cluster IDs are on exactly the same rows as Sam's old ones: the 1,130,859
  think and answer tokens. Prompt tokens are NaN.
- **Source.** Our run export, input pack SHA-256 `954a6c2c…2b8b83`. Its saved config for this source records:
  layers 0–28 in steps of 4, k = 50 fine clusters, 8 meta-clusters, seed 0, no PCA, task `identify`, and
  the Phi checkpoint `microsoft/Phi-4-mini-reasoning` at revision `0e3b1e2d…`, bf16.

## Caveats

- **The IDs 0–7 are arbitrary labels.** They don't correspond to Sam's old IDs. Old and new IDs only partly
  line up; the cross-tab is available on request.
- **The fit is per model, on D1 only.** Each model's clusters were fit on that model's D1 responses alone.
  Per an earlier code check, no D2 or spot-error data entered D1's normalization or clustering. As a
  result, IDs also don't line up with our D2, spot-error or TSP clusters.
- **There is no clustering object to share.** The run saved per-token assignments only, not cluster
  centres or normalization parameters, so new tokens (for example from STOP-steered generations) can't
  be assigned to these clusters. Doing that would mean re-running capture and clustering with those saved.
- **Sam's per-response sequence columns are not updated.** The `*_reasoning_primitive_sequence_*` columns in
  his `response_sequences.parquet` are collapsed runs of his **old merged** IDs over think and answer tokens.
  We checked that this reconstructs all 600 exactly. They are not meta-cluster sequences, so they're left
  unchanged.
- **Not verified:** how Sam's old meta-clusters were fit (data pool and settings). So we can't say how
  the new ones differ in method, only that the IDs differ.

## Cluster labels (added)

| File | One row per | Columns |
|---|---|---|
| `d1-fine-cluster-labels-L8-L16.parquet` (200 rows) | Model × layer (8, 16) × fine cluster (0–49) | `model`, `layer`, `cluster_id` (= `kmeans_L*`), `metacluster_id`, `count`, `top` (top tokens), `label`, `description` |
| `d1-meta-cluster-labels-L8-L16.parquet` (32 rows) | Model × layer × meta-cluster (0–7) | `model`, `layer`, `metacluster_id` (= `metacluster_L*`), `n_tokens`, `mean_pos`, `top`, `label`, `description` |

- **How the labels were made.** Each label and description was written by `gpt-5.6-sol_2026-07-09` from that cluster's own D1 tokens and excerpts.
  The saved config gives the prompt context: "These clusters contain tokens from language-model responses to narrative reasoning
  questions…"
- **Labels belong to D1 only.** Matching clusters across datasets by label is what the merge does. The merged-theme column is
  dropped here, because our merge included spot-error data.
- **Verified.** For every model and layer, each fine cluster's `metacluster_id` matches the tokens file. `count` and `n_tokens`
  equal the token counts in the tokens file, for all 50 fine and all 8 meta-clusters.
- **Source.** Our 2026-11-13 export: `fine_clusters.parquet` and `clusters.parquet`, rows for `d1/<model>/identify`.

Our primitives cover both models: Phi-4-mini-reasoning and Qwen3-4B, each on D1, D2 and spot-error, plus TSP.
