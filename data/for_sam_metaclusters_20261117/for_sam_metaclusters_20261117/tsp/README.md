# TSP fine clusters and meta-clusters for Sam (Phi-4-mini-reasoning and Qwen3-4B)

File: `tsp-phi4miniR_Qwen34B-tokens-metaclusters-20261116run.parquet`.
It has 2,627,535 rows. SHA-256: `7860a7f4a39879352f1be032d88512ae78973243fd8db965ecf4954739544fff`.

The layout matches the D1 file we sent: one row per token for the 200 TSP responses, 100 tasks × 2 models.
These IDs come from the per-dataset clustering, **before any merge**. No merged/theme columns are included.

## Columns

| Column | Meaning |
|---|---|
| `item_id`, `model`, `uid` | `item_id` is `tsp__<uid>`, the same `item_id`/`uid` as the collaborator's TSP `responses.parquet`. |
| `task_family`, `sample_id`, `position`, `response_position`, `prompt_len`, `token_id`, `token`, `segment` | Same meaning as in the D1 file. `segment` is prompt, think or answer. |
| `metacluster_L8`, `metacluster_L16` | Meta-cluster ID (0–7) at layer 8 or 16. Empty on prompt tokens. |
| `kmeans_L8`, `kmeans_L16` | Fine-cluster ID (0–49) at layer 8 or 16. Empty on prompt tokens. |

## Verified (2026-11-16)

- **Source.** Taken from the TSP run's own export (`joint-schema-tsp/export/tsp/tokens.parquet` on our blob storage; input pack
  SHA-256 `657f3c35…04232c3`). Only the columns above are kept.
- **Matches the raw clustering output.** For both models at layers 8 and 16, every assigned token's fine-cluster and
  meta-cluster ID equals the raw clustering output on GCR138 (`extracted/nodes/tsp/<model>/layer_<L>.npz`). There are zero mismatches over
  1,315,082 Phi and 1,139,985 Qwen response tokens.
- **Coverage.** IDs are assigned on all think and answer tokens (2,455,067) and on no prompt tokens. This is the same coverage
  rule as the D1 file.
- **Matches the collaborator's data.** All 200 responses match the collaborator's TSP `responses.parquet` on `item_id`, `model`, `uid`,
  `prompt_len` and exact response text.
- **Clustering settings.** Saved config for each model: 50 fine clusters, 8 meta-clusters, seed 0, no PCA, task `tsp`. These are the same
  settings as the schema run.

## Caveats

- **Token IDs were reconstructed, not taken from generation.** The collaborator's original sampled token IDs weren't
  available. They were rebuilt from the response text with the pinned tokenizers and chat templates. One Phi response
  (UID 49) comes out at 6,131 tokens vs the collaborator's recorded 6,132.
- **One Phi response hit the token cap.** It is included.
- **Fit on TSP only.** Clusters were fit on TSP only, separately per model. The IDs 0–7 don't correspond to the D1 IDs or to
  anyone's earlier IDs.
- **No clustering object.** Only per-token assignments were saved, not cluster centres or normalization parameters, so
  new tokens can't be assigned.
- **The labelling prompt says "narrative reasoning questions".** That wording was kept unchanged from the schema run so labels are
  comparable, even though TSP isn't narrative. See below.

## Cluster labels (added)

| File | One row per | Columns |
|---|---|---|
| `tsp-fine-cluster-labels-L8-L16.parquet` (200 rows) | Model × layer (8, 16) × fine cluster (0–49) | `model`, `layer`, `cluster_id` (= `kmeans_L*`), `metacluster_id`, `count`, `top` (top tokens), `label`, `description` |
| `tsp-meta-cluster-labels-L8-L16.parquet` (32 rows) | Model × layer × meta-cluster (0–7) | `model`, `layer`, `metacluster_id` (= `metacluster_L*`), `n_tokens`, `mean_pos`, `top`, `label`, `description` |

- **How the labels were made.** Each label and description was written by `gpt-5.6-sol_2026-07-09` from that cluster's TSP tokens and excerpts. The
  saved config gives the prompt context: "These clusters contain tokens from language-model responses to narrative reasoning
  questions. Describe the function supported by the token evidence and excerpts. Do not assume a particular question, dataset,
  model, or correct answer."
- **Labels belong to TSP only.** Matching TSP clusters to D1 clusters by label is what a merge does. That is a separate,
  later step, and nothing from it is included here.
- **Verified.** For every model and layer, each fine cluster's `metacluster_id` matches the tokens file. `count` and `n_tokens`
  equal the token counts in the tokens file, for all 50 fine and all 8 meta-clusters.
- **Source.** The TSP run's own export: `joint-schema-tsp/export/tsp/fine_clusters.parquet` and `clusters.parquet`. The
  merged-theme column is dropped.
