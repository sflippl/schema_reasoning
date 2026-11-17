# Fine clusters, meta-clusters and labels for D1, D2 and TSP

These are for Phi-4-mini-reasoning (`phi4mr`) and Qwen3-4B (`qwen34b`), at layers 8 and 16. All IDs come from the per-dataset
clustering, **before any merge**. No merged-theme columns are included.

| Folder | Data | Files |
|---|---|---|
| `d1/` | D1, 300 items × 2 models: the same responses as Sam's `binding_dynamics-phi4miniR_Qwen34B-v2` files | tokens file in Sam's layout, plus meta-cluster and fine-cluster label tables |
| `d2/` | D2, 297 items × 2 models | responses file in Sam's format, tokens file and label tables |
| `tsp/` | TSP, 100 tasks × 2 models | tokens file and label tables; `item_id`/`uid` match the collaborator's TSP `responses.parquet` |

Each folder has a README with its details, checks and caveats. The key points:

- **What the numbers are.** Meta-clusters are numbered 0–7 and fine clusters 0–49. The numbers are names within one dataset,
  model and layer. They have no meaning across datasets, models or older runs. Look up their meaning in that folder's label
  tables.
- **How the clusters were fit.** Each dataset was clustered separately per model: 50 fine clusters, then 8 meta-clusters, seed 0, no PCA. The
  spot-error (core_spot) data was not used for any of these clusters or labels.
- **Assignments only.** No cluster centres or normalization parameters were saved, so new tokens can't be assigned to these
  clusters.
- **TSP token IDs were rebuilt.** They were reconstructed from the collaborator's response text; see `tsp/README.md`.
