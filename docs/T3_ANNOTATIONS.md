# T3 Reference Labels: PBMC/BMMC

ATLAS trains a time prediction head during Phase III even when validation cell-type labels are absent. **POA and POAUC are post-hoc metrics** requiring independently assigned reference stages. Missing T3 metrics do not imply that Phase III was skipped.

## 1. What is missing?

For PBMC, the dataset config (configs/datasets/pbmc.yaml) expects one of:

- a **cell_type** column in the validation AnnData obs metadata; or
- an external CSV mapping **exact validation cell IDs** to Azimuth-style **predicted.celltype.l2** annotations.

The original 10x Multiome count matrix does **not** automatically contain Azimuth cell-type predictions. The existing stage_map only maps annotation **names** to ordinal stage values; it cannot provide individual labels.

Check available columns on the training server:

~~~bash
python - <<'PY'
import anndata as ad
p = "data/processed/PBMC/RNA_ATAC/results_ratio_loop/single_100/val_true_rna.h5ad"
a = ad.read_h5ad(p, backed="r")
print("Columns:", list(a.obs.columns))
print("Example cell IDs:", list(map(str, a.obs_names[:5])))
a.file.close()
PY
~~~

## 2. Obtain actual annotations

Use **one independently annotated PBMC file**, either CSV or H5AD, with cell barcodes matching the original 10x sample. A common source is the [Azimuth human PBMC reference](https://azimuth.hubmapconsortium.org/references/). If you already have a Seurat/Azimuth CSV with predicted.celltype.l2, use it directly.

Required CSV format:

~~~csv
cell_id,predicted.celltype.l2
EXAMPLE_CELL_A-1,B naive
EXAMPLE_CELL_B-1,CD14 Mono
~~~

**The rows above are artificial format examples, not validated labels. Do not use them as annotations.**

For example, after labeling the **original RNA count matrix** with Azimuth, export its metadata in R:

~~~r
meta <- query[[]]  # query is your correctly annotated Seurat object
ann <- data.frame(
  cell_id = rownames(meta),
  "predicted.celltype.l2" = meta[["predicted.celltype.l2"]],
  check.names = FALSE
)
write.csv(ann,
          "data/external/annotations/pbmc_azimuth_predictions.csv",
          row.names = FALSE)
~~~

Check that barcodes correspond exactly to the ATLAS processed data. Avoid guessing cell types or matching cells by row order. Record the Azimuth reference version and annotation procedure.

## 3. Compute T3 without retraining

From the ATLAS repository root **on the training server**:

~~~bash
# Inspect coverage / preview T3 without modifying files
python scripts/restore_t3_metrics.py \
  --dataset pbmc \
  --annotations data/external/annotations/pbmc_azimuth_predictions.csv \
  --dry-run

# Save validated label mapping, T3 metrics, and summary
python scripts/restore_t3_metrics.py \
  --dataset pbmc \
  --annotations data/external/annotations/pbmc_azimuth_predictions.csv
~~~

If the validation H5AD files already contain a cell_type or predicted.celltype.l2 column, omit --annotations; the script collects existing obs labels.

The script checks that at least 90% of validation cells have labels recognized by stage_map and at least two stages are present. It refuses to calculate scores for incompatible labels and prints unmapped annotation names. Lower --min-coverage only if partial coverage is explicitly justified.

The results folders must contain **predictions.npz**, written during the previous training run, with cell_ids and time arrays. **No retraining or model checkpoint is needed.** If you downloaded only CSV and JSON files to your laptop, run the script **on the server** where NPZ files remain.

Outputs:

~~~text
data/external/annotations/pbmc_t3_labels.csv
results/table2/PBMC/t3_summary.csv
results/table2/PBMC/pair_000/T3_stage_reference.csv
results/table2/PBMC/pair_000/metrics.json
results/table2/PBMC/pair_000/metrics.before_t3.json
~~~

The exported label CSV uses the annotation column name expected by the PBMC dataset config. Annotation files in data/external/annotations are excluded from Git commits by .gitignore.

For **future** training/evaluation runs, pass the verified mapping:

~~~bash
python experiments/run_table2.py --dataset pbmc \
  --pair-ratios 0 40 100 \
  --label-csv data/external/annotations/pbmc_t3_labels.csv \
  --device cuda:0
~~~

## 4. Interpretation and leakage safeguards

- The ordinal stage_map is a **cell-type-based temporal proxy**, not measured continuous developmental time.
- Azimuth predicts **reference cell types**, not experimental lineage ground truth.
- Reference labels must remain outside training, temporal teacher construction, and lag-bank construction.
- With orient_time_for_metric enabled, direction is selected from validation stage labels to maximize POA. **Report reference-oriented evaluation explicitly.**
- T3 defaults to paired_fused, which can use both validation modalities during latent-time prediction; T1 and T2 predict from ATAC only.
- Do not copy another PBMC cohort's annotation CSV without verifying actual cell barcode identity.
