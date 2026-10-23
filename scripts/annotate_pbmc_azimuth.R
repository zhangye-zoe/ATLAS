#!/usr/bin/env Rscript
# Generate cell-wise PBMC reference annotations for ATLAS T3 using Azimuth.
# This script operates on original 10x RNA counts, not ATLAS validation outcomes.
#
# Usage:
#   Rscript scripts/annotate_pbmc_azimuth.R \
#     data/raw/PBMC/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 \
#     data/external/annotations/pbmc_azimuth_predictions.csv
#
# Requires R packages: Seurat and Azimuth, plus access to the PBMC reference.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript annotate_pbmc_azimuth.R INPUT_10X_H5 OUTPUT_CSV")
}
if (!requireNamespace("Seurat", quietly = TRUE) ||
    !requireNamespace("Azimuth", quietly = TRUE)) {
  stop("Install the Seurat and Azimuth R packages before running this script.")
}

input_path <- args[[1]]
output_path <- args[[2]]
if (!file.exists(input_path)) stop("Input missing: ", input_path)

counts <- Seurat::Read10X_h5(input_path)
if (is.list(counts)) {
  if (!"Gene Expression" %in% names(counts)) {
    stop("Expected 'Gene Expression' in 10x Multiome H5; got: ",
         paste(names(counts), collapse = ", "))
  }
  counts <- counts[["Gene Expression"]]
}
query <- Seurat::CreateSeuratObject(counts = counts, assay = "RNA")
query <- Azimuth::RunAzimuth(query = query, reference = "pbmcref")
meta <- query[[]]
if (!"predicted.celltype.l2" %in% colnames(meta)) {
  stop("Azimuth returned no 'predicted.celltype.l2' metadata column")
}

annotation <- data.frame(
  cell_id = rownames(meta),
  "predicted.celltype.l2" = as.character(meta[["predicted.celltype.l2"]]),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(annotation, file = output_path, row.names = FALSE, na = "")
message("Saved ", nrow(annotation), " reference labels to ", output_path)
message("Keep the Azimuth reference version and R package versions for reproducibility.")
