#!/usr/bin/env python3
"""Create HGNC CD-molecule family table and official CD-to-gene mapping.

Usage (from the directory containing hgnc_complete_set.txt):
    python prepare_hgnc_cd_mapping.py

Outputs:
    group-471.csv               : all current CD-molecules gene-family members
    hgnc_cd_gene_mapping.csv    : members with a non-empty HGNC 'cd' field
"""

from pathlib import Path
import pandas as pd

# Place this script in the script/ directory
# hgnc_complete_set.txt in the sibling annotations/ directory
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT = BASE_DIR / "annotations" / "hgnc_complete_set.txt"
OUTPUT_GROUP = BASE_DIR / "annotations" / "group-471.csv"
OUTPUT_MAPPING = BASE_DIR / "annotations" / "hgnc_cd_gene_mapping.csv"
GROUP_ID = "471"  # HGNC CD molecules gene-family identifier

# Read every value as a string (e.g. HGNC:5, ENSG..., CDw338).
# Empty cells remain empty strings rather than NaN.
df = pd.read_csv(INPUT, sep="\t", dtype=str, keep_default_na=False)
print("Original HGNC table:", df.shape)

required = [
    "hgnc_id", "symbol", "name", "status", "gene_group_id",
    "gene_group", "cd", "ensembl_gene_id",
]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep active records only; group identifiers are |-separated (e.g. 454|471|811).
is_approved = df["status"].eq("Approved")
in_cd_group = df["gene_group_id"].apply(
    lambda x: GROUP_ID in x.split("|")
)
df_cd = df.loc[is_approved & in_cd_group].copy()

# Check that the chosen family ID corresponds to CD molecules.
is_named_cd = df_cd["gene_group"].apply(
    lambda x: "CD molecules" in x.split("|")
)
if not is_named_cd.all():
    raise ValueError("Some group-471 rows are not labeled CD molecules; check input.")

# Output 1: keep legacy notebook-friendly column names, plus useful new IDs.
rename_map = {
    "hgnc_id": "HGNC ID (gene)",
    "symbol": "Approved symbol",
    "name": "Approved name",
    "status": "Status",
    "prev_symbol": "Previous symbols",
    "alias_symbol": "Aliases",
    "location": "Chromosome",
    "ena": "Accession numbers",
    "refseq_accession": "RefSeq IDs",
    "gene_group_id": "Gene group ID",
    "gene_group": "Gene group name",
    "cd": "CD",
    "ensembl_gene_id": "Ensembl Gene ID",
    "entrez_id": "Entrez Gene ID",
    "uniprot_ids": "UniProt IDs",
}
output_cols = [rename_map[c] for c in rename_map if c in df_cd.columns]
full_group = df_cd.rename(columns=rename_map)[output_cols]
full_group.to_csv(OUTPUT_GROUP, index=False)

# Output 2: authoritative HGNC 'cd' entries only.
# Do NOT automatically fill missing 'cd' from alias_symbol: aliases can be historical.
mapping = df_cd.loc[df_cd["cd"].ne(""), [
    "hgnc_id", "symbol", "ensembl_gene_id", "cd"
]].copy()
mapping.to_csv(OUTPUT_MAPPING, index=False)

print("CD molecules family:", len(df_cd))
print("With HGNC official CD value:", len(mapping))
print("Without HGNC official CD value:", (df_cd["cd"] == "").sum())
print("Mappings missing Ensembl gene ID:", (mapping["ensembl_gene_id"] == "").sum())
print("Duplicated CD designations:", mapping["cd"].duplicated(keep=False).sum(), "rows")
print(f"Saved: {OUTPUT_GROUP}")
print(f"Saved: {OUTPUT_MAPPING}")
print("\nExample mappings:\n", mapping.head(8).to_string(index=False))
