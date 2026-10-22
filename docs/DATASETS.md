# ATLAS Data Preprocessing

This document describes how to obtain the raw training datasets, prepare the external annotation resources, and reproduce the preprocessing pipeline used by ATLAS. All paths below are relative to the ATLAS project root unless otherwise specified.

## 1. Training Data: Download and Preprocessing

### 1.1 PBMC RNA–ATAC

**Dataset:** [10k Human PBMCs Multiome (10x Genomics)](https://www.10xgenomics.com/datasets/10-k-human-pbm-cs-multiome-v-1-0-chromium-x-1-standard-2-0-0)

**Download:**

```text
pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
```

**Save to:** `data/raw/PBMC/`

### 1.2 BMMC RNA–ATAC / RNA–Protein

**Dataset:** [GSE194122 (GEO)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)

**Download:**

```text
GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
```

**Decompress:**

```bash
gunzip GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
gunzip GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
```

**Save to:** `data/raw/BMMC/`

For the Open Problems NeurIPS 2021 BMMC resource, please cite [1].

### 1.3 Raw Training Data Layout

```text
data/raw/
├── PBMC/
│   └── pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
└── BMMC/
    ├── GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad
    └── GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad
```

### 1.4 Training Data Preprocessing

The preprocessing scripts and configurations are organized as follows:

```text
preprocessing/
├── configs/
│   ├── bmmc.yaml
│   └── pbmc.yaml
├── logs/
├── scripts/
│   ├── common.py
│   ├── preprocess_bmmc_rna_atac.py
│   ├── preprocess_bmmc_rna_protein.py
│   └── preprocess_pbmc_rna_atac.py
├── run_bmmc.sh
└── run_pbmc.sh
```

Run from the ATLAS root:

```bash
cd /data5/zhangye/ATLAS
```

**PBMC RNA–ATAC:**

```bash
bash preprocessing/run_pbmc.sh
```

**BMMC RNA–ATAC and RNA–Protein:**

```bash
bash preprocessing/run_bmmc.sh
```

Alternatively, run the scripts individually:

```bash
python preprocessing/scripts/preprocess_pbmc_rna_atac.py \
  --config preprocessing/configs/pbmc.yaml

python preprocessing/scripts/preprocess_bmmc_rna_atac.py \
  --config preprocessing/configs/bmmc.yaml

python preprocessing/scripts/preprocess_bmmc_rna_protein.py \
  --config preprocessing/configs/bmmc.yaml
```

## 2. External Annotations: Download and Preparation

ATLAS uses two complementary external annotation resources. These are **reference annotations**, not additional training samples.

### 2.1 GENCODE v38: Gene Coordinates for RNA–ATAC

**Source:** [GENCODE Human Release 38 (GRCh38)](https://www.gencodegenes.org/human/release_38.html)

**File:** `gencode.v38.primary_assembly.annotation.gtf.gz`

**Direct download:** [GENCODE v38 primary-assembly GTF](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.primary_assembly.annotation.gtf.gz)

**Purpose:** Provides gene coordinates and transcript annotations on the GRCh38 primary assembly. It supports the genomic annotation needed to connect ATAC peaks with nearby genes (e.g., for gene-activity construction) in the PBMC and BMMC RNA–ATAC preprocessing pipelines. Ensure that ATAC peak coordinates use a compatible genome assembly; genomic proximity alone does not establish regulation.

Download and decompress from the ATLAS project root:

```bash
mkdir -p data/external/annotations
wget -c -P data/external/annotations \
  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.primary_assembly.annotation.gtf.gz

gunzip data/external/annotations/gencode.v38.primary_assembly.annotation.gtf.gz
```

**Output:**

```text
data/external/annotations/gencode.v38.primary_assembly.annotation.gtf
```

Please cite the GENCODE resource [2].

### 2.2 HGNC: CD Marker–Gene Mapping for RNA–Protein

**Source:** [HGNC download portal](https://hgnc.genenames.org/download/)

**File:** `hgnc_complete_set.txt`

**Direct download:** [HGNC complete approved-gene TSV](https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt)

**Purpose:** Provides standardized gene symbols and cross-database identifiers, including HGNC, Ensembl, and CD marker annotations. ATLAS uses a curated subset to link CD-labeled protein measurements with corresponding RNA gene identifiers in BMMC RNA–Protein preprocessing. CD-marker correspondence is an identity annotation; it does not imply that RNA expression and protein abundance are equivalent. This resource is optional when RNA–Protein preprocessing does not require CD marker conversion.

Download from the ATLAS project root:

```bash
mkdir -p data/external/annotations
wget -c -P data/external/annotations \
  https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
```

Generate the derived annotation files using the included script:

```bash
python data/external/script/prepare_hgnc_cd_mapping.py
```

The script reads `data/external/annotations/hgnc_complete_set.txt`, selects HGNC Gene Group **471 (CD molecules)**, and writes:

- `group-471.csv`: CD-molecule gene-group annotations, including standardized gene identifiers and aliases.
- `hgnc_cd_gene_mapping.csv`: CD marker–to–gene mappings using entries with an explicit HGNC `cd` field.

The mapping is intentionally restricted to the CD molecule subset and should **not** be used as an exhaustive mapping for all measured proteins. The HGNC complete set is updated periodically; for exact reproducibility, retain the downloaded source file or document its retrieval date/checksum.

Please cite the HGNC resource [3].

### 2.3 Prepared Files on Hugging Face

For convenient reproduction, we also provide prepared annotation files on Hugging Face, so users can avoid repeating the external-data preparation steps.

**Hugging Face repository:** **TODO — insert the public repository URL here.**

When using these files, please acknowledge and cite the original data/resource publications **[1–3]** as applicable. Redistribution and preprocessing do not replace the original authors' attribution or licensing terms.

### 2.4 External Annotation Layout After Preparation

```text
data/external/
├── annotations/
│   ├── gencode.v38.primary_assembly.annotation.gtf
│   ├── hgnc_complete_set.txt
│   ├── group-471.csv
│   └── hgnc_cd_gene_mapping.csv
└── script/
    └── prepare_hgnc_cd_mapping.py
```

## 3. Complete Project Data Layout

After downloading the training datasets, preparing the external annotations, and running the training-data preprocessing scripts, the relevant project structure is:

```text
ATLAS/
├── data/
│   ├── raw/
│   │   ├── PBMC/
│   │   │   └── pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
│   │   └── BMMC/
│   │       ├── GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad
│   │       └── GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad
│   ├── external/
│   │   ├── annotations/
│   │   │   ├── gencode.v38.primary_assembly.annotation.gtf
│   │   │   ├── hgnc_complete_set.txt
│   │   │   ├── group-471.csv
│   │   │   └── hgnc_cd_gene_mapping.csv
│   │   └── script/
│   │       └── prepare_hgnc_cd_mapping.py
│   ├── interim/
│   │   ├── PBMC/
│   │   │   └── RNA_ATAC/
│   │   │       ├── RNA_counts_qc.h5ad
│   │   │       ├── ATAC_counts_qc.h5ad
│   │   │       ├── ATAC_gas.h5ad
│   │   │       └── feature_aligned.h5ad
│   │   └── BMMC/
│   │       ├── RNA_ATAC/
│   │       │   ├── RNA_counts_qc.h5ad
│   │       │   ├── ATAC_counts_qc.h5ad
│   │       │   ├── ATAC_gas.h5ad
│   │       │   └── feature_aligned_rna_atac.h5ad
│   │       └── RNA_PROTEIN/
│   │           ├── RNA_counts_qc.h5ad
│   │           ├── protein_counts_qc.h5ad
│   │           └── feature_aligned_rna_protein.h5ad
│   └── processed/
│       ├── PBMC/
│       │   └── RNA_ATAC/
│       │       └── results_ratio_loop/
│       └── BMMC/
│           ├── RNA_ATAC/
│           │   └── results_ratio_loop_rna_atac/
│           └── RNA_PROTEIN/
│               └── results_ratio_loop_rna_protein/
├── preprocessing/
│   ├── configs/
│   │   ├── bmmc.yaml
│   │   └── pbmc.yaml
│   ├── logs/
│   ├── scripts/
│   │   ├── common.py
│   │   ├── preprocess_bmmc_rna_atac.py
│   │   ├── preprocess_bmmc_rna_protein.py
│   │   └── preprocess_pbmc_rna_atac.py
│   ├── run_bmmc.sh
│   └── run_pbmc.sh
└── README.md
```

Each `results_ratio_loop*` directory contains the following outputs:

```text
single_000/
single_020/
single_040/
single_060/
single_080/
single_100/
summary_all_ratios.csv
```

## References

[1] Luecken, M. D., Burkhardt, D. B., Cannoodt, R., et al. **A sandbox for prediction and integration of DNA, RNA, and proteins in single cells.** *Proceedings of the NeurIPS Track on Datasets and Benchmarks*, 2021. [Publication](https://biblio.ugent.be/publication/8740779).

[2] Frankish, A., Diekhans, M., Jungreis, I., et al. **GENCODE 2021.** *Nucleic Acids Research*, 49(D1), D916–D923, 2021. https://doi.org/10.1093/nar/gkaa1087.

[3] Seal, R. L., Braschi, B., Gray, K., McClay, J., Tweedie, S., & Bruford, E. A. **Genenames.org: the HGNC and PGNC resources in 2026.** *Nucleic Acids Research*, 54(D1), D1098–D1107, 2026. https://doi.org/10.1093/nar/gkaf1229.
