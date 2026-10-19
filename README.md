# Adaptive Temporal Learning for Single-Cell Multi-Omics Alignment and Dynamics

This repository combines the original **ATLAS** preprocessing project with the supplied **GLOAM_v05_T3final2** implementation, renamed under the `atlas` Python package. The supplied *code* is treated as authoritative where the manuscript predates fixes, notably paired and unpaired loss terms. See `docs/IMPLEMENTATION_NOTES.md` for provenance, limitations and evaluation definitions.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
```

## Data preprocessing

See `preprocessing/PREPROCESSING.md` and `docs/data_prepare.md`. Preprocessed PBMC/BMMC ratio-loop datasets are expected under `data/processed/`. These **are not bundled** with the repository. You can symlink a data directory at `ATLAS/data`.

## Train and evaluate T1–T3 (PBMC/BMMC)

```bash
python experiments/run_table2.py --dataset pbmc --pair-ratios 0 40 100 --device cuda:0
python experiments/run_table2.py --dataset bmmc --pair-ratios 0 40 100 --device cuda:0
```

Quick pipeline test with *your preprocessed data*:

```bash
python experiments/run_table2.py --dataset pbmc --pair-ratios 0 --device cpu --quick
```

Outputs go to `results/table2/<dataset>/pair_***/`. Use `--output-root` for another directory.

## Tasks and honest scope

- **T1**: historical CORAL-compatible FOSCTTM/PED on PCA of true RNA and ATAC-predicted RNA; not direct latent-space FOSCTTM.
- **T2**: ATAC-only source input to predict RNA, evaluated by Pearson and RMSE.
- **T3**: temporal teacher, learned TimeHead, POA/POAUC from labeled stages.
- **T4**: The HSPC-specific future-state/fate prediction experiment is **not present in either supplied source archive**, and original processed HSPC data and splits are absent. It is not claimed reproduced. The reusable generic T4 evaluation utility `atlas.future_state` and its tests are included, but its benchmark-specific data adapter/training are still required.

## Key implementation

`src/atlas/model.py`, `losses.py`, `trainer.py`, `temporal.py`, `lag.py`, `unpaired.py`, `data.py`, `metrics.py`.

The current source code implements OT-based unpaired objectives, temporal teaching, cross-omics lag bank and multi-stage optimization. Training requires real processed datasets. Code checks are not a replacement for reproducing published numerical results.
