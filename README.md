# Adaptive Temporal Learning for Single-Cell Multi-Omics Alignment and Dynamics

[![GitHub Stars](https://img.shields.io/github/stars/zhangye-zoe/ATLAS?style=flat&logo=github)](https://github.com/zhangye-zoe/ATLAS/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/zhangye-zoe/ATLAS?style=flat&logo=github)](https://github.com/zhangye-zoe/ATLAS/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/zhangye-zoe/FCIS/blob/main/LICENSE)

This is the official code repository for our paper accepted at NeurIPS 2026:
**ATLAS: Adaptive Temporal Learning for Single-Cell Multi-Omics Alignment and Dynamics**

- **Paper Link:** [Read the Paper](https://openreview.net/pdf?id=TEIXzoW2jj)
- **Publication:** Advances in Neural Information Processing Systems (NeurIPS) 2026
- **Authors:** Ye Zhang, Zijie Fang, Zhixiang Lin


## Progress
> ℹ️ Some components are still being refactored. The codebase will continue to be updated.

**2026-10-22**
- Added external annotation resources (GENCODE v38 and HGNC) and CD-to-gene mapping scripts.
- Reorganized the data directory structure and updated preprocessing documentation.
- Added `docs/RESULTS.md` to summarize experimental results, including benchmark comparisons, ablation studies, and reproducibility tracking.
- Updated the README with links to experimental results and evaluation resources.

**2026-10-19**  
- Added core ATLAS modules for multi-omics alignment, temporal inference, and lag-aware modeling.


**2026-09-11**
- PBMC RNA–ATAC
- BMMC RNA–ATAC
- BMMC RNA–Protein
- Detailed preprocessing instructions are available in [`preprocessing/PREPROCESSING.md`](preprocessing/PREPROCESSING.md).

## 🌟 Highlights

- Multi-omics representation learning and alignment for paired, partially paired, and unpaired data.
- Reliability-guided temporal distillation for cell-state ordering.
- Adaptive cross-omics lag modeling with paired and unpaired training objectives.
- Support for cross-modal alignment (T1), cross-omics prediction (T2), and trajectory inference (T3) in the current code.

## 🚀 Installation

```bash
git clone https://github.com/zhangye-zoe/ATLAS.git
cd ATLAS
pip install -r requirements.txt
pip install -e .
```

## 📊 Dataset Preparation

The preprocessing pipelines cover PBMC RNA–ATAC, BMMC RNA–ATAC, and BMMC RNA–protein datasets.

See [Data Preprocessing Guide](./preprocessing/PREPROCESSING.md) for details. Preprocessed training data are not included in the source archive.

## 🧪 Training and Evaluation

Run T1–T3 on PBMC or BMMC after preparing the data:

```bash
python experiments/run_table2.py --dataset pbmc --pair-ratios 0 40 100 --device cuda:0
python experiments/run_table2.py --dataset bmmc --pair-ratios 0 40 100 --device cuda:0
```

For a quick pipeline check:

```bash
python experiments/run_table2.py --dataset pbmc --pair-ratios 0 --device cpu --quick
```

**Missing PBMC/BMMC T3 metrics?** See [T3 reference annotations and post-hoc evaluation](./docs/T3_ANNOTATIONS.md) to obtain validated cell-type labels and rescore saved latent times without retraining.

The HSPC future-state prediction (T4) benchmark is not yet included as an end-to-end experiment in this source package.

## 📈 Experimental Results

See [Experimental Results](./docs/RESULTS.md) for the paper-reported reference tables, metric conventions, and reproduction tracking. Generated experiment artifacts are stored locally under `results/`.

## 📖 Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{zhang2026atlas,
  title={ATLAS: Adaptive Temporal Learning for Single-Cell Multi-Omics Alignment and Dynamics},
  author={Zhang, Ye and Fang, Zijie and Lin, Zhixiang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
