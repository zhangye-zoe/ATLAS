from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from .utils import paired_ratio_to_split_label, resolve_project_path


def _dense_float32(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _subset_to_features(adata: ad.AnnData, features: Sequence[str]) -> np.ndarray:
    return _dense_float32(adata[:, list(features)].X)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "Standardizer":
        mean = np.asarray(x.mean(axis=0), dtype=np.float32)
        std = np.asarray(x.std(axis=0), dtype=np.float32)
        std = np.where(std < eps, 1.0, std).astype(np.float32)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean[None, :]) / self.std[None, :]).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std[None, :] + self.mean[None, :]).astype(np.float32)


@dataclass
class Table2Data:
    dataset_name: str
    split_dir: Path
    paired_ratio: int
    cell_ids: np.ndarray
    modalities: Dict[str, np.ndarray]
    masks: Dict[str, np.ndarray]
    features: List[str]
    standardizers: Dict[str, Standardizer]
    val_cell_ids: np.ndarray
    val_modalities: Dict[str, np.ndarray]
    val_modalities_raw: Dict[str, np.ndarray]
    labels: Optional[np.ndarray]
    stage_map: Dict[str, int]
    split_info: Dict


class MultiOmicsDataset(Dataset):
    def __init__(self, modalities: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray]):
        self.modalities = dict(modalities)
        self.masks = dict(masks)
        sizes = {x.shape[0] for x in self.modalities.values()}
        if len(sizes) != 1:
            raise ValueError(f"All modality arrays must have the same number of rows: {sizes}")
        self.n = next(iter(sizes))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return (
            idx,
            {m: torch.from_numpy(x[idx]) for m, x in self.modalities.items()},
            {m: torch.tensor(float(mask[idx]), dtype=torch.float32) for m, mask in self.masks.items()},
        )


def collate_multiomics(batch):
    indices = torch.as_tensor([b[0] for b in batch], dtype=torch.long)
    names = sorted(batch[0][1].keys())
    x = {m: torch.stack([b[1][m] for b in batch], dim=0) for m in names}
    masks = {m: torch.stack([b[2][m] for b in batch], dim=0) for m in names}
    return indices, x, masks


def _load_external_labels(cfg: Dict, cell_ids: Sequence[str]) -> Optional[np.ndarray]:
    ext = cfg.get("external_labels") or {}
    csv_path = ext.get("csv")
    if not csv_path:
        return None
    path = resolve_project_path(cfg["project_root"], csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"external label file configured but missing: {path}. "
            "Set external_labels.csv to null if labels are already in AnnData.obs."
        )
    df = pd.read_csv(path)
    ccol = ext.get("cell_column", "cell_id")
    lcol = ext.get("label_column", "cell_type")
    lut = df.drop_duplicates(ccol).set_index(ccol)[lcol]
    return np.asarray([str(lut.get(c, "__missing__")) for c in cell_ids], dtype=object)


def load_table2_split(dataset_cfg: Dict, paired_ratio: int) -> Table2Data:
    """Load the retained ATLAS/ATLAS preprocessing split for PBMC/BMMC.

    Training arrays are formed on the union of training cell IDs. Missing
    modalities are represented by zero rows + explicit observation masks.
    The original target values are never synthesized for missing cells.
    """
    project_root = Path(dataset_cfg["project_root"])
    split_root = resolve_project_path(project_root, dataset_cfg["split_root"])
    split_dir = split_root / paired_ratio_to_split_label(paired_ratio)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    info_path = split_dir / "split_info.json"
    with info_path.open("r", encoding="utf-8") as f:
        split_info = json.load(f)
    if split_info.get("status") != "ok":
        raise RuntimeError(f"Split is not usable: {split_info}")

    rna = ad.read_h5ad(split_dir / dataset_cfg["modalities"]["rna"]["train_file"])
    atac = ad.read_h5ad(split_dir / dataset_cfg["modalities"]["atac"]["train_file"])

    common_features = sorted(set(map(str, rna.var_names)) & set(map(str, atac.var_names)))
    if not common_features:
        raise ValueError("No common gene-level features between train RNA and ATAC activity.")

    rna_ids = list(map(str, rna.obs_names))
    atac_ids = list(map(str, atac.obs_names))
    union_ids = sorted(set(rna_ids) | set(atac_ids))
    id_to_row = {c: i for i, c in enumerate(union_ids)}

    # Fit standardizers only on actually observed rows in each modality.
    rna_obs = _subset_to_features(rna, common_features)
    atac_obs = _subset_to_features(atac, common_features)
    standardizers = {
        "rna": Standardizer.fit(rna_obs),
        "atac": Standardizer.fit(atac_obs),
    }

    modalities = {
        "rna": np.zeros((len(union_ids), len(common_features)), dtype=np.float32),
        "atac": np.zeros((len(union_ids), len(common_features)), dtype=np.float32),
    }
    masks = {
        "rna": np.zeros(len(union_ids), dtype=np.float32),
        "atac": np.zeros(len(union_ids), dtype=np.float32),
    }
    for m, ids, obs in [("rna", rna_ids, rna_obs), ("atac", atac_ids, atac_obs)]:
        z = standardizers[m].transform(obs)
        rows = np.asarray([id_to_row[c] for c in ids], dtype=np.int64)
        modalities[m][rows] = z
        masks[m][rows] = 1.0

    # Use val_atac_activity rather than peak-level ATAC so dimensions match the RNA gene space.
    val_rna = ad.read_h5ad(split_dir / dataset_cfg["target_rna_file"])
    val_atac = ad.read_h5ad(split_dir / dataset_cfg["query_atac_file"])
    common_val_ids = sorted(set(map(str, val_rna.obs_names)) & set(map(str, val_atac.obs_names)))
    val_rna = val_rna[common_val_ids, common_features].copy()
    val_atac = val_atac[common_val_ids, common_features].copy()
    val_raw = {
        "rna": _dense_float32(val_rna.X),
        "atac": _dense_float32(val_atac.X),
    }
    val = {
        "rna": standardizers["rna"].transform(val_raw["rna"]),
        "atac": standardizers["atac"].transform(val_raw["atac"]),
    }

    label_key = dataset_cfg.get("label_key")
    labels = None
    if label_key and label_key in val_rna.obs.columns:
        labels = val_rna.obs[label_key].astype(str).to_numpy()
    elif label_key and label_key in val_atac.obs.columns:
        labels = val_atac.obs[label_key].astype(str).to_numpy()
    else:
        labels = _load_external_labels(dataset_cfg, common_val_ids)

    return Table2Data(
        dataset_name=str(dataset_cfg["name"]),
        split_dir=split_dir,
        paired_ratio=int(paired_ratio),
        cell_ids=np.asarray(union_ids, dtype=object),
        modalities=modalities,
        masks=masks,
        features=common_features,
        standardizers=standardizers,
        val_cell_ids=np.asarray(common_val_ids, dtype=object),
        val_modalities=val,
        val_modalities_raw=val_raw,
        labels=labels,
        stage_map={str(k): int(v) for k, v in (dataset_cfg.get("stage_map") or {}).items()},
        split_info=split_info,
    )
