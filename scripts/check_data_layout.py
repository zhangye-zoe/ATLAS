#!/usr/bin/env python3
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
for ds in ("pbmc", "bmmc"):
    cfg = yaml.safe_load((ROOT / f"configs/datasets/{ds}.yaml").read_text())
    base = Path(cfg["project_root"]) / cfg["split_root"]
    print(f"[{cfg['name']}] {base}")
    for paired in (0, 40, 100):
        label = f"single_{100-paired:03d}"
        d = base / label
        needed = ["split_info.json", "train_rna_ref.h5ad", "train_atac_activity.h5ad", "val_atac_activity.h5ad", "val_true_rna.h5ad"]
        missing = [x for x in needed if not (d / x).exists()]
        print(f"  paired={paired:3d}% -> {label}: {'OK' if not missing else 'MISSING ' + ','.join(missing)}")
