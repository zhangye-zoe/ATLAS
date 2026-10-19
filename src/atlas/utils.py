from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path


def paired_ratio_to_split_label(paired_ratio: int | float) -> str:
    """Map a *paired* percentage to the preprocessing folder name.

    The retained preprocessing code stores ``single_frac`` (fraction made
    single-modality), not paired fraction. Therefore:
      paired 100% -> single_000
      paired  40% -> single_060
      paired   0% -> single_100
    """
    r = float(paired_ratio)
    if r < 0 or r > 100:
        raise ValueError(f"paired_ratio must be within [0, 100], got {r}")
    single_pct = int(round(100.0 - r))
    return f"single_{single_pct:03d}"


def get_device(requested: str = "cuda") -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")
