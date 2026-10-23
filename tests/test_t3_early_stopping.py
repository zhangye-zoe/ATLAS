"""Regression tests for leakage-aware T3 checkpoint selection protocol."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.data import Table2Data
from atlas.early_stopping import split_monitor_report, subset_validation
from atlas.trainer import infer_validation_t3


def toy_table(n=120):
    labels = np.array(["HSPC"] * 30 + ["B naive"] * 30 +
                      ["CD14 Mono"] * 30 + ["B memory"] * 30, dtype=object)
    assert len(labels) == n
    x = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    return Table2Data(
        dataset_name="PBMC",
        split_dir=Path("/dummy/single_100"),
        paired_ratio=0,
        cell_ids=np.array([f"training-{i}" for i in range(n)], dtype=object),
        modalities={"atac": x, "rna": x},
        masks={"atac": np.ones(n), "rna": np.ones(n)},
        features=["g0", "g1", "g2", "g3"],
        standardizers={},
        val_cell_ids=np.array([f"val-{i}" for i in range(n)], dtype=object),
        val_modalities={"atac": x, "rna": x},
        val_modalities_raw={"atac": x, "rna": x},
        labels=labels,
        stage_map={"HSPC": 0, "B naive": 1, "CD14 Mono": 2, "B memory": 3},
        split_info={},
    )


def test_fixed_partition_is_disjoint_stage_stratified_and_reproducible():
    data = toy_table()
    monitor, report, info = split_monitor_report(
        data, monitor_fraction=0.30, seed=1234,
    )
    monitor2, report2, _ = split_monitor_report(
        data, monitor_fraction=0.30, seed=1234,
    )
    assert info["monitor_cells"] == 36
    assert info["report_cells"] == 84
    assert info["checkpoint_selection_uses_reference_labels"] is True
    assert info["training_losses_use_reference_labels"] is False
    assert set(monitor.val_cell_ids).isdisjoint(report.val_cell_ids)
    assert set(monitor.labels) == set(report.labels) == set(data.labels)
    np.testing.assert_array_equal(monitor.val_cell_ids, monitor2.val_cell_ids)
    np.testing.assert_array_equal(report.val_cell_ids, report2.val_cell_ids)
    assert data.val_modalities["rna"].shape[0] == 120
    assert monitor.modalities["rna"].shape[0] == 120
    assert report.modalities["rna"].shape[0] == 120
    np.testing.assert_array_equal(
        monitor.val_modalities["atac"],
        data.val_modalities["atac"][
            np.array([int(str(v).split("-")[1]) for v in monitor.val_cell_ids])
        ],
    )


def test_invalid_partition_fails_instead_of_leaking():
    data = toy_table()
    data.labels = None
    with pytest.raises(ValueError, match="independent reference"):
        split_monitor_report(data)
    data = toy_table()
    data.labels[:] = "unknown"
    with pytest.raises(ValueError, match="At least 20"):
        split_monitor_report(data)


def test_one_cell_stage_fails():
    data = toy_table()
    data.labels[0] = "Unique"
    data.stage_map["Unique"] = 4
    with pytest.raises(ValueError, match="only 1"):
        split_monitor_report(data)


def test_subset_validation_keeps_training_arrays_intact():
    data = toy_table()
    sub = subset_validation(data, np.array([1, 7, 9]))
    assert sub.val_cell_ids.tolist() == ["val-1", "val-7", "val-9"]
    assert sub.modalities["rna"].shape[0] == 120
    assert sub.val_modalities["rna"].shape[0] == 3
    with pytest.raises(ValueError, match="Invalid validation subset"):
        subset_validation(data, np.array([1, 1]))


class ToyTimeHeadModel(torch.nn.Module):
    def forward(self, x, masks, sample=False):
        return {"time": torch.sigmoid(x["atac"][:, 0] / 50.0)}


def test_monitor_and_final_use_same_timehead_inference():
    data = toy_table()
    monitor, report, _ = split_monitor_report(data, seed=22)
    model = ToyTimeHeadModel()
    cfg = {"evaluation": {"t3_input_mode": "paired_fused", "t3_time_mode": "head"}}
    train_t, _, est1, _, _, _ = infer_validation_t3(
        model, monitor, cfg, torch.device("cpu")
    )
    test_t, _, est2, _, _, _ = infer_validation_t3(
        model, report, cfg, torch.device("cpu")
    )
    assert est1 == est2 == "head"
    assert len(train_t) == len(monitor.val_cell_ids)
    assert len(test_t) == len(report.val_cell_ids)
    assert np.isfinite(train_t).all() and np.isfinite(test_t).all()
