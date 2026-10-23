"""End-to-end regression test for post-hoc T3 scoring; no model training needed."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "restore_t3_metrics.py"


def test_restore_pbmc_t3_from_validated_csv(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "table2" / "PBMC"
    run = run_root / "pair_000"
    run.mkdir(parents=True)

    ids = np.asarray([f"test-cell-{i}" for i in range(24)], dtype=str)
    labels = ["HSPC"] * 8 + ["B naive"] * 8 + ["B memory"] * 8
    # Inverted pseudotime; reference-oriented T3 should flip the direction.
    time = np.asarray([0.9] * 8 + [0.5] * 8 + [0.1] * 8, dtype=float)
    np.savez_compressed(run / "predictions.npz", cell_ids=ids, time=time)
    (run / "metrics.json").write_text(
        json.dumps({"T3": {"POA": None}, "protocol": {}})
    )

    annotations = tmp_path / "azimuth.csv"
    pd.DataFrame({
        "cell_id": ids,
        "predicted.celltype.l2": labels,
    }).to_csv(annotations, index=False)
    output_csv = tmp_path / "pbmc_t3_labels.csv"

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--dataset", "pbmc",
            "--annotations", str(annotations),
            "--results-root", str(run_root),
            "--output-csv", str(output_csv),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "POA=100.00%" in result.stdout
    m = json.loads((run / "metrics.json").read_text())
    assert m["T3"]["POA"] == 1.0
    assert m["T3"]["POAUC"] == 1.0
    assert m["T3"]["n_eval"] == 24
    assert (run / "metrics.before_t3.json").exists()
    assert (run / "T3_stage_reference.csv").exists()
    assert (run_root / "t3_summary.csv").exists()
    assert output_csv.exists()
    assert "predicted.celltype.l2" in pd.read_csv(output_csv).columns
