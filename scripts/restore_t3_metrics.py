#!/usr/bin/env python3
"""Prepare verified T3 annotations and rescore existing ATLAS predictions.

This script NEVER infers cell labels from ATLAS predictions or stage scores.
It requires reference annotations from an external CSV/annotated AnnData, or
existing labels in validation AnnData.obs. No model retraining is required.

Example:
  python scripts/restore_t3_metrics.py --dataset pbmc \
    --annotations path/to/azimuth_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atlas.metrics import evaluate_t3  # noqa: E402

MISSING = {"", "nan", "na", "none", "null", "__missing__"}


def read_annotation(path: Path, cell_column: str, label_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if cell_column not in df.columns:
            raise ValueError(
                f"{path}: missing cell-ID column '{cell_column}'. "
                f"Available columns: {list(df.columns)}"
            )
        if label_column not in df.columns:
            raise ValueError(
                f"{path}: missing label column '{label_column}'. "
                f"Available columns: {list(df.columns)}"
            )
        result = df[[cell_column, label_column]].rename(
            columns={cell_column: "cell_id", label_column: "label"}
        )
    elif path.suffix.lower() == ".h5ad":
        import anndata as ad

        x = ad.read_h5ad(path, backed="r")
        try:
            if label_column not in x.obs.columns:
                raise ValueError(
                    f"{path}: obs['{label_column}'] not found. "
                    f"Available columns: {list(x.obs.columns)}"
                )
            result = pd.DataFrame({
                "cell_id": np.asarray(x.obs_names, dtype=str),
                "label": x.obs[label_column].astype(str).to_numpy(),
            })
        finally:
            if x.isbacked:
                x.file.close()
    else:
        raise ValueError("Annotation source must be .csv or .h5ad")

    result = result.copy()
    result["cell_id"] = result["cell_id"].astype(str).str.strip()
    result["label"] = result["label"].astype(str).str.strip()
    result = result.loc[
        result["cell_id"].ne("")
        & ~result["label"].str.lower().isin(MISSING)
    ]
    conflicts = result.groupby("cell_id")["label"].nunique()
    if (conflicts > 1).any():
        examples = conflicts[conflicts > 1].index[:5].tolist()
        raise ValueError(f"Conflicting labels for identical cell IDs: {examples}")
    return result.drop_duplicates("cell_id").reset_index(drop=True)


def annotations_from_validation_obs(split_root: Path, label_column: str) -> pd.DataFrame:
    import anndata as ad

    pieces = []
    for directory in sorted(split_root.glob("single_*")):
        for name in ("val_true_rna.h5ad", "val_atac_activity.h5ad"):
            path = directory / name
            if not path.is_file():
                continue
            x = ad.read_h5ad(path, backed="r")
            try:
                if label_column in x.obs.columns:
                    pieces.append(pd.DataFrame({
                        "cell_id": np.asarray(x.obs_names, dtype=str),
                        "label": x.obs[label_column].astype(str).to_numpy(),
                    }))
            finally:
                if x.isbacked:
                    x.file.close()
    if not pieces:
        raise RuntimeError(
            f"No obs['{label_column}'] found under {split_root}.\n"
            "The 10x Multiome count matrix does not provide Azimuth cell types. "
            "Supply an independently annotated CSV or .h5ad with --annotations. "
            "Do not invent labels from predicted pseudotime."
        )
    combined = pd.concat(pieces, ignore_index=True)
    combined = combined[
        ~combined["label"].astype(str).str.strip().str.lower().isin(MISSING)
    ]
    conflicts = combined.groupby("cell_id")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting labels among validation .obs files")
    return combined.drop_duplicates("cell_id").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["pbmc", "bmmc"], default="pbmc")
    p.add_argument("--annotations", type=Path, help="Reference CSV or annotated H5AD")
    p.add_argument("--cell-column", default="cell_id")
    p.add_argument("--label-column", help="Label column; defaults to dataset's external_labels.label_column")
    p.add_argument("--results-root", type=Path, help="Default: results/table2/<dataset>")
    p.add_argument("--output-csv", type=Path, help="Default: data/external/annotations/<dataset>_t3_labels.csv")
    p.add_argument("--min-coverage", type=float, default=0.90,
                   help="Minimum fraction of validation cells with known stage labels")
    p.add_argument("--dry-run", action="store_true", help="Only inspect; write no files")
    args = p.parse_args()

    if not 0 < args.min_coverage <= 1:
        p.error("--min-coverage must be in (0,1]")
    cfg = yaml.safe_load((ROOT / "configs/datasets" / f"{args.dataset}.yaml").read_text())
    model_cfg = yaml.safe_load((ROOT / "configs/model_table2.yaml").read_text())
    stage_map = {str(k): int(v) for k, v in cfg["stage_map"].items()}
    label_column = args.label_column or cfg["external_labels"]["label_column"]
    split_root = ROOT / cfg["split_root"]
    results_root = args.results_root or ROOT / "results" / "table2" / cfg["name"]
    output_csv = args.output_csv or ROOT / "data" / "external" / "annotations" / f"{args.dataset}_t3_labels.csv"

    if args.annotations:
        labels = read_annotation(args.annotations, args.cell_column, label_column)
    else:
        labels = annotations_from_validation_obs(split_root, label_column)
    lookup = labels.set_index("cell_id")["label"]
    print(f"Loaded {len(lookup)} unique reference cell labels")
    unmapped = labels[~labels["label"].isin(stage_map)]
    if len(unmapped):
        print("Labels NOT in the predefined stage_map (excluded from scoring):")
        print(unmapped["label"].value_counts().head(25).to_string())

    rows = []
    for directory in sorted(results_root.glob("pair_*")):
        pred_file = directory / "predictions.npz"
        metrics_path = directory / "metrics.json"
        if not pred_file.exists():
            print(f"SKIP {directory.name}: {pred_file.name} unavailable (no retraining needed if restored)")
            continue
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        with np.load(pred_file, allow_pickle=False) as saved:
            ids = saved["cell_ids"].astype(str)
            time = np.asarray(saved["time"], dtype=np.float64).reshape(-1)
        if len(ids) != len(time):
            raise ValueError(f"{directory.name}: cell_ids/time length mismatch")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{directory.name}: duplicate cell IDs in predictions.npz")
        raw_labels = np.asarray([lookup.get(i, "") for i in ids], dtype=str)
        stages = np.asarray([stage_map.get(x, np.nan) for x in raw_labels], dtype=float)
        valid = np.isfinite(stages) & np.isfinite(time)
        coverage = float(valid.mean()) if len(ids) else 0.0
        unique_stages = np.unique(stages[valid])
        if coverage < args.min_coverage or len(unique_stages) < 2:
            raise ValueError(
                f"{directory.name}: valid-label coverage {coverage:.1%}, "
                f"{len(unique_stages)} unique stages; need >= {args.min_coverage:.0%} "
                "and >= 2 stages. Check cell IDs, label names, and stage_map."
            )

        orient = bool(model_cfg.get("evaluation", {}).get("orient_time_for_metric", True))
        t3 = evaluate_t3(time[valid], stages[valid], orient=orient)
        ratio = int(directory.name.removeprefix("pair_"))
        row = {
            "dataset": cfg["name"], "paired_ratio": ratio,
            "n_val": int(len(ids)), "n_eval": int(valid.sum()),
            "coverage": round(coverage, 6),
            "POA_pct": 100 * float(t3["POA"]),
            "POAUC_pct": 100 * float(t3["POAUC"]),
        }
        rows.append(row)
        print(f"{directory.name}: n={row['n_eval']}/{row['n_val']}, "
              f"POA={row['POA_pct']:.2f}%, POAUC={row['POAUC_pct']:.2f}%")
        if args.dry_run:
            continue

        df = pd.DataFrame({
            "cell_id": ids[valid], "label": raw_labels[valid],
            "stage": stages[valid], "pred_time": time[valid],
        })
        df.to_csv(directory / "T3_stage_reference.csv", index=False)
        if not (directory / "metrics.before_t3.json").exists():
            shutil.copy2(metrics_path, directory / "metrics.before_t3.json")
        m = json.loads(metrics_path.read_text())
        m["T3"] = {
            **t3, "n_eval": int(valid.sum()), "n_val": int(len(ids)),
            "coverage": coverage,
            "reference": f"external cell-type annotation ({label_column}), mapped to ordinal stage_map",
        }
        m.setdefault("protocol", {})["T3_input_mode"] = model_cfg["evaluation"]["t3_input_mode"]
        m["protocol"]["T3_reference_orientation_only_at_evaluation"] = orient
        metrics_path.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")

    if not rows:
        raise RuntimeError(
            f"No predictions.npz found in {results_root}. "
            "Restore the existing prediction files first; model retraining is unnecessary."
        )
    if not args.dry_run:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        labels.rename(columns={"label": label_column}).to_csv(output_csv, index=False)
        pd.DataFrame(rows).sort_values("paired_ratio").to_csv(
            results_root / "t3_summary.csv", index=False
        )
        print(f"Saved reference labels: {output_csv}")
        print(f"Saved T3 summary: {results_root / 't3_summary.csv'}")
    else:
        print("Dry run finished. No files changed.")


if __name__ == "__main__":
    main()
