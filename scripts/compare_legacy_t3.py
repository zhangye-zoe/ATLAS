#!/usr/bin/env python3
"""Compare legacy CORAL T3 estimators on *saved* ATLAS embeddings, without training.

Examples (from repository root):
  python scripts/compare_legacy_t3.py --dataset pbmc \
    --annotations data/external/annotations/pbmc_azimuth_annotations.csv \
    --modes head graph_peripheral graph_auto dpt

  # Same validation cells across all pairing ratios:
  python scripts/compare_legacy_t3.py --dataset pbmc --cell-set intersection

Outputs: results/table2/PBMC/T3_legacy_comparison.csv and
         results/table2/PBMC/T3_legacy_comparison_protocol.json

WARNING: this is a post-hoc *diagnostic*. Candidate scores must not be treated
as an independently selected test result; selecting the best candidate using
validation stage labels would inflate claimed performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atlas.legacy_t3 import get_t3_candidates, score_ordering_fast  # noqa: E402


def read_annotation(path: Path, cell_col: str, label_col: str) -> dict[str, str]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [x for x in [cell_col, label_col] if x not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}. Available: {list(df.columns)}"
        )
    df = df[[cell_col, label_col]].copy()
    df.columns = ["cell_id", "label"]
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df.cell_id != "") & (~df.label.str.lower().isin({"", "nan", "none", "null", "na"}))]
    nuniq = df.groupby("cell_id")["label"].nunique()
    if any(nuniq > 1):
        raise ValueError(f"Conflicting cell annotations for IDs: {list(nuniq[nuniq > 1].index[:5])}")
    return df.drop_duplicates("cell_id").set_index("cell_id")["label"].to_dict()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["pbmc", "bmmc"], default="pbmc")
    p.add_argument("--annotations", type=Path, default=None)
    p.add_argument("--cell-column", type=str, default=None)
    p.add_argument("--label-column", type=str, default=None)
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--cell-set", choices=["intersection", "per_run"], default="intersection")
    p.add_argument("--modes", nargs="+", default=["head", "graph_peripheral", "graph_auto", "dpt"],
                   choices=["head", "graph_peripheral", "graph_auto", "dpt", "graph_root"])
    p.add_argument("--graph-k", type=int, default=30)
    p.add_argument("--root-labels", nargs="*", default=None,
                   help="Optional label-guided root; diagnostics ONLY, not valid unsupervised T3")
    p.add_argument("--no-orient", action="store_true",
                   help="Disable reference-based orientation for all candidate metrics")
    p.add_argument("--min-coverage", type=float, default=0.5)
    args = p.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/datasets" / f"{args.dataset}.yaml").read_text())
    external = cfg.get("external_labels") or {}
    source = args.annotations or (Path(external["csv"]) if external.get("csv") else None)
    if source is not None and not source.is_absolute():
        source = ROOT / source
    cell_col = args.cell_column or external.get("cell_column", "cell_id")
    label_col = args.label_column or external.get("label_column", "cell_type")
    annotation = read_annotation(source, cell_col, label_col) if source else {}
    if not annotation:
        print("WARNING: no reference annotation CSV supplied; only label-free times can be computed.")
    stage_map = {str(k): int(v) for k, v in cfg.get("stage_map", {}).items()}
    results_root = args.results_root or ROOT / "results" / "table2" / cfg["name"]
    files = sorted(results_root.glob("pair_*/predictions.npz"))
    if not files:
        raise FileNotFoundError(
            f"No predictions.npz under {results_root}; needs saved embeddings and time, not model weights"
        )
    runs = []
    for file in files:
        with np.load(file, allow_pickle=False) as values:
            ids = values["cell_ids"].astype(str)
            h = np.asarray(values["h_fused"], dtype=np.float32)
            time = np.asarray(values["time"], dtype=np.float32)
            # External annotations take precedence; fall back to saved labels only when available.
            labels = (np.asarray(values["labels"]).astype(str)
                      if "labels" in values.files else None)
        if not (len(ids) == len(time) == len(h)):
            raise ValueError(f"Incompatible NPZ dimensions: {file}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate cell IDs: {file}")
        if annotation:
            labels = np.asarray([annotation.get(i, "") for i in ids], dtype=str)
        runs.append((file.parent.name, ids, h, time, labels))

    common = set(runs[0][1])
    for _, ids, _, _, _ in runs[1:]:
        common &= set(ids)
    common = sorted(common)
    print(f"Detected {len(runs)} runs; shared cell IDs: {len(common)}")
    if args.cell_set == "intersection" and len(common) < 3:
        raise ValueError(
            "Fewer than 3 cells are common across all ratios. Use --cell-set per_run, "
            "or compare a smaller set of runs with a stable evaluation cohort."
        )
    if "graph_root" in args.modes and not args.root_labels:
        raise ValueError("graph_root requires --root-labels (labels used to select root; diagnostic only)")

    rows, problems = [], []
    for dirname, ids, h, time, labels in runs:
        keep = (np.isin(ids, common) if args.cell_set == "intersection"
                else np.ones(len(ids), dtype=bool))
        h, time, ids = h[keep], time[keep], ids[keep]
        labels = labels[keep] if labels is not None else None
        if labels is not None:
            stages = np.asarray([stage_map.get(v, np.nan) for v in labels], dtype=float)
            coverage = float(np.isfinite(stages).mean())
            print(f"{dirname}: stage-map coverage = {coverage:.1%} ({len(ids)} cells)")
            if coverage < args.min_coverage:
                raise ValueError(
                    f"{dirname}: coverage {coverage:.1%} is below --min-coverage {args.min_coverage:.0%}. "
                    "Check cell barcodes and annotation naming."
                )
        else:
            stages, coverage = None, 0.0

        # Each candidate has an independent inference path; missing scanpy should
        # not prevent reporting the always-available head and peripheral candidates.
        for mode in args.modes:
            try:
                times, info = get_t3_candidates(
                    h, time, [mode], graph_k=args.graph_k,
                    root_labels=args.root_labels, cell_labels=labels,
                )
                t = times[mode]
                vals = (score_ordering_fast(t, stages, orient=not args.no_orient)
                        if stages is not None and len(np.unique(stages[np.isfinite(stages)])) >= 2
                        else None)
            except (ImportError, ValueError) as exc:
                problems.append({"ratio": dirname, "mode": mode, "error": str(exc)})
                print(f"SKIP {dirname}/{mode}: {exc}")
                continue
            row = {
                "ratio": dirname, "mode": mode,
                "n_cells": len(ids), "n_eval": vals["n_eval"] if vals else 0,
                "label_coverage": coverage,
                "POA": vals["POA"] if vals else None,
                "POAUC": vals["POAUC"] if vals else None,
                "flipped_by_reference": vals["flipped_by_reference"] if vals else None,
                "root_is_label_guided": mode == "graph_root",
                "source": info[mode],
                "evaluation_cell_set": args.cell_set,
            }
            rows.append(row)
            score_str = (f"POA={row['POA']:.4f}, POAUC={row['POAUC']:.4f}"
                         if vals is not None else "no reference labels")
            print(f"  {dirname}: {mode:20s} {score_str}")

    if not rows:
        raise RuntimeError("No T3 candidates were successfully computed")
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "T3_legacy_comparison.csv"
    meta_path = results_root / "T3_legacy_comparison_protocol.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    protocol = {
        "purpose": "post-hoc trajectory estimator diagnostic, NOT automatic test-model selection",
        "source": "saved predictions.npz: h_fused and time",
        "annotation_source": str(source) if source else None,
        "cell_set": args.cell_set,
        "intersection_size": len(common),
        "modes": args.modes,
        "graph_k": args.graph_k,
        "reference_based_orientation": not args.no_orient,
        "label_guided_graph_root_invalid_as_unsupervised": "graph_root" in args.modes,
        "root_labels": args.root_labels,
        "skipped": problems,
        "warning": (
            "Legacy T3 used multiple estimators and validation cohort rules. "
            "Compare the same mode and same cells; do not select best test candidate post hoc. "
            "No training or T4 future-state computation is performed."
        ),
    }
    meta_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {csv_path}")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
