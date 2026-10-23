#!/usr/bin/env python3
"""Aggregate baseline-vs-consensus T1/T2/T3 results across independent seeds.

This script NEVER selects the best model/epoch/estimator using test labels.
Missing metrics (for example missing PBMC stage annotations) stay missing.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["FOSCTTM_pct", "PED", "Pearson_pct", "Pearson_gene_pct",
           "RMSE", "POA_pct", "POAUC_pct"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, required=True)
    a = p.parse_args()
    records = []
    for arm in ("baseline", "consensus"):
        for file in sorted((a.results_root / arm).glob("seed_*/**/summary.csv")):
            match = re.fullmatch(r"seed_(\d+)", file.parent.parent.name)
            if not match:
                raise ValueError(f"Unrecognized seed layout: {file}")
            df = pd.read_csv(file)
            df["seed"] = int(match.group(1))
            df["arm"] = arm
            records.append(df)
    if not records:
        raise FileNotFoundError(f"No seed_*/<dataset>/summary.csv files found in {a.results_root}")
    table = pd.concat(records, ignore_index=True)
    keys = ["dataset", "paired_ratio", "arm"]
    metrics = [m for m in METRICS if m in table.columns]
    for m in metrics:
        table[m] = pd.to_numeric(table[m], errors="coerce")
    table.sort_values(["dataset", "paired_ratio", "arm", "seed"]).to_csv(
        a.results_root / "all_seed_runs.csv", index=False
    )
    columns = {}
    for m in metrics:
        columns[m + "_mean"] = (m, "mean")
        columns[m + "_std"] = (m, "std")
        columns[m + "_n"] = (m, "count")
    grouped = table.groupby(keys, dropna=False).agg(
        runs=("seed", "nunique"), **columns
    ).reset_index()
    grouped.to_csv(a.results_root / "summary_across_seeds.csv", index=False)
    print(grouped.to_string(index=False))

    if set(table["arm"]) == {"baseline", "consensus"}:
        base = table.loc[table.arm == "baseline"].set_index(
            ["dataset", "paired_ratio", "seed"])
        proposed = table.loc[table.arm == "consensus"].set_index(
            ["dataset", "paired_ratio", "seed"])
        both = base.index.intersection(proposed.index)
        if len(both) == 0:
            print("No matched dataset/ratio/seed pairs; cannot compute paired deltas")
            return
        deltas = proposed.loc[both, metrics] - base.loc[both, metrics]
        # Deltas are consensus-minus-baseline, so lower is better for FOS/PED/RMSE.
        deltas = deltas.reset_index()
        deltas.to_csv(a.results_root / "paired_seed_differences.csv", index=False)
        print(f"Saved {len(deltas)} matched-seed differences; interpret signs per metric.")


if __name__ == "__main__":
    main()
