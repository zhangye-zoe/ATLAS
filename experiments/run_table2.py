#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import pandas as pd

# Allow running directly from repository root without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas.data import load_table2_split
from atlas.early_stopping import split_monitor_report
from atlas.model import ATLAS
from atlas.trainer import evaluate_table2, train_atlas
from atlas.utils import get_device, load_yaml, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser(
        description="Run ATLAS Table-2 tasks T1/T2/T3 on PBMC or BMMC."
    )
    p.add_argument("--dataset", choices=["pbmc", "bmmc"], required=True)
    p.add_argument("--pair-ratios", type=int, nargs="+", default=[0, 40, 100])
    p.add_argument("--model-config", default=str(REPO_ROOT / "configs/model_table2.yaml"))
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--output-root", default=str(REPO_ROOT / "results/table2"))
    p.add_argument("--device", default=None, help="Override config device, e.g. cuda:0 or cpu")
    p.add_argument("--label-csv", default=None, help="Optional PBMC Azimuth annotation CSV override")
    p.add_argument("--label-column", default=None, help="Optional external label column override")
    p.add_argument("--quick", action="store_true", help="2/1/1 epochs for a pipeline smoke test")
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg = load_yaml(args.model_config)
    dataset_cfg_path = args.dataset_config or str(REPO_ROOT / f"configs/datasets/{args.dataset}.yaml")
    dataset_cfg = load_yaml(dataset_cfg_path)
    # Portable repository-local data path. A data symlink under this repo is supported.
    if str(dataset_cfg.get("project_root", ".")) in {".", "", "None"}:
        dataset_cfg["project_root"] = str(REPO_ROOT)
    if args.label_csv:
        dataset_cfg.setdefault("external_labels", {})["csv"] = args.label_csv
    if args.label_column:
        dataset_cfg.setdefault("external_labels", {})["label_column"] = args.label_column
    if args.quick:
        model_cfg = copy.deepcopy(model_cfg)
        model_cfg["training"]["phase1_epochs"] = 2
        model_cfg["training"]["phase2_epochs"] = 1
        model_cfg["training"]["phase3_epochs"] = 1
        model_cfg["training"]["refresh_dynamic_every"] = 1

    device = get_device(args.device or model_cfg.get("device", "cuda"))
    output_root = Path(args.output_root) / dataset_cfg["name"]
    output_root.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for pair_ratio in args.pair_ratios:
        seed = int(model_cfg.get("seed", 1234)) + int(pair_ratio)
        seed_everything(seed)
        data = load_table2_split(dataset_cfg, pair_ratio)
        run_dir = output_root / f"pair_{pair_ratio:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        early_cfg = model_cfg.get("early_stopping") or {}
        monitor_data = None
        report_data = data
        partition_info = None
        if early_cfg.get("enabled", False):
            monitor_data, report_data, partition_info = split_monitor_report(
                data,
                monitor_fraction=float(early_cfg.get("monitor_fraction", 0.30)),
                seed=seed,
            )
            save_json(partition_info, run_dir / "t3_validation_partition.json")
            print(
                f"T3 selection: {partition_info['monitor_cells']} labeled cells; "
                f"held-out reporting: {partition_info['report_cells']} cells"
            )

        mcfg = model_cfg["model"]
        input_dims = {m: x.shape[1] for m, x in data.modalities.items()}
        model = ATLAS(
            input_dims=input_dims,
            shared_dim=int(mcfg["shared_dim"]),
            private_dim=int(mcfg["private_dim"]),
            encoder_hidden=tuple(mcfg["encoder_hidden"]),
            decoder_hidden=tuple(mcfg["decoder_hidden"]),
            dropout=float(mcfg["dropout"]),
            lag_edges=(("atac", "rna"),),
        )
        print(f"\n=== ATLAS {data.dataset_name}: paired={pair_ratio}% -> {data.split_dir.name} on {device} ===")
        dynamic = train_atlas(
            model, data, model_cfg, device, run_dir, monitor_data=monitor_data,
        )
        metrics = evaluate_table2(model, report_data, dynamic, model_cfg, device, run_dir)
        if partition_info is not None:
            metrics["protocol"]["T3_evaluation_partition"] = (
                "held-out reporting subset; disjoint from early-stop monitor"
            )
            metrics["protocol"]["T3_monitor_cells"] = partition_info["monitor_cells"]
            metrics["protocol"]["T3_report_cells"] = partition_info["report_cells"]
            metrics["protocol"]["T3_supervised_checkpoint_selection"] = True
            save_json(metrics, run_dir / "metrics.json")
        all_metrics.append(metrics)
        print(metrics)

    # Flat summary convenient for Table 2 assembly.
    rows = []
    for m in all_metrics:
        row = {
            "dataset": m["dataset"],
            "paired_ratio": m["paired_ratio"],
            "split_label": m["split_label"],
            "FOSCTTM_pct": 100.0 * m["T1"]["FOSCTTM"],
            "PED": m["T1"]["PED"],
            "Top1_pct": 100.0 * m["T1"].get("Top1_ACC", float("nan")),
            "Top5_pct": 100.0 * m["T1"].get("Top5_ACC", float("nan")),
            "Top10_pct": 100.0 * m["T1"].get("Top10_ACC", float("nan")),
            "Pearson_pct": 100.0 * m["T2"]["Pearson"],
            "Pearson_gene_pct": 100.0 * m["T2"].get("Pearson_gene", float("nan")),
            "RMSE": m["T2"]["RMSE"],
            "POA_pct": None if m["T3"].get("POA") is None else 100.0 * m["T3"]["POA"],
            "POAUC_pct": None if m["T3"].get("POAUC") is None else 100.0 * m["T3"]["POAUC"],
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_root / "summary.csv", index=False)
    save_json(all_metrics, output_root / "summary.json")
    print(f"\nSaved: {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
