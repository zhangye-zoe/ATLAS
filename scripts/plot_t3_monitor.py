"""Plot T3 monitoring curves for one pair_* directory. Uses monitoring cells only.

Usage:
 python scripts/plot_t3_monitor.py \
   results/t3_early_stop/full/consensus/seed_1234/PBMC/pair_000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    file = args.run_dir / "t3_monitor_history.json"
    if not file.exists():
        parser.error(f"Missing monitor history: {file}")
    records = json.loads(file.read_text(encoding="utf-8"))
    if not records:
        parser.error("Empty monitoring history")

    x = [r["global_epoch"] for r in records]
    poa = [r["POA"] for r in records]
    poauc = [r["POAUC"] for r in records]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(x, poa, marker="o", markersize=3, label="Monitor POA")
    ax.plot(x, poauc, marker="s", markersize=3, label="Monitor POAUC")
    changes = [
        (records[i]["global_epoch"], records[i]["phase"])
        for i in range(1, len(records))
        if records[i]["phase"] != records[i-1]["phase"]
    ]
    for epoch, phase in changes:
        ax.axvline(epoch, linestyle="--", linewidth=1, alpha=0.4)
        ax.text(epoch, 0.98, phase, rotation=90,
                ha="right", va="top", transform=ax.get_xaxis_transform())
    ax.set(xlabel="Global training epoch", ylabel="T3 score",
           title=f"T3 checkpoint-selection curve: {args.run_dir.name}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    out = args.run_dir / "t3_monitor_curve.png"
    fig.savefig(out, dpi=180)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
