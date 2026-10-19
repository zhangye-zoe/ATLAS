#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python experiments/run_table2.py --dataset pbmc --pair-ratios 0 40 100 "$@"
