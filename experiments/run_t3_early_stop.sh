#!/usr/bin/env bash
# T3 early stop: selection cells (30%) vs held-out report (70%)
# Usage: PAIR_RATIOS="0" SEEDS="1234" bash experiments/run_t3_early_stop.sh pbmc cuda:0
# QUICK=1 triggers 2/1/1 epoch smoke test, not scientific model selection.
set -euo pipefail

DATASET="${1:-pbmc}"
DEVICE="${2:-cuda:0}"
case "$DATASET" in pbmc|bmmc) ;; *) echo "dataset must be pbmc/bmmc" >&2; exit 2;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PAIR_RATIOS="${PAIR_RATIOS:-0 20 40 60 80 100}"
SEEDS="${SEEDS:-1234}"
ARMS="${ARMS:-baseline consensus}"
QUICK="${QUICK:-0}"
if [[ "$QUICK" == "1" ]]; then RUN_KIND=smoke; else RUN_KIND=full; fi
OUTPUT_PARENT="${OUTPUT_PARENT:-$ROOT/results/t3_early_stop/$RUN_KIND}"
mkdir -p "$OUTPUT_PARENT"
read -r -a RATIOS <<< "$PAIR_RATIOS"
read -r -a SEED_ARR <<< "$SEEDS"
read -r -a ARM_ARR <<< "$ARMS"
for arm in "${ARM_ARR[@]}"; do
  case "$arm" in baseline|consensus) ;; *) echo "Unknown arm: $arm" >&2; exit 2;; esac
done

if [[ "$DEVICE" == cuda* ]]; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; refusing accidental CPU training")
PY
fi

for seed in "${SEED_ARR[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then echo "Invalid seed $seed" >&2; exit 2; fi
  for arm in "${ARM_ARR[@]}"; do
    OUT_ROOT="$OUTPUT_PARENT/$arm/seed_$seed"
    if [[ -e "$OUT_ROOT/${DATASET^^}" ]]; then
      echo "Refusing to overwrite existing run $OUT_ROOT/${DATASET^^}" >&2
      exit 3
    fi
    mkdir -p "$OUT_ROOT"
    CFG="$OUT_ROOT/config_used.yaml"
    python - "configs/model_table2_t3_early_stop.yaml" "$CFG" "$seed" "$arm" <<'PY'
import sys, yaml
source, target, seed, arm = sys.argv[1:5]
with open(source, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["seed"] = int(seed)
if arm == "baseline":
    cfg["temporal"]["fusion_method"] = "global"
    cfg["evaluation"]["t3_time_mode"] = "head"
with open(target, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    cmd=(python experiments/run_table2.py
      --dataset "$DATASET" --pair-ratios "${RATIOS[@]}"
      --model-config "$CFG" --output-root "$OUT_ROOT" --device "$DEVICE")
    if [[ "$QUICK" == "1" ]]; then cmd+=(--quick); fi
    printf '%q ' "${cmd[@]}" > "$OUT_ROOT/command.txt"
    printf '\n' >> "$OUT_ROOT/command.txt"
    echo "=== $DATASET $arm seed=$seed; stage-monitor/heldout report ==="
    "${cmd[@]}" 2>&1 | tee "$OUT_ROOT/train.log"
  done
done
python scripts/summarize_temporal_consensus.py --results-root "$OUTPUT_PARENT"
echo "Done: $OUTPUT_PARENT/summary_across_seeds.csv"
