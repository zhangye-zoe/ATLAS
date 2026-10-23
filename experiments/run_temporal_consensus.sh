#!/usr/bin/env bash
# Matched-seed experimental ablation:
#   QUICK=1 PAIR_RATIOS="0" bash experiments/run_temporal_consensus.sh pbmc cuda:0
#   PAIR_RATIOS="0 20 40 60 80 100" SEEDS="1234 2022 3407" \
#     bash experiments/run_temporal_consensus.sh pbmc cuda:0
# ARMS="consensus" runs only the proposed model.
# Optional full 2x2 ablation: ARMS="baseline inference_only teacher_only consensus".
# Set OUTPUT_PARENT to a fresh path for each run.
set -euo pipefail

DATASET="${1:-pbmc}"
DEVICE="${2:-cuda:0}"
case "$DATASET" in pbmc|bmmc) ;; *) echo "dataset must be pbmc or bmmc" >&2; exit 2;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PAIR_RATIOS="${PAIR_RATIOS:-0 20 40 60 80 100}"
SEEDS="${SEEDS:-1234}"
ARMS="${ARMS:-baseline consensus}"
QUICK="${QUICK:-0}"
if [[ "$QUICK" == "1" ]]; then RUN_KIND="smoke"; else RUN_KIND="full"; fi
OUTPUT_PARENT="${OUTPUT_PARENT:-${ROOT}/results/temporal_consensus/${RUN_KIND}}"
mkdir -p "$OUTPUT_PARENT"

read -r -a RATIOS <<< "$PAIR_RATIOS"
read -r -a SEED_ARR <<< "$SEEDS"
read -r -a ARM_ARR <<< "$ARMS"
if ((${#RATIOS[@]} == 0 || ${#SEED_ARR[@]} == 0 || ${#ARM_ARR[@]} == 0)); then
  echo "PAIR_RATIOS, SEEDS, ARMS cannot be empty" >&2
  exit 2
fi

if [[ "$DEVICE" == cuda* ]]; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA requested but unavailable; refusing accidental CPU training")
PY
fi
for arm in "${ARM_ARR[@]}"; do
  case "$arm" in baseline|inference_only|teacher_only|consensus) ;; *) echo "invalid arm=$arm" >&2; exit 2;; esac
done

printf 'Dataset=%s; device=%s; ratios=%s; seeds=%s; arms=%s\n' \
  "$DATASET" "$DEVICE" "$PAIR_RATIOS" "$SEEDS" "$ARMS"

for seed in "${SEED_ARR[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "invalid seed: $seed" >&2
    exit 2
  fi
  for arm in "${ARM_ARR[@]}"; do
    if [[ "$arm" == baseline || "$arm" == inference_only ]]; then
      SOURCE_CFG="$ROOT/configs/model_table2.yaml"
    else
      SOURCE_CFG="$ROOT/configs/model_table2_temporal_consensus.yaml"
    fi
    OUT_ROOT="$OUTPUT_PARENT/$arm/seed_$seed"
    RUN_DIR="$OUT_ROOT/${DATASET^^}"
    if [[ -e "$RUN_DIR" ]]; then
      echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
      echo "Use a new OUTPUT_PARENT to preserve previous experiments" >&2
      exit 3
    fi
    mkdir -p "$OUT_ROOT"
    CFG="$OUT_ROOT/config_used.yaml"
    python - "$SOURCE_CFG" "$CFG" "$seed" "$arm" <<'PY'
import sys, yaml
source, target, seed, arm = sys.argv[1:5]
with open(source, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["seed"] = int(seed)
# The extra arms isolate the source of any gain (training vs inference):
# baseline=global/head, inference_only=global/consensus,
# teacher_only=local/head, consensus=local/consensus.
if arm == "inference_only":
    cfg["evaluation"]["t3_time_mode"] = "consensus"
elif arm == "teacher_only":
    cfg["evaluation"]["t3_time_mode"] = "head"
with open(target, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    echo "=== $arm seed=$seed dataset=$DATASET output=$OUT_ROOT ==="
    cmd=(python experiments/run_table2.py --dataset "$DATASET"
      --pair-ratios "${RATIOS[@]}" --model-config "$CFG"
      --output-root "$OUT_ROOT" --device "$DEVICE")
    if [[ "$QUICK" == "1" ]]; then cmd+=(--quick); fi
    printf '%q ' "${cmd[@]}" > "$OUT_ROOT/command.txt"
    printf '\n' >> "$OUT_ROOT/command.txt"
    "${cmd[@]}" 2>&1 | tee "$OUT_ROOT/train.log"
  done
done
python scripts/summarize_temporal_consensus.py --results-root "$OUTPUT_PARENT"
echo "Finished. See $OUTPUT_PARENT/summary_across_seeds.csv"
