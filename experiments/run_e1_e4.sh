#!/usr/bin/env bash
# E1–E4 controlled ATLAS ablations. Run from any directory.
# E1 full final; E2 PhaseIII-only best; E3 freeze representation; E4 fixed teacher.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASETS="${DATASETS:-pbmc bmmc}"
PAIR_RATIOS="${PAIR_RATIOS:-0 20 100}"
SEEDS="${SEEDS:-1234}"
EXPERIMENTS="${EXPERIMENTS:-E1 E2 E3 E4}"
ESTIMATOR="${ESTIMATOR:-consensus}"
DEVICE="${DEVICE:-cuda:0}"
QUICK="${QUICK:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT/configs/model_table2_t3_early_stop.yaml}"
OUTPUT_PARENT="${OUTPUT_PARENT:-$ROOT/results/e1_e4_ablations/$(if [[ "$QUICK" == "1" ]]; then echo smoke; else echo full; fi)_${ESTIMATOR}}"

read -r -a datasets <<< "$DATASETS"
read -r -a ratios <<< "$PAIR_RATIOS"
read -r -a seeds <<< "$SEEDS"
read -r -a experiments <<< "$EXPERIMENTS"
[[ "$ESTIMATOR" == "consensus" || "$ESTIMATOR" == "head" ]] || { echo "ESTIMATOR must be head/consensus" >&2; exit 2; }
[[ "$QUICK" == "0" || "$QUICK" == "1" ]] || { echo "QUICK must be 0/1" >&2; exit 2; }
for e in "${experiments[@]}"; do
  [[ "$e" =~ ^E[1234]$ ]] || { echo "invalid experiment: $e" >&2; exit 2; }
done
for d in "${datasets[@]}"; do
  [[ "$d" == "pbmc" || "$d" == "bmmc" ]] || { echo "invalid dataset: $d" >&2; exit 2; }
done
for r in "${ratios[@]}"; do
  [[ "$r" =~ ^(0|20|40|60|80|100)$ ]] || { echo "invalid ratio: $r" >&2; exit 2; }
done
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "invalid seed: $seed" >&2; exit 2; }
done
[[ -f "$BASE_CONFIG" ]] || { echo "Missing base configuration: $BASE_CONFIG" >&2; exit 2; }
if [[ "$DEVICE" == cuda* ]]; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('Requested CUDA but it is unavailable')
PY
fi

mkdir -p "$OUTPUT_PARENT"
python - "$ROOT" <<'PY'
import sys
sys.path.insert(0,str(__import__('pathlib').Path(sys.argv[1])/'src'))
import inspect
from atlas.trainer import train_atlas
code=inspect.getsource(train_atlas)
assert 'selection_policy' in code and 'freeze_phase3' in code and 'fixed_teacher' in code, 'Please apply scripts/apply_e1_e4_patch.py first.'
print('E1-E4 trainer protocol verified.')
PY

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    for arm in "${experiments[@]}"; do
      RUN_ROOT="$OUTPUT_PARENT/$arm/seed_$seed"
      DATASET_DIR="$RUN_ROOT/${dataset^^}"
      mkdir -p "$RUN_ROOT"
      # Refuse to overwrite or mix incomplete runs. Re-running with SKIP_EXISTING=1
      # only skips a complete dataset directory with summary.csv.
      if [[ -e "$DATASET_DIR" ]]; then
        if [[ "$SKIP_EXISTING" == "1" && -f "$DATASET_DIR/summary.csv" ]]; then
          echo "SKIP complete: $DATASET_DIR"
          continue
        fi
        echo "Existing directory, refusing overwrite: $DATASET_DIR" >&2
        echo "Use a new OUTPUT_PARENT. Incomplete training isn't automatically resumed." >&2
        exit 3
      fi

      CFG="$RUN_ROOT/${arm}_${dataset}_config_used.yaml"
      python - "$BASE_CONFIG" "$CFG" "$arm" "$seed" "$ESTIMATOR" "$QUICK" <<'PY'
import sys, yaml
source,out,arm,seed,estimator,quick=sys.argv[1:]
with open(source,encoding='utf-8') as f: cfg=yaml.safe_load(f)
assert arm in {'E1','E2','E3','E4'}
assert estimator in {'head','consensus'}
cfg['seed']=int(seed)
cfg['fixed_seed_across_ratios']=True
cfg.setdefault('temporal',{})['fusion_method']='local_consensus' if estimator=='consensus' else 'global'
cfg.setdefault('evaluation',{})['t3_time_mode']=estimator
es=cfg.setdefault('early_stopping',{})
es.update({
    'enabled':True,                       # same monitor/report split in ALL arms
    'selection_policy':'final' if arm=='E1' else 'phase3',
    'freeze_representation_phase3':arm=='E3',
    'fixed_teacher_phase3':arm=='E4',
    'monitor_fraction':0.30,
    'metric':'POAUC',
    'eval_every':1 if quick=='1' else 5,
    'patience':4,
    'min_delta':0.002,
    'min_phase3_epochs':1 if quick=='1' else 10,
})
with open(out,'w',encoding='utf-8') as f: yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=True)
PY
      cmd=(python experiments/run_table2.py --dataset "$dataset"
           --pair-ratios "${ratios[@]}" --model-config "$CFG"
           --output-root "$RUN_ROOT" --device "$DEVICE")
      if [[ "$QUICK" == "1" ]]; then cmd+=(--quick); fi
      printf '%q ' "${cmd[@]}" > "$RUN_ROOT/command_${dataset}.txt"
      printf '\n' >> "$RUN_ROOT/command_${dataset}.txt"
      echo "========================================================"
      echo "  $arm | $dataset | seed=$seed | ratios=$PAIR_RATIOS"
      echo "  estimator=$ESTIMATOR | QUICK=$QUICK"
      echo "  output=$DATASET_DIR"
      echo "========================================================"
      # Allow shell PIPESTATUS to fail training; do not publish comparisons on incomplete runs.
      "${cmd[@]}" 2>&1 | tee "$RUN_ROOT/train_${dataset}.log"
    done
  done
done
python scripts/summarize_e1_e4.py --results-root "$OUTPUT_PARENT"
echo "Finished. Summary: $OUTPUT_PARENT/summary_e1_e4.csv"
