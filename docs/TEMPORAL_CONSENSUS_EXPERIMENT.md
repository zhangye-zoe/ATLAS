# Temporal consensus ablation for ATLAS T3

## Why this change

In the fixed PBMC validation intersection (378 cells over six pairing ratios), no single
time estimator (head, graph_peripheral, graph_auto, DPT) won at every pairing ratio.
For example, head won at 20%, while DPT/graph_auto were better at 60%. The scores
also show that at 0% paired, **all four methods were weak**. This is evidence
against selecting one global evaluator simply because it wins a particular
validation subset; it is **not evidence that fusion will necessarily improve T3**.

The refactored model already builds graph, diffusion-like and attribution-proxy
time candidates to distill the neural TimeHead. Its historical fusion uses one
**global** reliability weight per estimator. This experimental change introduces:

1. **Local reliability fusion**, computing per-cell estimator weights from
   neighborhood continuity and cross-estimator agreement in rank space. Candidate
   signs are aligned to an unsupervised graph reference; no stage labels are used.
2. **Confidence-weighted temporal distillation**, downweighting high-disagreement
   cells within the Phase-III TimeHead MSE / pair-ordering term (not zeroing them).
3. **Explicit consensus T3 inference** over head/graph/diffusion/attribution
   time estimates, using the same label-free local fusion, instead of treating
   the learned TimeHead as the only choice.

The existing original implementation stays unchanged by default. The experimental
configuration is kept in a **separate YAML**. No T1/T2 model architecture or
unpaired prediction loss changes are made.

## Files

- src/atlas/temporal.py — optional method=local_consensus
- src/atlas/losses.py — optional confidence for time_distillation_loss
- src/atlas/trainer.py — training integration and evaluation.t3_time_mode=consensus
- configs/model_table2_temporal_consensus.yaml — proposed experiment settings
- experiments/run_temporal_consensus.sh — matched-seed training and isolated outputs
- scripts/summarize_temporal_consensus.py — seed-level aggregation
- tests/test_temporal_consensus.py — synthetic unit tests

## Validation: quick smoke test

Run from the ATLAS root on a GPU server, with the original processed data and
the validated PBMC cell-type annotation configured in
configs/datasets/pbmc.yaml (external_labels.csv):

~~~bash
cd /data5/zhangye/ATLAS
python -m pytest -q tests/test_temporal_consensus.py

QUICK=1 PAIR_RATIOS="0" SEEDS="1234" \
  ARMS="baseline consensus" \
  bash experiments/run_temporal_consensus.sh pbmc cuda:0
~~~

This uses 2/1/1 epochs (a functional smoke test, **not a performance test**).
Results are written under results/temporal_consensus/smoke/ and never overwrite
results/table2/.

## Actual paired comparison on the same data/seed

~~~bash
PAIR_RATIOS="0 20 40 60 80 100" \
  SEEDS="1234 2022 3407" \
  ARMS="baseline consensus" \
  bash experiments/run_temporal_consensus.sh pbmc cuda:0
~~~

For a more informative, but expensive, **2x2 isolation** of the two changes:

~~~bash
OUTPUT_PARENT="/data5/zhangye/ATLAS/results/temporal_consensus/full_factorial" \
  PAIR_RATIOS="0 20 40 60 80 100" \
  SEEDS="1234 2022 3407" \
  ARMS="baseline inference_only teacher_only consensus" \
  bash experiments/run_temporal_consensus.sh pbmc cuda:0
~~~

Definitions:

| arm | training teacher | T3 inference estimator |
| --- | --- | --- |
| baseline | original global risk weights | TimeHead |
| inference_only | original global risk weights | new local consensus |
| teacher_only | new local consensus + confidence-weighted distillation | TimeHead |
| consensus | new teacher | new local consensus |

The **same seed** and **same dataset pairing ratio** are applied across arms.
The script uses the original epoch counts (80/60/40) for all arms, preserving
the existing architecture, optimizer, training splits, and T1/T2 definitions.
It creates a separate config_used.yaml, command.txt, train.log and model
checkpoint for each run. Existing result directories cause the script to exit,
rather than silently overwrite previous runs. One seed and 6 ratios over 4
arms requires 24 complete trainings; start with the smoke test.

Other examples:

~~~bash
# Save resources: only compare 0%, 20% and 60% in first real run
PAIR_RATIOS="0 20 60" SEEDS="1234" \
  ARMS="baseline inference_only teacher_only consensus" \
  bash experiments/run_temporal_consensus.sh pbmc cuda:0

# BMMC (check external stage labels before interpreting POA/POAUC)
PAIR_RATIOS="0 20 60 100" SEEDS="1234" \
  bash experiments/run_temporal_consensus.sh bmmc cuda:0
~~~

## Results

For the default full run:

- results/temporal_consensus/full/baseline/seed_1234/PBMC/pair_000/metrics.json
- results/temporal_consensus/full/consensus/seed_1234/PBMC/pair_000/metrics.json
- results/temporal_consensus/full/consensus/seed_1234/PBMC/pair_000/T3_consensus_diagnostics.json
- results/temporal_consensus/full/consensus/seed_1234/PBMC/pair_000/temporal_teacher.json
- results/temporal_consensus/full/summary_across_seeds.csv
- results/temporal_consensus/full/paired_seed_differences.csv

The aggregation records **mean, sample standard deviation and valid run count**.
Use the paired-seed differences to avoid comparing unrelated random seeds.
POA/POAUC require reference stage labels; without them the columns remain empty.
Because preprocessing currently changes the validation ATAC-query cohort with
the pairing ratio, comparison is controlled *within the same ratio*. A fixed
held-out test cohort should be used for final cross-ratio claims.

### Interpretation / safeguards

- New fused time is a **pseudotime estimator**, not ground-truth elapsed time.
- Its sign is aligned using the unsupervised graph reference; reference stage
  labels are permitted only by the existing **evaluation-only** direction
  selection and POA/POAUC scoring.
- If some modes improve while others fall, do not cherry-pick the mode by
  test-set labels; select one on independent validation evidence, then freeze it.
- Label-based stage-ranking in legacy V2 scripts is a *different, supervised*
  protocol; this experiment does not bring it into training.
- The four methods shown in the previous diagnostic were evaluated on a
  shared set of **378 PBMC cells**. Those scores do not measure a consensus
  estimator and **cannot establish that the new training will improve**.
- The experimental code introduces no T4 capability. Future-state evaluation
  requires a separate validated benchmark and care to prevent lineage
  label leakage.

## If the GPU server cannot fetch from GitHub

Download this branch on a network-connected workstation:

~~~bash
git clone --depth 1 --branch experiment/robust-time-consensus \
  https://github.com/zhangye-zoe/ATLAS.git ATLAS_time_patch
~~~

Then transfer the changed files via scp to corresponding directories under
/data5/zhangye/ATLAS/, without replacing any original experiment directories.
The README and the pull request link the exact source files.
