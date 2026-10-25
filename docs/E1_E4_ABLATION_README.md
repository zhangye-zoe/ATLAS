# ATLAS E1–E4 controlled training ablations

**Purpose:** diagnose whether T3 stage-order degradation is driven by checkpoint
selection, changes in the encoder/fusion representation, or periodic temporal
teacher updates. This package **does not require rerunning any old experiment**
and **does not overwrite old results**.

## Exact experiment definitions

| Arm | Phase I/II | Phase III | Final checkpoint | Temporal teacher |
|---|---|---|---|---|
| E1 | Normal 80/60 | Normal 40 | **Final** Phase-III weights; no stopping | Every 5 epochs |
| E2 | Normal 80/60 | Normal (up to 40) | Best **Phase-III** monitor POAUC (eligible epoch >=10) | Every 5 epochs |
| E3 | Normal 80/60 | Freeze encoders, fusion, fusion_refine, decoders and cross predictors; update TimeHead + lag modules/decoders | Best Phase-III monitor POAUC | Every 5 epochs (representation now fixed) |
| E4 | Normal 80/60 | Normal (up to 40) | Best Phase-III monitor POAUC | **Fixed** at Phase-III entry; **lag banks still refreshed** with latest latent states and the fixed time axis |

This uses **one** T3 inference estimator per experiment sweep, controlled by
`ESTIMATOR=consensus` (default) or `ESTIMATOR=head`. Do **not** compare arms
trained under different estimator settings as though they were a single study.
All arms use the exact same base YAML and `lambda_lag`, loss weights, batch size,
seed and dataset paths. No new experimental metric is substituted for T1/T2/T3.

**Selection/fairness:** All four arms expose ~30% of validation reference-stage
labels to the monitoring protocol and hold out the SAME ~70% for final T3
reporting. E1 uses those monitor labels only to choose orientation, not to
select its checkpoint. E2-E4 use monitor labels for supervised checkpoint
selection and orientation. Models never train directly on stage labels.
The same unlabeled validation cells remain in each transductive inference graph.
**Do not interpret held-out reference stages as an independent unseen dataset.**
Selecting the overall experimental design on these reported scores also leaks
information; new held-out datasets are needed for final paper claims.

**Timing and patient grouping:** The split is label-stratified by cell, not donor;
results may not represent donor-level generalization. BMMC's stage map has
branches; POAUC is a coarse ordering validation, not true kinetic time.

**Important compatibility:** This patch targets the source layout of the
ATLAS **PR #5** / T3 early-stopping branch. It uses strict text anchors. The
patch command aborts if your server's local `trainer.py` differs in a relevant
place. It backs up your original trainer and run_table2.py files with timestamped
`.bak` suffixes. Your README, `.gitignore`, custom PBMC annotation path, and
other source files are left unchanged.

Default `selection_policy` is `legacy_all` for compatibility with the old
`run_t3_early_stop.sh`. Only newly generated E1-E4 config files use `final` or
`phase3`. QUICK smoke mode reduces epochs to 2/1/1 and changes the eligibility
threshold to 1 epoch; QUICK results are *not* model comparisons.

## Install: copy the archive to the Linux server

From Mac terminal, assuming the ZIP was downloaded to `~/Downloads`:

```bash
scp ~/Downloads/ATLAS_E1_E4_Experiments.zip zhangye@10.249.40.97:/tmp/
```

On Linux:

```bash
cd /data5/zhangye/ATLAS
unzip -o /tmp/ATLAS_E1_E4_Experiments.zip -d .

# Review local modifications; do not reset anything.
git status --short

# Validate that the patch matches your local version BEFORE editing code:
python scripts/apply_e1_e4_patch.py --root . --check-only

# Apply backup-aware patch:
python scripts/apply_e1_e4_patch.py --root .
python -m py_compile src/atlas/trainer.py experiments/run_table2.py
python scripts/test_e1_e4_patch.py
python -m pytest -q tests/test_t3_early_stopping.py tests/test_temporal_consensus.py
```

Test the full runner on only one ratio/arm and a unique output path:

```bash
QUICK=1 DATASETS='pbmc' PAIR_RATIOS='0' SEEDS='1234' \
EXPERIMENTS='E1 E2 E3 E4' ESTIMATOR=consensus DEVICE=cuda:0 \
OUTPUT_PARENT=/data5/zhangye/ATLAS/results/e1_e4_ablations/smoke_01 \
bash experiments/run_e1_e4.sh
```

## Full diagnostic runs (recommended first pass)

The 2 datasets × 3 paired ratios × 4 arms = **24 training runs** with one seed.
They run sequentially on `cuda:0` to avoid GPU contention.

```bash
QUICK=0 DATASETS='pbmc bmmc' PAIR_RATIOS='0 20 100' \
SEEDS='1234' EXPERIMENTS='E1 E2 E3 E4' ESTIMATOR=consensus DEVICE=cuda:0 \
OUTPUT_PARENT=/data5/zhangye/ATLAS/results/e1_e4_ablations/full_01 \
bash experiments/run_e1_e4.sh
```

To run **all six paired ratios**:

```bash
QUICK=0 DATASETS='pbmc bmmc' PAIR_RATIOS='0 20 40 60 80 100' \
SEEDS='1234' EXPERIMENTS='E1 E2 E3 E4' ESTIMATOR=consensus DEVICE=cuda:0 \
OUTPUT_PARENT=/data5/zhangye/ATLAS/results/e1_e4_ablations/all_ratios_01 \
bash experiments/run_e1_e4.sh
```

To repeat with 3 seeds, create a **new** `OUTPUT_PARENT` and set
`SEEDS='1234 2022 3407'`. Unlike the legacy runner, this patch keeps each
seed the same across paired ratios; the input cell sets can still differ.

To check the original trainable TimeHead rather than the ensemble, use a
completely separate run with `ESTIMATOR=head`.

## Outputs

Per arm/seed/dataset/ratio:

```
results/e1_e4_ablations/full_01/
  E1/seed_1234/PBMC/pair_000/metrics.json
  E2/seed_1234/PBMC/pair_000/early_stopping_summary.json
  E3/seed_1234/BMMC/pair_100/e1_e4_experiment_protocol.json
  E4/seed_1234/BMMC/pair_100/teacher_refresh_history.json
  ...
  all_e1_e4_runs.csv
  summary_e1_e4.csv
  paired_vs_e1.csv
  paired_vs_e2.csv
  PBMC_e1_e4_metrics.png/pdf
  BMMC_e1_e4_metrics.png/pdf
```

- `all_e1_e4_runs.csv`: one row per run with selected phase/epoch, stop flag,
  cohort size, estimator, metrics, and paths.
- `summary_e1_e4.csv`: mean, std, count by dataset/pair ratio/arm.
- `paired_vs_e1.csv`: metric differences relative to the final-checkpoint E1.
- `paired_vs_e2.csv`: isolates the added effect of E3's freeze and E4's
  fixed-teacher mechanism beyond Phase-III-only checkpoint selection.
- For `better_*_vs_*` columns, positive ALWAYS means improvement; for
  FOSCTTM/RMSE/PED, baseline minus candidate is used.
- Before computing comparisons, the summarizer verifies identical held-out
  report-label partitions, estimator identity, stage sample counts, and
  that no E2/E3/E4 run selected a Phase-I/II checkpoint.

## Limitations / cautions

1. **No automatic resume after interruption**: use a fresh OUTPUT_PARENT or
   manually run missing dataset/arm combinations. To skip completed *datasets*
   set `SKIP_EXISTING=1`; incomplete datasets must be handled separately.
2. E3 freezes the **encoder/fusion/decoder/cross-prediction** modules and uses
   deterministic encoder means during Phase III. Only TimeHead and lag heads
   remain trainable. This intentionally also removes stochastic encoder
   sampling; do not attribute every E3 change solely to gradient freezing.
3. E4 fixes the initial teacher time vector and local teacher confidence, but
   refreshes lag banks from current source-modality embeddings every configured
   `refresh_dynamic_every` epochs. This isolates teacher drift from lag-bank
   staleness, while still permitting representation adaptation.
4. Early stopping uses **monitor stage labels** (supervised model selection),
   and T3 sign is chosen by **POA** on monitor labels, then POAUC is calculated.
   These different weightings can disagree. Inspect per-stage-pair scores too.
5. E1/E2/E3/E4 with `ESTIMATOR=consensus` do not reproduce original
   **Table 2 TimeHead baseline** automatically. For that use `ESTIMATOR=head`
   as a separate, fully controlled sweep.
6. Your current 0%-paired T1 retrieval may be poor and requires a separate
   representation-learning investigation. Do not judge the complete model
   solely on held-out T3 POAUC.
7. **No full PBMC/BMMC training was performed while preparing this package**;
   those raw h5ad inputs and server CUDA are not available in the authoring
   environment. Offline patch/reporter tests passed; real smoke tests must
   run on your server before large jobs.
