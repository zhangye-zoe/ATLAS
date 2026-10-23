# ATLAS vs ATLAS_Debug: T3/T4 regression audit

This document reviews the legacy scripts in [ATLAS_Debug](https://github.com/zhangye-zoe/ATLAS_Debug) against the refactored [ATLAS](https://github.com/zhangye-zoe/ATLAS). Findings are **source-code observations**, not claims that original checkpoints have been recovered or that published metrics have been reproduced.

## Executive findings

1. **T3 estimator changed.** Legacy CORAL exposes trained head, DPT, graph-geodesic automatic root, and label-guided graph root. Different runner defaults used **different estimators**. Refactored ATLAS previously evaluated only TimeHead, even though its training teacher combines graph/diffusion/attribution proxies. This PR makes the estimator explicit, preserves the default, and adds post-hoc comparison.
2. **T3 cohorts and input protocol differ.** Legacy scripts can evaluate `val_all`, `val_atac`, `val_paired`, and fixed cell IDs. Current ATLAS evaluates the ATAC-query validation cohort and defaults to both RNA+ATAC (`paired_fused`) for T3. Compare exact barcode sets, modalities, and stage maps.
3. **Some old T3 experiments used supervision that is absent from the new code.** V2 optionally trains TimeHead using stage-aware pairwise ranking. The V3 fair-T3 variant disables time rank, prediction and several dynamic losses by default and chooses `graph_auto` to avoid evaluating an untrained head. These are **different experimental protocols**, not equivalent model revisions. Stage labels used for ranking must be restricted to **training** cells and disclosed; using validation reference labels for training would invalidate a fully unsupervised T3 claim.
4. **The loss/model/training implementations changed.** The legacy dynamic scripts include a velocity head, kNN topo/time/forward/dynamic contrastive losses, batch-adversarial regularization and phase-specific optimizer/freezing. Refactored ATLAS instead uses a reliability-weighted teacher, target-profile OT for unpaired prediction/lag, separate source-specific lag banks, an updated VAE, and a single optimizer throughout phases. Consequently, losses and checkpoints are not numerically or architecturally identical.
5. **T4 was not fully ported.** Refactored `future_state.py` evaluates supplied probability scores only; no trained velocity head, transition operator or HSPC future-state benchmark runner is present. The legacy conditional T4 scorer uses true lineage labels to designate terminal cells and then tests on the same labeled cells. This is **label-informed / potentially leaky** and must not be ported as a trustworthy held-out future-prediction score.

## Legacy script inventory and intent

| Legacy file | Role / distinctive protocol | Priority for comparison |
| --- | --- | --- |
| `coral_alignment_time_batchfix.py` | Dynamic trainer; batch discriminator/GRL, graph-root/DPT/head alternatives and conditional velocity-based T4 | High: full dynamic reference |
| `coral_optimized_batchfix_bioconserve.py` | Above with explicit scMRDR-like biological isometry constraint | High: biological regularization |
| `coral_timehead_fast.py` | Faster head-focused dynamic implementation, but T3 CLI still defaults to graph_root | High: distinguish trained head from reported graph score |
| `coral_ratio_loop_T1T2T3_bioconserve.py` | Ratio runner, default T3 estimator `dpt`, explicit stage orientation and cell-set filters | High: historical eval protocol |
| `coral_ratio_loop_T1T2T3_timehead_fast.py` | Ratio runner with `head` default, training time-source control and candidate diagnostics | High: head-to-head comparison |
| `coral_ratio_loop_alignment_time.py` | Ratio runner with alignment-time variants, default DPT | Medium: evaluation ablations |
| `scripts_V2/coral_time_rank_smooth.py` | TimeHead trained via stage-based `L_rank` (when enabled) and latent kNN time smoothness | High: check supervision provenance |
| `scripts_V2/coral_ratio_loop_time_rank_smooth.py` | V2 runner; stage CSV can enter **training** stage arrays, monitors T3 on validation during Phase III | High: potential validation-selection bias |
| `scripts_V3/coral_fair_t3_nopred_norank.py` | Conservative default: prediction and rank disabled, no lag/topo/time dynamics by default | High: fair protocol reference |
| `scripts_V3/coral_ratio_loop_fair_t3.py` | Fair-T3 runner defaults to `graph_auto`; script comments warn TimeHead may be untrained | High: immediate suspect for T3 discrepancy |
| `storage/coral_v1_graphtime.py` | Early graph-root/DPT-enabled dynamic CORAL model | High: graph estimator provenance |
| `storage/coral_dynamic_lag_unsup.py` | Earlier unsupervised lag dynamics, with transition consistency and time regularization | Medium: dynamic loss evolution |
| `storage/coral_static_ratio_loop.py` | Static T1/T2 training/evaluation; Phase III off unless requested | Medium: isolate representation quality |
| `storage/coral_ratio_loop_T1T2T3_direct.py` | Direct T3 runner on saved/inferred time | Medium: historical T3 cohort |
| `storage/coral_ratio_loop_T1T2T3_graphtime.py` | Graph-time ratio runner; defaults to DPT | High: estimator mismatch |
| `storage/coral_ratio_loop_T1T2T3T4_dynamic.py` | Conditional velocity/transition-based T4 runner; **default Phase III epochs=0** unless enabled | High for T4 provenance; caution on validity |
| `reeval_all_checkpoints_T1T2T3.py` | Re-evaluates historical checkpoints with alternative T1/T2/T3 protocols | High: identify the specific successful checkpoint |
| `storage/reevaluate_coral_checkpoints_graphtime.py` | Graph-root re-evaluation across checkpoints | High: identify which estimator produced reported T3 |
| `storage/check_t3_time_columns.py` | Validates saved T3 time columns, orientation and inter-stage metrics | Medium |
| `run_scib_metrics_full.py`, `run_scib_metrics_full_compatible.py` | scIB integration/batch metric calculation rather than T3 training | Low for T3 regression |
| `run_pbmc_coral_and_scib.sh` | Launch script chaining CORAL and scIB jobs | Inspect actual historical flags to identify active variant |
| `scripts/Monucle/prepare_monocle3_input_from_coral.py` | Exports trajectory input for Monocle3; external pseudotime follow-up | Secondary |
| `data.ipynb`, `scib_baselines_bmmc_template.json` | Interactive exploration / integration baseline configuration | Secondary |

**Caution:** File naming alone does not prove which version yielded the best paper scores. The decisive evidence is the archived command line / `train_config.json`, output `T3_time_candidate_metrics.csv`, checkpoint path and T3 evaluation cohort.

## Specific differences: historical vs current T3

- **Estimator:** legacy T3 chooses among `CORAL_time_head`, `CORAL_time_dpt`, `CORAL_time_graph_auto`, and `CORAL_time_graph_root`. Current `src/atlas/trainer.py` previously used `out["time"]` only; this patch adds explicit `evaluation.t3_time_mode` selection, default `head`.
- **Root:** legacy `graph_auto` chooses a diffusion-map root, followed by weighted kNN graph shortest paths; `graph_root` may use an external early-stage label centroid. Current teacher chooses a peripheral root and blends candidates with reliability weights. These are *not* the same estimator.
- **Time objective:** V2 optionally uses true ordinal training-stage labels to form a pairwise margin rank loss. Current teacher loss matches model time to a reference computed from unlabeled training embeddings. The V3 fair default explicitly disables label-ranking and uses graph time.
- **Graph inference:** graph-root/DPT applied after computing validation latent embeddings; current model head is parametric and can assign times independently. Their results can diverge substantially even with identical latent representations.
- **Validation cohort:** comparing changing subsets across pairing ratios introduces a confound. Use the intersection of exact saved validation cell IDs for a diagnostic; for final experiments, preprocess all ratios with a fixed validation query set.
- **Orientation:** both old and current scripts can reference-orient time based on stage information. This is evaluation-stage reference use and should be disclosed, especially when reporting an unsupervised trajectory model.
- **Epochs & optimizer:** earlier full dynamic scripts often configure 80/60/80 and select best checkpoints per phase; current YAML config is 80/60/40, with persistent AdamW, and saves the final state. This is a meaningful optimization change even before considering loss functions.
- **Model:** old conditional velocity and batch-adversarial modules are absent from refactored current model. Do not load old weights as if state dictionaries were identical.

## Run the low-cost T3 diagnostic FIRST (no new training)

Existing refactored runs store both `time` and `h_fused` in `results/table2/PBMC/pair_XXX/predictions.npz`.

~~~bash
cd /data5/zhangye/ATLAS

# Uses your external reference labels and existing NPZ embeddings.
python scripts/compare_legacy_t3.py --dataset pbmc \
  --annotations data/external/annotations/pbmc_azimuth_annotations.csv \
  --cell-set intersection \
  --modes head graph_peripheral graph_auto dpt

cat results/table2/PBMC/T3_legacy_comparison.csv
~~~

**Dependencies:** `graph_auto` and `dpt` require Scanpy. Without Scanpy run `--modes head graph_peripheral`. The script records omitted modes/errors rather than guessing their scores. It does **not** change `metrics.json`, change checkpoints or retrain any model.

If `graph_auto` improves substantially over `head` **on the same cells**, this supports an estimator/graph-root mismatch, not necessarily improved representation learning. The full T3 model should then be selected **without choosing its estimator using test-stage labels**; choose the estimator from method design or a dedicated validation protocol and report all diagnostic candidates.

Optional subsequent experiment: specify `evaluation.t3_time_mode: graph_auto` in `configs/model_table2.yaml` for a **future** complete training/evaluation run. Default remains `head` for backward compatibility. Never label-guided `graph_root` as unsupervised.

## T4 gap and scientifically safe restoration plan

Historical T4 comprises (a) learned velocity field, (b) neighbor transition scores based on direction/time/geometry, (c) terminal-fate aggregation and F1/AUROC. The present model has only lag attention and the metric function `src/atlas/future_state.py`.

**Important invalidity risk:** in the old `compute_t4_metrics`, the scorer uses `y_true_str` to select the top-time terminal cells *per true class*, builds class probabilities from exactly these labeled cells, then evaluates F1/AUROC on the same set. This is not a genuinely held-out future-state prediction protocol.

To restore T4 rigorously, we need:

1. Confirm target definition and experimentally supported HSPC lineage/future labels, along with train/validation/test cell-ID splits.
2. Add the optional velocity head and directed transition operator to `model.py`, reusing only pre-training input and **no target test labels**.
3. Define terminal prototypes from **training/reference cells only** (or an independent reference), freeze their labels, and evaluate disjoint held-out query cells.
4. Record trajectory horizons, future labels, AUROC/F1 class aggregation, seeds and baseline comparisons. Do not report score comparability to the old leaky procedure as an exact replication.

This PR intentionally **does not invent T4 scores** or pretend T4 is restored when no validated HSPC input/target pipeline is in the refactored repository.

## Validation checklist

- Compare the same dataset/feature set and identical validation cells across variants.
- Check old `train_config.json` / launch command for `--run-phase3`, `--t3-time-mode`, `--root-labels`, `--label-csv`, `--enable-time-rank-loss`, and `--t3-cell-set`.
- Run saved-embedding T3 diagnostics; determine whether the head vs graph estimator accounts for most of the reported difference.
- Reproduce with at least three independent seeds and report means/deviations before asserting the regression is fixed.
- Distinguish label-free, reference-oriented, and training-stage-supervised T3 results in output tables.
