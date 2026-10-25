#!/usr/bin/env python3
"""Patch ATLAS PR#5-era trainer/runner in place; back up and verify anchors.

Usage (ATLAS repo root): python scripts/apply_e1_e4_patch.py --root .
This is an opt-in experimental patch, not a git reset or checkout.
"""
from __future__ import annotations
import argparse
import ast
import shutil
from datetime import datetime
from pathlib import Path


def change(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly 1 anchor, found {count}: {old[:130]!r}')
    return text.replace(old, new, 1)


def patch_trainer(s):
    # First, add experiment policies. With policy=final, monitoring is logged but
    # there is NO checkpoint selection and all 80/60/40 epochs are performed.
    s = change(s,
        '    es_enabled = bool(es_cfg.get("enabled", False))\n',
        '''    es_enabled = bool(es_cfg.get("enabled", False))
    selection_policy = str(es_cfg.get("selection_policy", "legacy_all"))
    if es_enabled and selection_policy not in {"legacy_all", "final", "phase3"}:
        raise ValueError("early_stopping.selection_policy must be legacy_all, final or phase3")
    freeze_phase3 = bool(es_cfg.get("freeze_representation_phase3", False))
    fixed_teacher = bool(es_cfg.get("fixed_teacher_phase3", False))
    if (freeze_phase3 or fixed_teacher) and not es_enabled:
        raise ValueError("E3/E4 experiments need the monitor/report label protocol")
    freeze_modules = ("encoders", "fusion", "fusion_refine", "decoders", "cross_predictors")
    frozen_temporal = None
    temporal_refresh_records = []
''', 'experiment settings')
    # In E3 time-head receives deterministic latent representations; the
    # original trainers' VAE sampling would otherwise inject noise even when frozen.
    s = change(s,
        '    for phase, n_epochs in phases:\n        for epoch in range(n_epochs):\n',
        '''    for phase, n_epochs in phases:
        if phase == "phase3" and freeze_phase3:
            for name in freeze_modules:
                for parameter in getattr(model, name).parameters():
                    parameter.requires_grad_(False)
            print("[E3] Phase III frozen: " + ", ".join(freeze_modules))
        for epoch in range(n_epochs):
''', 'phase3 freeze point')
    old_refresh = '''                if dynamic is None or phase3_epoch % max(refresh, 1) == 0:
                    dynamic = build_dynamic_cache(model, data, cfg, device)
                    save_json(
'''
    new_refresh = '''                if dynamic is None or phase3_epoch % max(refresh, 1) == 0:
                    if fixed_teacher and frozen_temporal is not None:
                        # The temporal teacher is fixed at Phase-III entry.
                        # Re-extract current shared embeddings and refresh lag banks
                        # conditioned on the SAME teacher. This isolates teacher
                        # refresh, without freezing the representations/lag features.
                        h_new, shared_new, _ = infer_training_latents(model, data, device)
                        refreshed_banks = build_lag_banks(
                            shared=shared_new, masks=data.masks,
                            time=frozen_temporal.teacher,
                            edges=(("atac", "rna"),),
                            n_bins=int(cfg["lag"]["bins"]),
                        )
                        dynamic = DynamicCache(
                            h=h_new, shared=shared_new,
                            time_teacher=frozen_temporal.teacher,
                            temporal=frozen_temporal, lag_banks=refreshed_banks,
                        )
                    else:
                        dynamic = build_dynamic_cache(model, data, cfg, device)
                        if fixed_teacher and frozen_temporal is None:
                            frozen_temporal = dynamic.temporal
                    temporal_refresh_records.append({
                        "phase3_epoch": phase3_epoch + 1,
                        "teacher_fixed": bool(fixed_teacher),
                        "teacher_mean": float(np.mean(dynamic.time_teacher)),
                        "teacher_std": float(np.std(dynamic.time_teacher)),
                        "teacher_fusion_method": dynamic.temporal.method,
                    })
                    save_json(temporal_refresh_records, output_dir / "teacher_refresh_history.json")
                    save_json(
'''
    s = change(s, old_refresh, new_refresh, 'teacher refresh')
    # Ensure deterministic latent reference in E3 even if model.train() enables
    # dropout and variational sampling in frozen encoders.
    s = change(s,
        '''            model.train()
            accum: Dict[str, float] = {}
''',
        '''            model.train()
            if phase == "phase3" and freeze_phase3:
                for name in freeze_modules:
                    getattr(model, name).eval()
            accum: Dict[str, float] = {}
''', 'frozen module evaluation')
    s = change(s,
        '                out = model(x, masks, lag_banks=lag_banks_t, lag_valid=lag_valid_t)\n',
        '''                out = model(
                    x, masks, lag_banks=lag_banks_t, lag_valid=lag_valid_t,
                    sample=False if phase == "phase3" and freeze_phase3 else None,
                )
''', 'deterministic E3 forward')
    s = change(s,
        '''                improved = score > best_score + es_min_delta
                if improved:
''',
        '''                eligible = (
                    selection_policy == "legacy_all"
                    or (
                        selection_policy == "phase3"
                        and phase == "phase3"
                        and epoch + 1 >= es_min_phase3
                    )
                )
                improved = eligible and score > best_score + es_min_delta
                if improved:
''', 'eligible checkpoint')
    s = change(s,
        '''                if phase == "phase3":
                    bad_phase3_checks = 0 if improved else bad_phase3_checks + 1
''',
        '''                if phase == "phase3" and eligible:
                    bad_phase3_checks = 0 if improved else bad_phase3_checks + 1
''', 'patience eligible only')
    s = change(s,
        '''                    "best_so_far": float(best_score),
                    "improved": bool(improved),
''',
        '''                    "best_so_far": float(best_score) if np.isfinite(best_score) else None,
                    "eligible_for_selection": bool(eligible),
                    "improved": bool(improved),
''', 'monitor history fields')
    # End block: E1 does not restore anything. E2-E4 restore phase3 selected.
    s = change(s,
        '''    if es_enabled:
        if best_info is None:
            raise RuntimeError("No valid checkpoint-monitor evaluation was recorded")
        best_ckpt = torch.load(output_dir / "best_t3.pt", map_location=device)
''',
        '''    if es_enabled and selection_policy != "final":
        if best_info is None:
            raise RuntimeError("No valid checkpoint available for selection")
        if selection_policy == "phase3" and best_info["phase"] != "phase3":
            raise AssertionError("An earlier-phase model cannot be selected")
        best_ckpt = torch.load(output_dir / "best_t3.pt", map_location=device)
''', 'restore best condition')
    s = change(s,
        '''    dynamic = build_dynamic_cache(model, data, cfg, device)
    torch.save(
''',
        '''    if es_enabled and selection_policy == "final":
        save_json({
            "selection_policy": "final",
            "selected_phase": "phase3",
            "selected_epoch": int(phases[-1][1]),
            "early_stopped_phase3": False,
            "checkpoint_loaded_for_final_evaluation": "final_weights",
        }, output_dir / "early_stopping_summary.json")
    if es_enabled:
        save_json({
            "selection_policy": selection_policy,
            "freeze_representation_phase3": freeze_phase3,
            "fixed_teacher_phase3": fixed_teacher,
        }, output_dir / "e1_e4_experiment_protocol.json")
    dynamic = build_dynamic_cache(model, data, cfg, device)
    torch.save(
''', 'end summary')
    # Best checkpoint metadata is already stored in best_info.
    return s


def patch_runner(s):
    s = change(s,
        '        seed = int(model_cfg.get("seed", 1234)) + int(pair_ratio)\n',
        '''        # Experimental protocol: keep seed identical across ratios.
        # Default false preserves historical behavior outside E1-E4.
        seed_offset = 0 if model_cfg.get("fixed_seed_across_ratios", False) else int(pair_ratio)
        seed = int(model_cfg.get("seed", 1234)) + seed_offset
''', 'seed independence')
    return s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    trainer=root/'src/atlas/trainer.py'
    runner=root/'experiments/run_table2.py'
    for p in (trainer,runner):
        if not p.is_file():
            raise SystemExit(f'Missing required project file: {p}')
    a=trainer.read_text(encoding='utf-8')
    b=runner.read_text(encoding='utf-8')
    if 'selection_policy = str(es_cfg.get("selection_policy"' in a:
        if 'fixed_seed_across_ratios' in b:
            print('Already patched: no changes needed.')
            return
        raise RuntimeError('Partial patch detected. Restore from backup and retry.')
    ca=patch_trainer(a)
    cb=patch_runner(b)
    for p, text in ((trainer,ca),(runner,cb)):
        ast.parse(text,filename=str(p))
    if args.check_only:
        print('All patch anchors and Python syntax verified; no files changed.')
        return
    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    backups=[]
    for p in (trainer,runner):
        dest=p.with_name(f'{p.name}.pre_e1e4_{ts}.bak')
        shutil.copy2(p,dest)
        backups.append(dest)
    trainer.write_text(ca,encoding='utf-8')
    runner.write_text(cb,encoding='utf-8')
    print('Patched:',trainer)
    print('Patched:',runner)
    print('Backups:')
    for x in backups: print(' -',x)
    print('Syntax validated; run unit tests and smoke experiment before full training.')

if __name__=='__main__':
    main()
