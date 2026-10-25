"""Offline smoke checks of the patcher's anchor replacements (no ATLAS datasets needed).
Run: python scripts/test_e1_e4_patch.py
"""
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
from apply_e1_e4_patch import patch_trainer, patch_runner

# Minimal syntactically valid trainer source preserving all expected anchors.
TRAINER = '''def train_atlas(model, data, cfg, device, output_dir, monitor_data=None):
    es_cfg = cfg.get("early_stopping") or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_min_delta = 0.002
    best_score = float("-inf")
    es_min_phase3 = 10
    es_patience = 4
    bad_phase3_checks = 0
    stop_phase3 = False
    best_info = None
    phase3_epoch = 0
    dynamic = None
    phases = [("phase1",2),("phase2",1),("phase3",1)]
    for phase, n_epochs in phases:
        for epoch in range(n_epochs):
            if phase == "phase3":
                refresh = 5
                if dynamic is None or phase3_epoch % max(refresh, 1) == 0:
                    dynamic = build_dynamic_cache(model, data, cfg, device)
                    save_json(
                        {"teacher": "ok"}, output_dir / "temporal_teacher.json"
                    )
                phase3_epoch += 1
            model.train()
            accum: Dict[str, float] = {}
            n_batches = 0
            for idx in [0]:
                out = model(x, masks, lag_banks=lag_banks_t, lag_valid=lag_valid_t)
            if es_enabled:
                improved = score > best_score + es_min_delta
                if improved:
                    best_score = score
                    best_info = {"phase": phase, "phase_epoch": epoch + 1}
                if phase == "phase3":
                    bad_phase3_checks = 0 if improved else bad_phase3_checks + 1
                    if epoch + 1 >= es_min_phase3 and bad_phase3_checks >= es_patience:
                        stop_phase3 = True
                monitoring_history.append({
                    "best_so_far": float(best_score),
                    "improved": bool(improved),
                })
    if es_enabled:
        if best_info is None:
            raise RuntimeError("No valid checkpoint-monitor evaluation was recorded")
        best_ckpt = torch.load(output_dir / "best_t3.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        save_json({
            "best": best_info, "early_stopped_phase3": bool(stop_phase3),
            "patience": es_patience
        }, output_dir / "early_stopping_summary.json")
    dynamic = build_dynamic_cache(model, data, cfg, device)
    torch.save(
        {"model_state":model.state_dict()}, output_dir / "model.pt"
    )
'''

RUNNER = '''def main():
    for pair_ratio in [0, 20, 100]:
        seed = int(model_cfg.get("seed", 1234)) + int(pair_ratio)
        seed_everything(seed)
'''


def main():
    transformed=patch_trainer(TRAINER)
    rr=patch_runner(RUNNER)
    compile(transformed, 'trainer.py', 'exec')
    compile(rr,'run_table2.py','exec')
    assert 'eligible_for_selection' in transformed
    assert 'frozen_temporal' in transformed
    assert 'selected_phase' in transformed
    assert 'fixed_seed_across_ratios' in rr
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        (root/'src/atlas').mkdir(parents=True)
        (root/'experiments').mkdir(parents=True)
        (root/'src/atlas/trainer.py').write_text(TRAINER)
        (root/'experiments/run_table2.py').write_text(RUNNER)
        script=Path(__file__).with_name('apply_e1_e4_patch.py')
        cmd=[sys.executable,str(script),'--root',str(root)]
        subprocess.run(cmd+['--check-only'],check=True,capture_output=True,text=True)
        subprocess.run(cmd,check=True,capture_output=True,text=True)
        subprocess.run(cmd,check=True,capture_output=True,text=True)
        assert len(list((root/'src/atlas').glob('trainer.py.pre_e1e4_*.bak')))==1
    print('PASS: patch anchors, syntax, backups, idempotency')

if __name__=='__main__': main()
