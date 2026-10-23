from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MultiOmicsDataset, Table2Data, collate_multiomics
from .lag import LagBank, build_lag_banks
from .losses import (
    biological_structure_loss,
    cross_prediction_objective,
    lag_objective,
    multilevel_alignment_loss,
    reconstruction_and_kl,
    time_distillation_loss,
)
from .metrics import evaluate_t1_coral_style, evaluate_t2, evaluate_t3, labels_to_stages
from .model import ATLAS
from .temporal import TemporalTeacher, build_temporal_teacher
from .utils import save_json


@dataclass
class DynamicCache:
    h: np.ndarray
    shared: Dict[str, np.ndarray]
    time_teacher: np.ndarray
    temporal: TemporalTeacher
    lag_banks: Dict[str, LagBank]


def _to_device_dict(d: Mapping[str, torch.Tensor], device: torch.device):
    return {k: v.to(device, non_blocking=True) for k, v in d.items()}


def _batch_tensor_arrays(data: Table2Data, indices: np.ndarray, device: torch.device):
    x = {m: torch.as_tensor(data.modalities[m][indices], device=device) for m in data.modalities}
    masks = {m: torch.as_tensor(data.masks[m][indices], device=device) for m in data.masks}
    return x, masks


@torch.no_grad()
def infer_training_latents(
    model: ATLAS,
    data: Table2Data,
    device: torch.device,
    chunk_size: int = 1024,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    model.eval()
    hs = []
    ts = []
    shared = {m: [] for m in data.modalities}
    n = len(data.cell_ids)
    for lo in range(0, n, chunk_size):
        idx = np.arange(lo, min(n, lo + chunk_size))
        x, masks = _batch_tensor_arrays(data, idx, device)
        out = model(x, masks, sample=False)
        hs.append(out["h"].cpu().numpy())
        ts.append(out["time"].cpu().numpy())
        for m in shared:
            # Use posterior mean to make global teacher/banks deterministic.
            shared[m].append(out["enc"][m]["mu_shared"].cpu().numpy())
    return (
        np.concatenate(hs, axis=0),
        {m: np.concatenate(v, axis=0) for m, v in shared.items()},
        np.concatenate(ts, axis=0),
    )


def build_dynamic_cache(model: ATLAS, data: Table2Data, cfg: Dict, device: torch.device) -> DynamicCache:
    h, shared, _ = infer_training_latents(model, data, device)
    tc = cfg["temporal"]
    temporal = build_temporal_teacher(
        h=h,
        shared=shared,
        masks=data.masks,
        k_neighbors=int(tc["k_neighbors"]),
        tau=float(tc["tau"]),
        lambda_sm=float(tc["lambda_sm"]),
        lambda_mv=float(tc["lambda_mv"]),
        lambda_sp=float(tc["lambda_sp"]),
        target_std=float(tc["target_std"]),
    )
    banks = build_lag_banks(
        shared=shared,
        masks=data.masks,
        time=temporal.teacher,
        edges=(("atac", "rna"),),
        n_bins=int(cfg["lag"]["bins"]),
    )
    return DynamicCache(h=h, shared=shared, time_teacher=temporal.teacher, temporal=temporal, lag_banks=banks)


def _attention_smoothness(alpha: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    if alpha.shape[0] < 3:
        return alpha.new_zeros(())
    with torch.no_grad():
        d = torch.cdist(h, h)
        d.fill_diagonal_(float("inf"))
        nn = d.argmin(dim=1)
    return torch.mean(torch.abs(alpha - alpha[nn]))


def _local_teacher_pairs(teacher_time: torch.Tensor, max_pairs: int = 1024) -> torch.Tensor:
    n = teacher_time.numel()
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=teacher_time.device)
    p = min(max_pairs, n * 4)
    i = torch.randint(0, n, (p,), device=teacher_time.device)
    j = torch.randint(0, n, (p,), device=teacher_time.device)
    keep = i != j
    return torch.stack([i[keep], j[keep]], dim=1)


def train_atlas(
    model: ATLAS,
    data: Table2Data,
    cfg: Dict,
    device: torch.device,
    output_dir: Path,
) -> DynamicCache:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    align_cfg = cfg["alignment"]
    lag_cfg = cfg["lag"]
    temporal_cfg = cfg["temporal"]

    ds = MultiOmicsDataset(data.modalities, data.masks)
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_multiomics,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    model.to(device)

    phases = [
        ("phase1", int(train_cfg["phase1_epochs"])),
        ("phase2", int(train_cfg["phase2_epochs"])),
        ("phase3", int(train_cfg["phase3_epochs"])),
    ]
    history = []
    dynamic: Optional[DynamicCache] = None
    phase3_epoch = 0

    for phase, n_epochs in phases:
        for epoch in range(n_epochs):
            if phase == "phase3":
                refresh = int(train_cfg.get("refresh_dynamic_every", 5))
                if dynamic is None or phase3_epoch % max(refresh, 1) == 0:
                    dynamic = build_dynamic_cache(model, data, cfg, device)
                    save_json(
                        {
                            "weights": dynamic.temporal.weights,
                            "risks": dynamic.temporal.risks,
                        },
                        output_dir / "temporal_teacher.json",
                    )
                phase3_epoch += 1

            model.train()
            accum: Dict[str, float] = {}
            n_batches = 0
            for idx, x, masks in loader:
                idx_np = idx.numpy()
                x = _to_device_dict(x, device)
                masks = _to_device_dict(masks, device)

                lag_banks_t = None
                lag_valid_t = None
                if phase == "phase3" and dynamic is not None:
                    lag_banks_t = {
                        key: torch.as_tensor(bank.bank[idx_np], device=device)
                        for key, bank in dynamic.lag_banks.items()
                    }
                    lag_valid_t = {
                        key: torch.as_tensor(bank.valid[idx_np], device=device)
                        for key, bank in dynamic.lag_banks.items()
                    }

                out = model(x, masks, lag_banks=lag_banks_t, lag_valid=lag_valid_t)
                rec, kl = reconstruction_and_kl(out, x, masks)
                mmd, ot, con = multilevel_alignment_loss(
                    out,
                    masks,
                    epsilon=float(align_cfg["ot_epsilon"]),
                    n_iter=int(align_cfg["ot_iterations"]),
                    max_cells=int(align_cfg["max_ot_cells_per_modality"]),
                )
                # Contrastive belongs to LVAE in the manuscript and is only same-cell paired.
                loss = (
                    float(loss_cfg["lambda_rec"]) * rec
                    + float(loss_cfg["lambda_con"]) * con
                    + float(loss_cfg["lambda_kl"]) * kl
                )

                components = {"rec": rec, "con": con, "kl": kl, "mmd": mmd, "ot": ot}
                if phase in {"phase2", "phase3"}:
                    bio = biological_structure_loss(
                        out, x, masks, k=int(align_cfg["bio_k"])
                    )
                    loss = (
                        loss
                        + float(loss_cfg["lambda_mmd"]) * mmd
                        + float(loss_cfg["lambda_ot"]) * ot
                        + float(loss_cfg["lambda_bio"]) * bio
                    )
                    components["bio"] = bio
                    pred_teacher = None
                    if phase == "phase3" and dynamic is not None:
                        pred_teacher = torch.as_tensor(
                            dynamic.time_teacher[idx_np], device=device, dtype=out["time"].dtype
                        )
                    lpred, pred_parts = cross_prediction_objective(
                        out,
                        x,
                        masks,
                        teacher_time=pred_teacher,
                        source="atac",
                        target="rna",
                        lambda_unpaired=float(loss_cfg.get("lambda_unpaired_pred", 1.0)),
                        alpha_time=float(lag_cfg["alpha_time"]),
                        alpha_latent=float(lag_cfg["alpha_latent"]),
                        temporal_window=float(lag_cfg["temporal_window"]),
                        local_k=int(lag_cfg["local_k"]),
                        epsilon=float(lag_cfg["ot_epsilon"]),
                        n_iter=int(lag_cfg["ot_iterations"]),
                    )
                    loss = loss + float(loss_cfg.get("lambda_pred", 1.0)) * lpred
                    components.update({
                        "pred": lpred,
                        "pred_paired": pred_parts["paired"],
                        "pred_unpaired": pred_parts["unpaired"],
                    })

                if phase == "phase3" and dynamic is not None:
                    teacher_t = torch.as_tensor(
                        dynamic.time_teacher[idx_np], device=device, dtype=out["time"].dtype
                    )
                    pairs = _local_teacher_pairs(
                        teacher_t, int(temporal_cfg.get("teacher_pair_samples", 1024))
                    )
                    ltime, ltime_reg = time_distillation_loss(
                        out["time"],
                        teacher_t,
                        out["h"],
                        teacher_pairs=pairs,
                        target_std=float(temporal_cfg["target_std"]),
                    )
                    llag, lag_parts = lag_objective(
                        out,
                        x,
                        masks,
                        teacher_t,
                        source="atac",
                        target="rna",
                        lambda_unpaired=float(loss_cfg["lambda_unpaired_lag"]),
                        alpha_time=float(lag_cfg["alpha_time"]),
                        alpha_latent=float(lag_cfg["alpha_latent"]),
                        temporal_window=float(lag_cfg["temporal_window"]),
                        local_k=int(lag_cfg["local_k"]),
                        epsilon=float(lag_cfg["ot_epsilon"]),
                        n_iter=int(lag_cfg["ot_iterations"]),
                    )
                    smooth = out["h"].new_zeros(())
                    if "atac__to__rna" in out["lag"]:
                        smooth = _attention_smoothness(
                            out["lag"]["atac__to__rna"]["alpha"], out["h"]
                        )
                    # Lag warm-up prevents unstable transport from dominating the newly activated head.
                    lag_scale = min(
                        1.0,
                        phase3_epoch / max(int(train_cfg.get("lag_warmup_epochs", 5)), 1),
                    )
                    loss = (
                        loss
                        + float(loss_cfg["lambda_time"])
                        * (ltime + float(loss_cfg["lambda_time_reg"]) * ltime_reg)
                        + lag_scale
                        * float(loss_cfg["lambda_lag"])
                        * (llag + float(loss_cfg["lambda_lag_smooth"]) * smooth)
                    )
                    components.update(
                        {
                            "time": ltime,
                            "time_reg": ltime_reg,
                            "lag": llag,
                            "lag_paired": lag_parts["paired"],
                            "lag_unpaired": lag_parts["unpaired"],
                            "lag_smooth": smooth,
                        }
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                optimizer.step()

                components["total"] = loss
                for k, v in components.items():
                    accum[k] = accum.get(k, 0.0) + float(v.detach().cpu())
                n_batches += 1

            row = {"phase": phase, "epoch": epoch + 1}
            row.update({k: v / max(n_batches, 1) for k, v in accum.items()})
            history.append(row)
            print(
                f"[{data.dataset_name} pair={data.paired_ratio:03d}] {phase} "
                f"{epoch + 1:03d}/{n_epochs:03d} loss={row.get('total', float('nan')):.4f}"
            )

    dynamic = build_dynamic_cache(model, data, cfg, device)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dims": model.input_dims,
            "features": data.features,
            "paired_ratio": data.paired_ratio,
            "dataset": data.dataset_name,
        },
        output_dir / "model.pt",
    )
    save_json(history, output_dir / "training_history.json")
    return dynamic


@torch.no_grad()
def evaluate_table2(
    model: ATLAS,
    data: Table2Data,
    dynamic: DynamicCache,
    cfg: Dict,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """Evaluate T1/T2/T3 using the historical CORAL-compatible Table-2 protocol.

    T1 and T2 deliberately share the same strict ATAC-only prediction path:
      ATAC -> predicted RNA; true RNA is held out until post-hoc evaluation.
    T1 FOSCTTM/PED are then computed by joint PCA of [true RNA; predicted RNA],
    exactly as in the retained CORAL ratio-loop script.
    """
    model.eval()
    val = data.val_modalities
    n = len(data.val_cell_ids)
    eval_cfg = cfg["evaluation"]

    # ------------------------------------------------------------------ T2 prediction first
    # Strict source-only path: no RNA tensor is accepted by this API.
    x_atac = torch.as_tensor(val["atac"], device=device)
    pred_rna_std = model.predict_from_source("atac", "rna", x_atac).cpu().numpy()
    pred_rna_raw = data.standardizers["rna"].inverse(pred_rna_std)
    true_rna_raw = data.val_modalities_raw["rna"]
    t2 = evaluate_t2(pred_rna_raw, true_rna_raw)

    # ------------------------------------------------------------------ T1: EXACT original CORAL/scMRDR-style protocol
    # NOT latent ATAC-vs-RNA distance.  Both groups are in RNA feature space,
    # followed by one joint PCA and cross-distance evaluation.
    t1, t1_true_pca, t1_pred_pca = evaluate_t1_coral_style(
        pred_mat=pred_rna_raw,
        true_mat=true_rna_raw,
        seed=int(cfg.get("seed", 1234)),
        max_pcs=int(eval_cfg.get("t1_pca_components", 30)),
    )

    # Save the same diagnostic tables as the historical CORAL evaluator.
    import pandas as pd
    pd.DataFrame(
        {
            "metric": ["FOSCTTM", "PED", "Top1_ACC", "Top5_ACC", "Top10_ACC"],
            "value": [
                t1["FOSCTTM"], t1["PED"], t1["Top1_ACC"],
                t1["Top5_ACC"], t1["Top10_ACC"],
            ],
        }
    ).to_csv(output_dir / "T1_metrics_scMRDR_style.csv", index=False)
    pca_all = np.vstack([t1_true_pca, t1_pred_pca])
    n_pc_save = min(5, pca_all.shape[1])
    pca_df = pd.DataFrame(
        pca_all[:, :n_pc_save],
        columns=[f"PC{i+1}" for i in range(n_pc_save)],
    )
    pca_df["group"] = ["True_RNA"] * n + ["Pred_from_ATAC"] * n
    pca_df["cell"] = (
        [f"true_{c}" for c in data.val_cell_ids.astype(str)]
        + [f"pred_{c}" for c in data.val_cell_ids.astype(str)]
    )
    pca_df.to_csv(output_dir / "pca_true_pred_coords.csv", index=False)
    pd.DataFrame(
        {
            "metric": ["pearson_cell_mean", "pearson_gene_mean", "rmse"],
            "value": [t2["Pearson_cell"], t2["Pearson_gene"], t2["RMSE"]],
        }
    ).to_csv(output_dir / "T2_metrics_scMRDR_style.csv", index=False)

    # Keep latent embeddings only as diagnostics; they are NOT used for T1 metrics.
    za = model.encode_modality_only("atac", x_atac).cpu().numpy()
    x_rna = torch.as_tensor(val["rna"], device=device)
    zr = model.encode_modality_only("rna", x_rna).cpu().numpy()

    # Source-only tensors reused if T3 is configured as ATAC-only.
    x_src = {
        "atac": x_atac,
        "rna": torch.zeros((n, val["rna"].shape[1]), device=device),
    }
    m_src = {
        "atac": torch.ones(n, device=device),
        "rna": torch.zeros(n, device=device),
    }

    # ------------------------------------------------------------------ T3
    mode = str(eval_cfg.get("t3_input_mode", "paired_fused"))
    if mode == "paired_fused":
        x_t3 = {"atac": x_atac, "rna": x_rna}
        m_t3 = {"atac": torch.ones(n, device=device), "rna": torch.ones(n, device=device)}
    elif mode == "atac_only":
        x_t3, m_t3 = x_src, m_src
    else:
        raise ValueError(f"Unknown evaluation.t3_input_mode={mode}")
    t3_out = model(x_t3, m_t3, sample=False)
    t3_time_head = t3_out["time"].cpu().numpy()
    h_fused = t3_out["h"].cpu().numpy()

    # Historical CORAL sometimes reported DPT/graph-geodesic pseudotime
    # instead of TimeHead. Keep TimeHead as the default to avoid silently
    # changing existing Table-2 numbers. Select explicitly in evaluation YAML.
    t3_estimator = str(eval_cfg.get("t3_time_mode", "head"))
    if t3_estimator == "head":
        t3_time = t3_time_head
        t3_estimator_source = "trained TimeHead"
    else:
        from .legacy_t3 import get_t3_candidates
        if t3_estimator == "graph_root":
            raise ValueError(
                "T3 graph_root uses reference labels to choose a root; "
                "for label-free Table-2 use graph_auto or graph_peripheral. "
                "Run scripts/compare_legacy_t3.py for labeled-root diagnostics."
            )
        candidates, sources = get_t3_candidates(
            h_fused,
            t3_time_head,
            modes=[t3_estimator],
            graph_k=int(eval_cfg.get("t3_graph_k", 30)),
        )
        t3_time = candidates[t3_estimator]
        t3_estimator_source = sources[t3_estimator]

    if data.labels is None:
        t3 = {
            "POA": None,
            "POAUC": None,
            "warning": "No validation cell-type labels found. Configure external_labels.csv or pass --label-csv.",
        }
    else:
        stages = labels_to_stages(data.labels, data.stage_map)
        valid = np.isfinite(stages)
        if valid.sum() < 2:
            t3 = {
                "POA": None,
                "POAUC": None,
                "warning": "Labels were loaded but did not match stage_map; inspect label naming.",
            }
        else:
            # Save the exact GT stage labels used by POA/POAUC.
            import pandas as pd
            pd.DataFrame({
                "pred_time": t3_time[valid],
                "stage": stages[valid],
                "label": np.asarray(data.labels)[valid],
            }).to_csv(output_dir / "T3_stage_reference.csv", index=False)

            t3 = evaluate_t3(
                t3_time[valid],
                stages[valid],
                orient=bool(eval_cfg.get("orient_time_for_metric", True)),
            )
            t3["n_eval"] = int(valid.sum())
            t3["reference"] = "external stage_map labels (0-3), not continuous biological time"

    metrics = {
        "dataset": data.dataset_name,
        "paired_ratio": data.paired_ratio,
        "split_label": data.split_dir.name,
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "protocol": {
            "T1": (
                "EXACT CORAL/scMRDR-style: ATAC-only predicted RNA vs true RNA; "
                "joint PCA; cross-distance FOSCTTM/PED"
            ),
            "T1_fos_denominator": "N targets per query, strict distance < true-pair distance (historical CORAL implementation)",
            "T2": "ATAC-only model input; true RNA used after inference for metrics only",
            "T3_input_mode": mode,
            "T3_time_estimator": t3_estimator,
            "T3_time_estimator_source": t3_estimator_source,
            "T3_reference_orientation_only_at_evaluation": bool(
                eval_cfg.get("orient_time_for_metric", True)
            ),
        },
    }
    save_json(metrics, output_dir / "metrics.json")

    arrays = dict(
        cell_ids=data.val_cell_ids.astype(str),
        z_atac=za,
        z_rna=zr,
        h_fused=h_fused,
        pred_rna=pred_rna_raw,
        true_rna=true_rna_raw,
        t1_true_pca=t1_true_pca,
        t1_pred_pca=t1_pred_pca,
        time=t3_time,
        time_head=t3_time_head,
    )
    if data.labels is not None:
        arrays["labels"] = np.asarray(data.labels, dtype=str)
    np.savez_compressed(output_dir / "predictions.npz", **arrays)
    return metrics
