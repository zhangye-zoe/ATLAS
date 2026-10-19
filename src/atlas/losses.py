from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return _zero_like(pred)
    return F.mse_loss(pred[mask], target[mask])


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return _zero_like(mu)
    mu, logvar = mu[mask], logvar[mask]
    per = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return per.mean()


def reconstruction_and_kl(out: Dict, x: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]):
    rec = _zero_like(out["h"])
    kl = _zero_like(out["h"])
    n = 0
    for m in sorted(x):
        rec = rec + masked_mse(out["recon"][m], x[m], masks[m])
        e = out["enc"][m]
        kl = kl + kl_standard_normal(e["mu_shared"], e["logvar_shared"], masks[m])
        kl = kl + kl_standard_normal(e["mu_private"], e["logvar_private"], masks[m])
        n += 1
    return rec / max(n, 1), kl / max(n, 1)


def paired_contrastive_loss(
    za: torch.Tensor,
    zb: torch.Tensor,
    paired_mask: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Same-cell contrastive term is correctly restricted to true paired observations."""
    p = paired_mask.bool()
    if p.sum() < 2:
        return _zero_like(za)
    a = F.normalize(za[p], dim=1)
    b = F.normalize(zb[p], dim=1)
    logits = a @ b.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return _zero_like(x)
    with torch.no_grad():
        if sigma is None:
            z = torch.cat([x, y], dim=0)
            d = torch.pdist(z, p=2)
            sigma_t = torch.median(d[d > 0]) if torch.any(d > 0) else x.new_tensor(1.0)
        else:
            sigma_t = x.new_tensor(float(sigma))
        gamma = 1.0 / (2.0 * sigma_t.clamp_min(1e-4).pow(2))
    kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
    kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2))
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2))
    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


def sinkhorn_plan(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 30) -> torch.Tensor:
    """Differentiable entropic OT with uniform marginals."""
    if cost.numel() == 0:
        return torch.zeros_like(cost)
    n, m = cost.shape
    # Log-domain Sinkhorn is substantially safer for large distances.
    log_a = cost.new_full((n,), -torch.log(cost.new_tensor(float(n))))
    log_b = cost.new_full((m,), -torch.log(cost.new_tensor(float(m))))
    log_k = -cost / max(float(epsilon), 1e-6)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        u = log_a - torch.logsumexp(log_k + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_k + u[:, None], dim=0)
    log_p = log_k + u[:, None] + v[None, :]
    return torch.exp(log_p)


def ot_alignment_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    n_iter: int = 30,
) -> torch.Tensor:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return _zero_like(x)
    cost = torch.cdist(x, y, p=2).pow(2)
    plan = sinkhorn_plan(cost.detach(), epsilon=epsilon, n_iter=n_iter)
    # Detach plan for stable alternating-style transport optimization.
    return (plan * cost).sum() / plan.sum().clamp_min(1e-8)


def _subsample_rows(x: torch.Tensor, max_n: int) -> torch.Tensor:
    if x.shape[0] <= max_n:
        return x
    # Deterministic within current torch RNG (seeded by runner).
    idx = torch.randperm(x.shape[0], device=x.device)[:max_n]
    return x[idx]


def multilevel_alignment_loss(
    out: Dict,
    masks: Mapping[str, torch.Tensor],
    epsilon: float,
    n_iter: int,
    max_cells: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper-consistent alignment.

    Critical fix over legacy CORAL: MMD and OT use each modality's complete
    observed population independently. They do *not* require a same-cell paired
    mask. Therefore these losses remain active at 0% pairing.
    """
    names = sorted(out["shared"])
    reference = out["h"]
    mmd = _zero_like(reference)
    ot = _zero_like(reference)
    con = _zero_like(reference)
    npairs = 0
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ma, mb = masks[a].bool(), masks[b].bool()
            za = _subsample_rows(out["shared"][a][ma], max_cells)
            zb = _subsample_rows(out["shared"][b][mb], max_cells)
            if za.shape[0] >= 2 and zb.shape[0] >= 2:
                mmd = mmd + rbf_mmd(za, zb)
                ot = ot + ot_alignment_loss(za, zb, epsilon, n_iter)
            con = con + paired_contrastive_loss(
                out["shared"][a], out["shared"][b], ma & mb
            )
            npairs += 1
    d = max(npairs, 1)
    return mmd / d, ot / d, con / d


def _normalize_distance_matrix(d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    positive = d[d > eps]
    if positive.numel() == 0:
        return d
    return d / positive.mean().clamp_min(eps)


def biological_structure_loss(
    out: Dict,
    x: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    k: int = 10,
    max_cells: int = 256,
) -> torch.Tensor:
    """scMRDR-style isometric biological-structure preservation.

    The complete modality-specific latent representation [z_s, z_p] is treated
    as a structure-rich reference, and the shared state z_s is encouraged to
    preserve its pairwise geometry. This matches the biological-conservation
    construction present in the strongest legacy branch more closely than an
    arbitrary input-space kNN penalty, while remaining label-free.

    ``x`` and ``k`` are retained in the public signature for config/backward
    compatibility; the loss itself is defined on latent geometry.
    """
    del x, k
    total = _zero_like(out["h"])
    used = 0
    for m in sorted(out["shared"]):
        idx = torch.where(masks[m].bool())[0]
        if idx.numel() < 3:
            continue
        if idx.numel() > max_cells:
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_cells]
            idx = idx[perm]
        z_shared = out["shared"][m][idx]
        z_full = torch.cat([out["shared"][m][idx], out["private"][m][idx]], dim=1)
        d_shared = _normalize_distance_matrix(torch.cdist(z_shared, z_shared, p=2))
        # Stop-gradient reference avoids the trivial solution in which both sides
        # simply deform together to reduce the geometry-matching loss.
        with torch.no_grad():
            d_full = _normalize_distance_matrix(torch.cdist(z_full, z_full, p=2))
        total = total + (d_shared - d_full).pow(2).mean()
        used += 1
    return total / max(used, 1)


def time_distillation_loss(
    pred_time: torch.Tensor,
    teacher_time: torch.Tensor,
    h: torch.Tensor,
    teacher_pairs: torch.Tensor | None = None,
    target_std: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    distill = F.mse_loss(pred_time, teacher_time)
    # Smoothness in latent kNN is injected through sampled pairs when provided;
    # pairwise ranking also makes the learned head retain teacher ordering.
    reg = F.relu(pred_time.new_tensor(target_std) - pred_time.std())
    if teacher_pairs is not None and teacher_pairs.numel() > 0:
        i, j = teacher_pairs[:, 0], teacher_pairs[:, 1]
        sign = torch.sign(teacher_time[i] - teacher_time[j])
        keep = sign != 0
        if keep.any():
            margin = 0.02
            reg = reg + F.relu(margin - sign[keep] * (pred_time[i[keep]] - pred_time[j[keep]])).mean()
    return distill, reg


def lag_objective(
    out: Dict,
    x: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    teacher_time: torch.Tensor,
    source: str = "atac",
    target: str = "rna",
    lambda_unpaired: float = 1.0,
    alpha_time: float = 0.5,
    alpha_latent: float = 0.25,
    temporal_window: float = 0.35,
    local_k: int = 32,
    epsilon: float = 0.05,
    n_iter: int = 30,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Corrected paired + unpaired lag objective from the rebuttal.

    Paired cells: direct same-cell target prediction.
    Source-only cells: predict target from source-observed input, then align those
    predictions to *other observed target cells* using time/local conditioned OT.
    No target value from the source-only cell is read.
    """
    key = f"{source}__to__{target}"
    if key not in out.get("lag", {}):
        z = _zero_like(out["h"])
        return z, {"paired": z, "unpaired": z}
    pred = out["lag"][key]["pred_target"]
    ma, mb = masks[source].bool(), masks[target].bool()
    paired = ma & mb
    source_only = ma & ~mb

    lp = _zero_like(pred)
    if paired.any():
        lp = F.mse_loss(pred[paired], x[target][paired])

    lu = _zero_like(pred)
    target_obs = mb
    if source_only.any() and target_obs.any():
        src_idx = torch.where(source_only)[0]
        tgt_idx = torch.where(target_obs)[0]
        y = pred[src_idx]
        xb = x[target][tgt_idx]
        # Main transport cost is target-profile discrepancy.
        cost_profile = torch.cdist(y, xb, p=2).pow(2) / max(y.shape[1], 1)
        # Compatibility terms are detached: they define a local transport support,
        # not an alternative path for leaking target information into source-only h.
        dt = torch.cdist(
            teacher_time[src_idx, None].detach(), teacher_time[tgt_idx, None].detach(), p=1
        )
        zsrc = out["shared"][source][src_idx].detach()
        ztgt = out["shared"][target][tgt_idx].detach()
        dz = torch.cdist(F.normalize(zsrc, dim=1), F.normalize(ztgt, dim=1), p=2)
        compat = alpha_time * dt + alpha_latent * dz
        cost = cost_profile + compat

        # Temporal/local neighborhood mask. Keep at least local_k candidates/row.
        allowed = dt <= temporal_window
        k = min(max(int(local_k), 1), tgt_idx.numel())
        nearest = torch.topk(compat, k=k, largest=False).indices
        rows = torch.arange(src_idx.numel(), device=pred.device)[:, None]
        allowed = allowed.clone()
        allowed[rows, nearest] = True
        large = cost.detach().max().clamp_min(1.0) + 1e3
        masked_cost = torch.where(allowed, cost, large)
        plan = sinkhorn_plan(masked_cost.detach(), epsilon=epsilon, n_iter=n_iter)
        plan = plan * allowed.float()
        if plan.sum() > 0:
            lu = (plan * cost_profile).sum() / plan.sum().clamp_min(1e-8)

    total = lp + float(lambda_unpaired) * lu
    return total, {"paired": lp, "unpaired": lu}


def cross_prediction_objective(
    out: Dict,
    x: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    teacher_time: torch.Tensor | None = None,
    source: str = "atac",
    target: str = "rna",
    lambda_unpaired: float = 1.0,
    alpha_time: float = 0.5,
    alpha_latent: float = 0.25,
    temporal_window: float = 0.35,
    local_k: int = 32,
    epsilon: float = 0.05,
    n_iter: int = 30,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Paired + unpaired source-only cross-omics prediction objective.

    The current manuscript's Appendix source-only loss is paired-only, which
    leaves G_{a->b} unsupervised when the observed pairing ratio is 0%. This
    corrected objective keeps the paper's strict source-only head but adds a
    population OT term for source-only cells, analogous to the rebuttal's
    corrected unpaired lag supervision.
    """
    key = f"{source}__to__{target}"
    if key not in out.get("cross", {}):
        z = _zero_like(out["h"])
        return z, {"paired": z, "unpaired": z}
    pred = out["cross"][key]
    ma, mb = masks[source].bool(), masks[target].bool()
    paired = ma & mb
    source_only = ma & ~mb

    lp = _zero_like(pred)
    if paired.any():
        lp = F.mse_loss(pred[paired], x[target][paired])

    lu = _zero_like(pred)
    if source_only.any() and mb.any():
        src_idx = torch.where(source_only)[0]
        tgt_idx = torch.where(mb)[0]
        y = pred[src_idx]
        xb = x[target][tgt_idx]
        cost_profile = torch.cdist(y, xb, p=2).pow(2) / max(y.shape[1], 1)

        zsrc = out["shared"][source][src_idx].detach()
        ztgt = out["shared"][target][tgt_idx].detach()
        dz = torch.cdist(F.normalize(zsrc, dim=1), F.normalize(ztgt, dim=1), p=2)
        compat = float(alpha_latent) * dz
        allowed = torch.ones_like(compat, dtype=torch.bool)
        if teacher_time is not None:
            dt = torch.cdist(
                teacher_time[src_idx, None].detach(),
                teacher_time[tgt_idx, None].detach(),
                p=1,
            )
            compat = compat + float(alpha_time) * dt
            allowed = dt <= float(temporal_window)
        k = min(max(int(local_k), 1), tgt_idx.numel())
        nearest = torch.topk(compat, k=k, largest=False).indices
        rows = torch.arange(src_idx.numel(), device=pred.device)[:, None]
        allowed = allowed.clone()
        allowed[rows, nearest] = True

        cost = cost_profile + compat
        large = cost.detach().max().clamp_min(1.0) + 1e3
        masked_cost = torch.where(allowed, cost, large)
        plan = sinkhorn_plan(masked_cost.detach(), epsilon=epsilon, n_iter=n_iter)
        plan = plan * allowed.float()
        if plan.sum() > 0:
            lu = (plan * cost_profile).sum() / plan.sum().clamp_min(1e-8)

    return lp + float(lambda_unpaired) * lu, {"paired": lp, "unpaired": lu}
