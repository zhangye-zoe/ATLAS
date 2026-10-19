from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.1, last_activation: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or last_activation:
                layers += [nn.LayerNorm(dims[i + 1]), nn.GELU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VariationalSplitEncoder(nn.Module):
    """True shared/private variational encoder matching the manuscript formulation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        shared_dim: int,
        private_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = MLP([input_dim, *hidden_dims], dropout=dropout, last_activation=True)
        hdim = hidden_dims[-1]
        self.mu_s = nn.Linear(hdim, shared_dim)
        self.logvar_s = nn.Linear(hdim, shared_dim)
        self.mu_p = nn.Linear(hdim, private_dim)
        self.logvar_p = nn.Linear(hdim, private_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar.clamp(-12.0, 8.0))
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        mu_s, lv_s = self.mu_s(h), self.logvar_s(h)
        mu_p, lv_p = self.mu_p(h), self.logvar_p(h)
        z_s = self.reparameterize(mu_s, lv_s, sample)
        z_p = self.reparameterize(mu_p, lv_p, sample)
        return {
            "shared": z_s,
            "private": z_p,
            "mu_shared": mu_s,
            "logvar_shared": lv_s,
            "mu_private": mu_p,
            "logvar_private": lv_p,
        }


class MaskAwareFusion(nn.Module):
    def __init__(self, shared_dim: int, modality_names: Sequence[str]):
        super().__init__()
        self.modality_names = list(modality_names)
        n_mod = len(self.modality_names)
        self.gate = nn.Sequential(
            nn.Linear(n_mod * shared_dim + n_mod, max(32, shared_dim)),
            nn.GELU(),
            nn.Linear(max(32, shared_dim), n_mod),
        )

    def forward(
        self, shared: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.stack([shared[m] for m in self.modality_names], dim=1)
        mask = torch.stack([masks[m].float() for m in self.modality_names], dim=1)
        gate_in = torch.cat([z.flatten(1), mask], dim=1)
        logits = self.gate(gate_in).masked_fill(mask <= 0, -1e9)
        # Every dataset row must have >=1 observed modality.
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Mask-aware fusion received a cell with no observed modality.")
        alpha = F.softmax(logits, dim=1)
        h = torch.sum(alpha.unsqueeze(-1) * z, dim=1)
        return h, alpha


class TimeHead(nn.Module):
    def __init__(self, shared_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), nn.GELU(), nn.Linear(shared_dim, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h)).squeeze(-1)


class DirectedLagAttention(nn.Module):
    """Attention over source-modality historical states followed by Eq.-16 update."""

    def __init__(self, shared_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(shared_dim * 3, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )

    def forward(
        self,
        current_h: torch.Tensor,
        source_bank: torch.Tensor,
        bank_valid: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        # source_bank: [B, K, D], importantly source-specific z_s^(a), not fused h.
        b, k, d = source_bank.shape
        cur = current_h[:, None, :].expand(b, k, d)
        score = self.score(torch.cat([cur, source_bank, cur - source_bank], dim=-1)).squeeze(-1)
        if bank_valid is not None:
            valid = bank_valid.bool()
            # If all bins are invalid for a cell, keep first zero bin available to avoid NaN.
            all_bad = ~valid.any(dim=1)
            if all_bad.any():
                valid = valid.clone()
                valid[all_bad, 0] = True
            score = score.masked_fill(~valid, -1e9)
        alpha = F.softmax(score, dim=1)
        context = torch.sum(alpha[..., None] * source_bank, dim=1)
        h_tilde = self.update(torch.cat([current_h, context], dim=1))
        return {"context": context, "alpha": alpha, "h_tilde": h_tilde}


class ATLAS(nn.Module):
    def __init__(
        self,
        input_dims: Mapping[str, int],
        shared_dim: int = 32,
        private_dim: int = 16,
        encoder_hidden: Sequence[int] = (256, 128),
        decoder_hidden: Sequence[int] = (128, 256),
        dropout: float = 0.1,
        lag_edges: Sequence[Tuple[str, str]] = (("atac", "rna"),),
    ):
        super().__init__()
        self.modality_names = sorted(input_dims)
        self.shared_dim = int(shared_dim)
        self.private_dim = int(private_dim)
        self.input_dims = dict(input_dims)
        self.encoders = nn.ModuleDict(
            {
                m: VariationalSplitEncoder(
                    d, encoder_hidden, shared_dim, private_dim, dropout
                )
                for m, d in input_dims.items()
            }
        )
        self.fusion = MaskAwareFusion(shared_dim, self.modality_names)
        self.fusion_refine = MLP([shared_dim, shared_dim, shared_dim], dropout=dropout)
        self.decoders = nn.ModuleDict(
            {
                m: MLP(
                    [shared_dim + private_dim, *decoder_hidden, d],
                    dropout=dropout,
                )
                for m, d in input_dims.items()
            }
        )
        # T2 heads are deliberately source-only. They never receive fused h or target latents.
        self.cross_predictors = nn.ModuleDict()
        for src, tgt in lag_edges:
            if src in input_dims and tgt in input_dims:
                self.cross_predictors[f"{src}__to__{tgt}"] = MLP(
                    [shared_dim + private_dim, *decoder_hidden, input_dims[tgt]],
                    dropout=dropout,
                )
        self.time_head = TimeHead(shared_dim)
        self.lag_modules = nn.ModuleDict()
        self.lag_decoders = nn.ModuleDict()
        for src, tgt in lag_edges:
            key = f"{src}__to__{tgt}"
            self.lag_modules[key] = DirectedLagAttention(shared_dim)
            # Eq. 17 must decode h_tilde, not the raw lag context.
            self.lag_decoders[key] = MLP(
                [shared_dim, *decoder_hidden, input_dims[tgt]], dropout=dropout
            )

    def encode(
        self,
        x: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        sample: bool | None = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if sample is None:
            sample = self.training
        out = {}
        for m in self.modality_names:
            # Missing rows are zeros, but their encoder outputs are ignored by masks in all losses/fusion.
            out[m] = self.encoders[m](x[m], sample=sample)
        return out

    def fuse(
        self,
        enc: Mapping[str, Mapping[str, torch.Tensor]],
        masks: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = {m: enc[m]["shared"] for m in self.modality_names}
        h, alpha = self.fusion(shared, masks)
        return self.fusion_refine(h), alpha

    def forward(
        self,
        x: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        lag_banks: Mapping[str, torch.Tensor] | None = None,
        lag_valid: Mapping[str, torch.Tensor] | None = None,
        sample: bool | None = None,
    ) -> Dict:
        enc = self.encode(x, masks, sample=sample)
        h, fusion_alpha = self.fuse(enc, masks)
        recon = {
            m: self.decoders[m](torch.cat([h, enc[m]["private"]], dim=1))
            for m in self.modality_names
        }
        cross = {}
        for key, head in self.cross_predictors.items():
            src, _ = key.split("__to__")
            cross[key] = head(torch.cat([enc[src]["shared"], enc[src]["private"]], dim=1))
        out = {
            "enc": enc,
            "shared": {m: enc[m]["shared"] for m in self.modality_names},
            "private": {m: enc[m]["private"] for m in self.modality_names},
            "h": h,
            "fusion_alpha": fusion_alpha,
            "recon": recon,
            "cross": cross,
            "time": self.time_head(h),
            "lag": {},
        }
        if lag_banks:
            for key, bank in lag_banks.items():
                if key not in self.lag_modules:
                    continue
                valid = lag_valid.get(key) if lag_valid is not None else None
                lo = self.lag_modules[key](h, bank, valid)
                lo["pred_target"] = self.lag_decoders[key](lo["h_tilde"])
                out["lag"][key] = lo
        return out

    @torch.no_grad()
    def encode_modality_only(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic modality-specific shared mean for T1."""
        return self.encoders[modality](x, sample=False)["mu_shared"]

    @torch.no_grad()
    def predict_from_source(self, source: str, target: str, x_source: torch.Tensor) -> torch.Tensor:
        """Strict source-only T2 path; no target tensor is accepted by this API."""
        enc = self.encoders[source](x_source, sample=False)
        key = f"{source}__to__{target}"
        return self.cross_predictors[key](torch.cat([enc["mu_shared"], enc["mu_private"]], dim=1))
