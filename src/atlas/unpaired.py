"""Unpaired alignment utilities for ATLAS v0.2."""
import torch

def ot_barycentric_alignment(z_src, z_tgt, transport, eps=1e-8):
    """Soft OT correspondence loss.

    transport: [Ns,Nt] Sinkhorn plan.
    """
    bary = transport @ z_tgt
    mass = transport.sum(1, keepdim=True).clamp_min(eps)
    bary = bary / mass
    return ((z_src - bary) ** 2).sum(dim=1).mean()
