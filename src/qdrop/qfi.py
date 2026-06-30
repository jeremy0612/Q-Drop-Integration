"""QFI-Drop math: per-gate spectral leverage + freeze-and-precondition update.

Framework-agnostic (torch only). See docs/superpowers/specs/2026-06-29-qfi-drop-design.md.
"""
from __future__ import annotations

import torch


def spectral_leverage(F: torch.Tensor, spectral_ratio: float = 1e-3) -> torch.Tensor:
    """Per-gate kept-energy ``e_j = sum_{i: lambda_i >= ratio*lambda_max} U_ji**2``, in [0, 1].

    F: (p, p) symmetric PSD QFIM. Returns (p,) leverage scores — the fraction of each
    gate's energy lying in the high-curvature eigensubspace.
    """
    F = 0.5 * (F + F.T)                                   # symmetrize numerical noise
    evals, evecs = torch.linalg.eigh(F)                  # ascending; eigenvectors are columns
    lam_max = evals.max().clamp_min(torch.finfo(F.dtype).tiny)
    keep = evals >= spectral_ratio * lam_max
    return (evecs[:, keep] ** 2).sum(dim=1)              # row-sum over kept columns


def prune_and_precondition(
    grad: torch.Tensor,
    F: torch.Tensor,
    e: torch.Tensor,
    energy_threshold: float = 0.5,
    reg: float = 1e-4,
) -> torch.Tensor:
    """Freeze gates with leverage < ``energy_threshold``; QNG-precondition the survivors.

    Returns ``(F_SS + reg*I)^-1 g_S`` on surviving indices, 0 on frozen, same shape as grad.
    """
    g = grad.reshape(-1).to(F.dtype)
    survive = e >= energy_threshold
    out = torch.zeros_like(g)
    if survive.any():
        idx = survive.nonzero(as_tuple=True)[0]
        F_ss = F.index_select(0, idx).index_select(1, idx)
        eye = torch.eye(idx.numel(), dtype=F.dtype, device=F.device)
        # ponytail: inline (F_SS+reg*I)^-1 g_S. qng.apply_fubini_study_precondition does the
        # same, but it lives on the feat/qml_techniques branch; a 2-line solve avoids the dep.
        out[idx] = torch.linalg.solve(F_ss + reg * eye, g[idx])
    return out.reshape(grad.shape).to(grad.dtype)
