"""Adaptive LayerNorm modulation conditioned on log(sigma).

Maps a per-sample log-sigma scalar through a small SiLU MLP to ``(gamma, beta)``
modulation parameters, then applies ``y = (1 + gamma) * x + beta`` to a
pre-normalized input. The final Linear is zero-initialized so the module is the
identity at construction time, which is critical for stable diffusion training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaLNModulation(nn.Module):
    """Adaptive LayerNorm modulation conditioned on log(sigma).

    Args:
        dim: feature dimension to modulate.
        hidden: MLP hidden size (default 32).
    """

    def __init__(self, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )
        # Zero-init final Linear so gamma = beta = 0 at start (identity output).
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        x: torch.Tensor,
        log_sigma: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Modulate ``x`` using ``log_sigma``.

        Args:
            x: ``[N, dim]`` normalized node features.
            log_sigma: scalar tensor (single sample) or ``[batch_size]`` tensor.
            batch: optional ``[N]`` index mapping each node to a sample. Required
                when ``log_sigma`` is a vector with multiple samples.

        Returns:
            ``[N, dim]`` modulated features.
        """
        if log_sigma.dim() == 0:
            ls = log_sigma.view(1, 1)
            params = self.mlp(ls)  # [1, 2*dim]
            gamma, beta = params.chunk(2, dim=-1)
            return (1 + gamma) * x + beta

        ls = log_sigma.view(-1, 1)  # [B, 1]
        params = self.mlp(ls)        # [B, 2*dim]
        gamma, beta = params.chunk(2, dim=-1)  # [B, dim] each

        if batch is None:
            # Vector log_sigma without a batch index is only valid if it has a
            # single entry; treat it like the scalar case.
            return (1 + gamma) * x + beta

        gamma_n = gamma.index_select(0, batch)  # [N, dim]
        beta_n = beta.index_select(0, batch)    # [N, dim]
        return (1 + gamma_n) * x + beta_n
