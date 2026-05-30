"""Learnable per-class embedding table for QDB diffusion targets.

Rows are initialized orthogonally so that classes start maximally decorrelated.
When ``freeze_norm=True`` a parametrization forces every row to unit L2 norm at
every access, including after optimizer steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _UnitNorm(nn.Module):
    """Parametrization that L2-normalizes each row of a 2D weight tensor."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return F.normalize(weight, p=2.0, dim=1, eps=1e-12)

    def right_inverse(self, weight: torch.Tensor) -> torch.Tensor:
        return F.normalize(weight, p=2.0, dim=1, eps=1e-12)


class ClassEmbedding(nn.Module):
    """Learnable ``[n_classes, dim]`` embedding table used as a diffusion target.

    Args:
        n_classes: number of classes.
        dim: embedding dimension (= n_qubits for QDB use).
        freeze_norm: if True, enforce each row has unit L2 norm via a
            parametrization. If False, plain :class:`~torch.nn.Parameter`.

    Forward: takes a ``LongTensor`` of class indices, returns embeddings
    ``[batch, dim]``.

    Special case ``dim=1``: :func:`torch.nn.init.orthogonal_` is degenerate on
    single-column matrices, so the weight is initialized as uniform in
    ``[-1, 1]`` and then (if ``freeze_norm=True``) normalized to unit norm.
    """

    def __init__(self, n_classes: int, dim: int, freeze_norm: bool = True) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.dim = int(dim)
        self.freeze_norm = bool(freeze_norm)

        raw = torch.empty(self.n_classes, self.dim)
        if self.dim == 1:
            raw.uniform_(-1.0, 1.0)
        else:
            nn.init.orthogonal_(raw)
        self._weight = nn.Parameter(raw)

        if self.freeze_norm:
            nn.utils.parametrize.register_parametrization(
                self, "_weight", _UnitNorm()
            )

    @property
    def weight(self) -> torch.Tensor:
        """Return the ``[n_classes, dim]`` embedding matrix (normalized if frozen)."""
        return self._weight

    def forward(self, class_indices: torch.Tensor) -> torch.Tensor:
        if class_indices.dtype != torch.long:
            class_indices = class_indices.long()
        return F.embedding(class_indices, self.weight)

    def nearest_class(self, z: torch.Tensor) -> torch.Tensor:
        """Predict class indices by cosine similarity to embedding rows.

        Args:
            z: tensor of shape ``[batch, dim]``.

        Returns:
            ``LongTensor`` of shape ``[batch]`` with the argmax cosine match.
        """
        w = self.weight
        z_norm = F.normalize(z, p=2.0, dim=1, eps=1e-12)
        w_norm = F.normalize(w, p=2.0, dim=1, eps=1e-12)
        sims = z_norm @ w_norm.T
        return sims.argmax(dim=1).long()
