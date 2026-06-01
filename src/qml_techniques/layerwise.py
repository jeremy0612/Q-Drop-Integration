"""Layerwise learning schedule (Skolik et al. 2021).

Train quantum weights one conv-layer at a time on a curriculum. Phase k
unfreezes layers 0..k-1. Reduces barren-plateau symptoms by limiting the
parameter space being explored at any one time.
"""

from __future__ import annotations

import torch.nn as nn


class LayerwiseSchedule:
    """Toggle ``requires_grad`` on per-layer quantum weights per training phase."""

    def __init__(self, model: nn.Module, total_epochs: int, n_phases: int = 2) -> None:
        if n_phases < 1:
            raise ValueError("n_phases must be >= 1")
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1")
        self.model = model
        self.total_epochs = total_epochs
        self.n_phases = n_phases
        self.layers = list(getattr(model, "layers", []))
        self._epochs_per_phase = max(1, total_epochs // n_phases)

    def apply_for_epoch(self, epoch: int) -> None:
        """Update requires_grad on quantum weights for the current phase."""
        active_through = min(self.n_phases, 1 + (epoch // self._epochs_per_phase))
        for idx, layer in enumerate(self.layers):
            quantum_weights = getattr(getattr(layer, "quantum_layer", None), "weights", None)
            if quantum_weights is None:
                continue
            quantum_weights.requires_grad_(idx < active_through)
