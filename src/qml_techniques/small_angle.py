"""Small-angle init for quantum rotation parameters (Grant et al. 2019)."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def make_small_angle_init(std: float = 0.1) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable that fills its tensor with N(0, std^2).

    Use as ``init_method={"weights": make_small_angle_init(0.1)}`` in
    ``qml.qnn.TorchLayer``. Default PennyLane init is uniform [0, 2*pi],
    which places parameters in the near-flat-gradient regime (barren
    plateau neighborhood) for deep circuits. Small-angle init keeps the
    circuit near identity at construction so early gradients are
    informative.
    """

    def _init(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return _init
