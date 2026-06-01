"""Quantum Natural Gradient (Stokes et al. 2020) Adam optimizer wrapper.

For each quantum parameter group, compute the Fubini-Study metric tensor
F at the current parameter values, then precondition the gradient:
    g' = (F + reg*I)^{-1} g
before passing to Adam. Classical parameters use plain Adam.

The metric tensor is obtained from a caller-supplied `metric_provider`
callable so this module can be unit-tested without invoking PennyLane.
In production wiring, the provider closes over ``qml.metric_tensor(qnode)``
and a representative input batch.
"""

from __future__ import annotations

from typing import Callable, Iterable, List

import torch
import torch.nn as nn
import torch.optim as optim


def apply_fubini_study_precondition(
    grad: torch.Tensor, metric: torch.Tensor, reg: float = 1e-4
) -> torch.Tensor:
    """Return ``(F + reg*I)^{-1} g`` flattened to grad's shape."""
    flat = grad.reshape(-1)
    n = flat.numel()
    regularized = metric + reg * torch.eye(n, dtype=metric.dtype, device=metric.device)
    preconditioned = torch.linalg.solve(regularized, flat)
    return preconditioned.reshape(grad.shape)


class QNGAdam(optim.Optimizer):
    """Two-group Adam where the quantum group's gradient is QNG-preconditioned."""

    def __init__(
        self,
        quantum_params: Iterable[nn.Parameter],
        classical_params: Iterable[nn.Parameter],
        metric_provider: Callable[[nn.Parameter], torch.Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        qng_reg: float = 1e-4,
    ) -> None:
        quantum_list: List[nn.Parameter] = list(quantum_params)
        classical_list: List[nn.Parameter] = list(classical_params)
        param_groups = [
            {
                "params": quantum_list,
                "name": "quantum",
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
            {
                "params": classical_list,
                "name": "classical",
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        ]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)
        self.metric_provider = metric_provider
        self.qng_reg = qng_reg
        # Delegate Adam math to a vanilla Adam over the same parameter lists.
        self._adam = optim.Adam(
            [{"params": quantum_list}, {"params": classical_list}],
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Precondition quantum gradients in place BEFORE Adam consumes them.
        for p in self.param_groups[0]["params"]:
            if p.grad is None:
                continue
            metric = self.metric_provider(p)
            p.grad = apply_fubini_study_precondition(p.grad, metric, reg=self.qng_reg)

        self._adam.step()
        return loss
