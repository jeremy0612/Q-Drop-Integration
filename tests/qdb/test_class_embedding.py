"""Tests for qdb.class_embedding.ClassEmbedding."""

from __future__ import annotations

import pytest
import torch

from qdb.class_embedding import ClassEmbedding


def test_module_can_be_constructed() -> None:
    module = ClassEmbedding(n_classes=2, dim=8)
    assert isinstance(module, torch.nn.Module)


def test_forward_returns_correct_shape() -> None:
    module = ClassEmbedding(n_classes=2, dim=8)
    indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    out = module(indices)
    assert out.shape == (4, 8)


def test_orthogonal_init_rows_are_decorrelated() -> None:
    torch.manual_seed(0)
    module = ClassEmbedding(n_classes=4, dim=16, freeze_norm=False)
    w = module.weight.detach()
    gram = w @ w.T
    off = gram - torch.diag(torch.diagonal(gram))
    assert off.abs().max().item() < 1e-5


def test_freeze_norm_enforces_unit_rows() -> None:
    module = ClassEmbedding(n_classes=4, dim=8, freeze_norm=True)
    w = module.weight
    norms = w.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_freeze_norm_survives_optimizer_step() -> None:
    torch.manual_seed(1)
    module = ClassEmbedding(n_classes=4, dim=8, freeze_norm=True)
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target = torch.randn(4, 8)
    out = module(indices)
    loss = torch.nn.functional.mse_loss(out, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    norms = module.weight.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_no_freeze_norm_allows_norms_to_drift() -> None:
    torch.manual_seed(2)
    module = ClassEmbedding(n_classes=4, dim=8, freeze_norm=False)
    opt = torch.optim.SGD(module.parameters(), lr=1.0)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target = torch.randn(4, 8) * 5.0
    out = module(indices)
    loss = torch.nn.functional.mse_loss(out, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    norms = module.weight.norm(dim=1)
    assert (norms - 1.0).abs().max().item() > 1e-3


def test_nearest_class_recovers_index_from_exact_row() -> None:
    torch.manual_seed(3)
    module = ClassEmbedding(n_classes=3, dim=8, freeze_norm=True)
    z = module.weight.detach().clone()
    preds = module.nearest_class(z)
    assert preds.dtype == torch.long
    assert preds.shape == (3,)
    assert torch.equal(preds, torch.tensor([0, 1, 2], dtype=torch.long))


def test_nearest_class_handles_noisy_input() -> None:
    torch.manual_seed(4)
    module = ClassEmbedding(n_classes=5, dim=16, freeze_norm=True)
    z = module.weight.detach().clone() + 0.01 * torch.randn(5, 16)
    preds = module.nearest_class(z)
    assert torch.equal(preds, torch.tensor([0, 1, 2, 3, 4], dtype=torch.long))


def test_dim_one_fallback_does_not_crash() -> None:
    module = ClassEmbedding(n_classes=4, dim=1, freeze_norm=False)
    w = module.weight
    assert w.shape == (4, 1)
    assert torch.isfinite(w).all().item()
