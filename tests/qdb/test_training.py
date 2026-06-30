"""Training-loop tests for QDB (per-block independence + score-matching loss)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from qdb.models import QDBGCN
from qdb.training import (
    QDBTrainConfig,
    build_per_block_optimizers,
    score_matching_loss,
    train_one_step,
)


class _StubConv(nn.Module):
    def __init__(self, in_channels: int, n_qubits: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, n_qubits)

    def forward(self, x, edge_index):
        return self.proj(x)


def _stub_factory(in_channels: int, n_qubits: int) -> nn.Module:
    return _StubConv(in_channels, n_qubits)


def _tiny_batch(batch_size: int = 4, nodes_per_graph: int = 3, in_channels: int = 4):
    n = batch_size * nodes_per_graph
    x = torch.randn(n, in_channels)
    edges = []
    for g in range(batch_size):
        base = g * nodes_per_graph
        for i in range(nodes_per_graph):
            for j in range(nodes_per_graph):
                if i != j:
                    edges.append([base + i, base + j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.repeat_interleave(torch.arange(batch_size), nodes_per_graph)
    labels = torch.randint(0, 2, (batch_size,))
    return x, edge_index, batch, labels


def test_score_matching_loss_is_zero_when_y_hat_equals_y():
    y = torch.randn(4, 8)
    sigma = torch.tensor([1.0, 2.0, 0.5, 3.0])
    loss = score_matching_loss(y, y, sigma)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_score_matching_loss_is_positive_when_predictions_differ():
    y = torch.randn(4, 8)
    y_hat = y + torch.randn_like(y)
    sigma = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss = score_matching_loss(y_hat, y, sigma)
    assert loss.item() > 0.0


def test_build_per_block_optimizers_returns_one_per_block():
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    opts = build_per_block_optimizers(model, lr=1e-3, weight_decay=1e-3)
    assert len(opts) == 2
    assert all(o is not None for o in opts)


def test_per_block_optimizers_share_only_class_embedding():
    """Block conv params disjoint; class_embedding intentionally shared.

    Class embedding must receive a gradient step regardless of which block
    runs (otherwise it stays at random orthogonal init and accuracy parks
    at chance). The remaining (per-block conv + AdaLN + pre_projection)
    params stay disjoint.
    """
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    opts = build_per_block_optimizers(model, lr=1e-3)
    p_in_opt0 = {id(p) for grp in opts[0].param_groups for p in grp["params"]}
    p_in_opt1 = {id(p) for grp in opts[1].param_groups for p in grp["params"]}
    shared = p_in_opt0 & p_in_opt1
    class_emb_ids = {id(p) for p in model.class_embedding.parameters()}
    # The intersection must be exactly the class embedding params.
    assert shared == class_emb_ids
    # And nothing else: block conv params remain disjoint.
    block0_only = p_in_opt0 - class_emb_ids
    block1_only = p_in_opt1 - class_emb_ids
    assert block0_only.isdisjoint(block1_only)


def test_train_one_step_decreases_loss_on_repeated_calls():
    """Sanity: same batch fed repeatedly -> loss trends down."""
    torch.manual_seed(0)
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    opts = build_per_block_optimizers(model, lr=1e-2)
    cfg = QDBTrainConfig(n_blocks=2, batch_size=4, epochs=1, seed=0)
    rng = np.random.default_rng(0)
    x, edge_index, batch, labels = _tiny_batch()
    losses = []
    for _ in range(20):
        loss = train_one_step(model, 0, opts[0], x, edge_index, batch, labels, cfg, rng)
        losses.append(loss)
    # Loss should trend downward (compare first 5 mean to last 5 mean)
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_block_independence_one_block_step_leaves_other_block_grads_none():
    """After stepping block 0 only, block 1 params should have None or zero grad."""
    torch.manual_seed(0)
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    opts = build_per_block_optimizers(model, lr=1e-2)
    cfg = QDBTrainConfig(n_blocks=2, batch_size=4, epochs=1, seed=0)
    rng = np.random.default_rng(0)
    x, edge_index, batch, labels = _tiny_batch()

    # Zero all grads first
    for p in model.parameters():
        p.grad = None

    train_one_step(model, 0, opts[0], x, edge_index, batch, labels, cfg, rng)

    # Block 1 params received no gradient (None or zero)
    for p in model.block_parameters(1):
        assert p.grad is None or torch.all(p.grad == 0)


def test_block_independence_param_values_isolated():
    """Stepping block 1 must not change block 0 parameter values."""
    torch.manual_seed(0)
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    opts = build_per_block_optimizers(model, lr=1e-2)
    cfg = QDBTrainConfig(n_blocks=2, batch_size=4, epochs=1, seed=0)
    rng = np.random.default_rng(0)
    x, edge_index, batch, labels = _tiny_batch()

    before = [p.detach().clone() for p in model.block_parameters(0)]
    train_one_step(model, 1, opts[1], x, edge_index, batch, labels, cfg, rng)
    after = [p.detach().clone() for p in model.block_parameters(0)]
    for b, a in zip(before, after):
        assert torch.allclose(b, a)
