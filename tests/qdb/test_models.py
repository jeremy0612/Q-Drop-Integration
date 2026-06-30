"""Smoke + unit tests for QDB model containers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from qdb.models import QDBGCN, QDBGAT
from qdb.blocks import QDBGCNBlock, QDBGATBlock


class _StubConv(nn.Module):
    """Identity-ish conv: [N, in_channels] -> [N, n_qubits] via single Linear."""

    def __init__(self, in_channels: int, n_qubits: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, n_qubits)

    def forward(self, x, edge_index):
        return self.proj(x)


def _stub_factory(in_channels: int, n_qubits: int) -> nn.Module:
    return _StubConv(in_channels, n_qubits)


def _tiny_batch(batch_size: int = 2, nodes_per_graph: int = 3, in_channels: int = 4):
    """Build a minimal PyG-style batch by hand."""
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
    return x, edge_index, batch


def test_qdbgcn_constructs_with_two_blocks():
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    assert len(model.blocks) == 2
    assert model.n_blocks == 2
    assert model.n_qubits == 8


def test_sigma_boundaries_length_is_n_blocks_plus_one():
    model = QDBGCN(
        n_blocks=4, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    assert model.sigma_boundaries.shape == (5,)
    # Descending order: hi first, lo last.
    assert model.sigma_boundaries[0].item() > model.sigma_boundaries[-1].item()


def test_block_sigma_range_returns_ascending_pair():
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    lo, hi = model.block_sigma_range(0)
    assert lo < hi


def test_forward_block_shape_matches_batch():
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    x, edge_index, batch = _tiny_batch(batch_size=2, nodes_per_graph=3, in_channels=4)
    z_sigma = torch.randn(2, 8)
    log_sigma = torch.zeros(2)
    out = model.forward_block(0, x, edge_index, batch, z_sigma, log_sigma)
    assert out.shape == (2, 8)


def test_block_parameters_are_disjoint_across_blocks():
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    p0 = {id(p) for p in model.block_parameters(0)}
    p1 = {id(p) for p in model.block_parameters(1)}
    assert p0.isdisjoint(p1)


def test_denoise_returns_per_graph_latent():
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    x, edge_index, batch = _tiny_batch(batch_size=3, nodes_per_graph=2, in_channels=4)
    z_t = model.denoise(x, edge_index, batch, batch_size=3)
    assert z_t.shape == (3, 8)


def test_predict_returns_long_indices_in_class_range():
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    x, edge_index, batch = _tiny_batch(batch_size=2, nodes_per_graph=3, in_channels=4)
    preds = model.predict(x, edge_index, batch, batch_size=2)
    assert preds.dtype == torch.long
    assert preds.shape == (2,)
    assert (preds >= 0).all() and (preds < 2).all()


def test_invalid_block_idx_raises():
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    with pytest.raises(IndexError):
        model.block_sigma_range(5)


def test_qdbgat_subclass_uses_gat_block():
    model = QDBGAT(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    assert isinstance(model.blocks[0], QDBGATBlock)
