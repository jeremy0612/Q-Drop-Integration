"""Tests for ``qdb.blocks`` — QDB GCN/GAT block wrappers.

A stub conv module is used in lieu of the real quantum convolutions so unit
tests stay fast and deterministic (no PennyLane).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qdb.blocks import QDBGATBlock, QDBGCNBlock


class _StubConv(nn.Module):
    """Identity-ish conv mapping ``[N, in_channels] -> [N, n_qubits]``.

    Uses a single Linear so we can verify shape contracts and gradient flow
    without invoking PennyLane in unit tests.
    """

    def __init__(self, in_channels: int, n_qubits: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, n_qubits)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _make_inputs(in_channels: int = 4, n_qubits: int = 8):
    """Build a fixed two-graph batch (3 nodes per graph) for tests."""
    torch.manual_seed(0)
    x = torch.randn(6, in_channels)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]], dtype=torch.long
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    z_sigma = torch.randn(2, n_qubits)
    log_sigma = torch.tensor(0.0)
    return x, edge_index, batch, z_sigma, log_sigma


def test_qdb_gcn_block_can_be_constructed() -> None:
    block = QDBGCNBlock(_StubConv(4, 8), in_channels=4, n_qubits=8)
    assert isinstance(block, nn.Module)


def test_qdb_gcn_block_forward_shape() -> None:
    block = QDBGCNBlock(_StubConv(4, 8), in_channels=4, n_qubits=8)
    x, edge_index, batch, z_sigma, log_sigma = _make_inputs()
    out = block(x, edge_index, batch, z_sigma, log_sigma)
    assert out.shape == (2, 8)


def test_zero_init_z_half_means_z_sigma_ignored_at_init() -> None:
    block = QDBGCNBlock(_StubConv(4, 8), in_channels=4, n_qubits=8)
    x, edge_index, batch, z_sigma_a, log_sigma = _make_inputs()
    z_sigma_b = torch.randn_like(z_sigma_a) * 5.0  # very different values

    block.eval()
    with torch.no_grad():
        out_a = block(x, edge_index, batch, z_sigma_a, log_sigma)
        out_b = block(x, edge_index, batch, z_sigma_b, log_sigma)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_per_graph_routing_isolates_z_sigma_across_graphs() -> None:
    block = QDBGCNBlock(_StubConv(4, 8), in_channels=4, n_qubits=8)
    # Manually break the zero-init on the z half so z_sigma actually matters.
    with torch.no_grad():
        block.pre_projection.weight[:, 4:].normal_(std=0.5)

    x, edge_index, batch, z_sigma, log_sigma = _make_inputs()
    z_sigma_perturbed = z_sigma.clone()
    z_sigma_perturbed[0] += 10.0  # perturb only graph 0's latent

    block.eval()
    with torch.no_grad():
        out_base = block(x, edge_index, batch, z_sigma, log_sigma)
        out_pert = block(x, edge_index, batch, z_sigma_perturbed, log_sigma)
    # Graph 1 must be unaffected.
    assert torch.allclose(out_base[1], out_pert[1], atol=1e-6)
    # Graph 0 must change.
    assert not torch.allclose(out_base[0], out_pert[0], atol=1e-4)


def test_differentiable_wrt_z_sigma() -> None:
    block = QDBGCNBlock(_StubConv(4, 8), in_channels=4, n_qubits=8)
    # Take a tiny optimizer step to break exact zero-init on z half.
    opt = torch.optim.SGD(block.parameters(), lr=1e-2)
    x, edge_index, batch, z_sigma, log_sigma = _make_inputs()
    target = torch.zeros(2, 8)
    out = block(x, edge_index, batch, z_sigma, log_sigma)
    loss = ((out - target) ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    z_sigma2 = z_sigma.detach().clone().requires_grad_(True)
    out2 = block(x, edge_index, batch, z_sigma2, log_sigma)
    out2.sum().backward()
    assert z_sigma2.grad is not None
    assert torch.any(z_sigma2.grad.abs() > 0)


def test_pre_projection_z_half_is_zero_at_init() -> None:
    block = QDBGCNBlock(_StubConv(4, 8), in_channels=4, n_qubits=8)
    z_half = block.pre_projection.weight[:, 4:]
    assert torch.all(z_half == 0)


def test_pre_projection_x_half_is_nonzero_at_init() -> None:
    block = QDBGCNBlock(_StubConv(4, 8), in_channels=4, n_qubits=8)
    x_half = block.pre_projection.weight[:, :4]
    assert torch.any(x_half != 0)


def test_qdb_gat_block_same_contract() -> None:
    block = QDBGATBlock(_StubConv(4, 8), in_channels=4, n_qubits=8)
    x, edge_index, batch, z_sigma, log_sigma = _make_inputs()
    out = block(x, edge_index, batch, z_sigma, log_sigma)
    assert out.shape == (2, 8)


def test_log_sigma_vector_broadcasts_per_graph() -> None:
    block = QDBGCNBlock(_StubConv(4, 8), in_channels=4, n_qubits=8)
    # Break zero-init on AdaLN so different log_sigmas actually matter.
    opt = torch.optim.SGD(block.parameters(), lr=1e-2)
    x, edge_index, batch, z_sigma, log_sigma_scalar = _make_inputs()
    out = block(x, edge_index, batch, z_sigma, log_sigma_scalar)
    out.sum().backward()
    opt.step()
    opt.zero_grad()

    log_sigma_vec = torch.tensor([0.0, 1.0])
    block.eval()
    with torch.no_grad():
        out_vec = block(x, edge_index, batch, z_sigma, log_sigma_vec)
        out_same = block(
            x, edge_index, batch, z_sigma, torch.tensor([0.0, 0.0])
        )
    assert not torch.allclose(out_vec[0], out_vec[1], atol=1e-4)
    # When all log_sigmas equal, both graphs should equal the scalar=0 result up to numerics.
    assert torch.allclose(out_vec[0], out_same[0], atol=1e-6)
