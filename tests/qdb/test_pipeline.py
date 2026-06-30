"""End-to-end pipeline smoke test on synthetic graphs (no real quantum)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qdb.models import QDBGCN
from qdb.training import QDBTrainConfig, evaluate, train_qdb


class _StubConv(nn.Module):
    def __init__(self, in_channels: int, n_qubits: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, n_qubits)

    def forward(self, x, edge_index):
        return self.proj(x)


def _stub_factory(in_channels: int, n_qubits: int) -> nn.Module:
    return _StubConv(in_channels, n_qubits)


def _make_synthetic_dataset(n_graphs: int = 8, in_channels: int = 4, nodes_per_graph: int = 4):
    """Two-class linearly-separable graph dataset.

    Each graph has random node features but the label is determined by
    whether the per-graph mean of feature 0 is positive.
    """
    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(n_graphs):
        x = torch.from_numpy(rng.normal(size=(nodes_per_graph, in_channels)).astype("float32"))
        # Bias label by feature 0 sum so the task is learnable.
        label = int((x[:, 0].mean() > 0).item())
        edges = []
        for i in range(nodes_per_graph):
            for j in range(nodes_per_graph):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([label])))
    return graphs


def test_pipeline_smoke_full_train_one_epoch():
    """Build model + dataset, run 1 training epoch end-to-end. No crash, returns history."""
    torch.manual_seed(0)
    graphs = _make_synthetic_dataset(n_graphs=8)
    loader = DataLoader(graphs, batch_size=4, shuffle=False)
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    cfg = QDBTrainConfig(n_blocks=2, batch_size=4, epochs=1, seed=0)
    history = train_qdb(model, loader, loader, cfg, torch.device("cpu"))
    assert "block_loss" in history
    assert len(history["block_loss"]) == 2
    assert len(history["val_accuracy"]) == 1


def test_pipeline_evaluate_returns_accuracy_dict():
    torch.manual_seed(0)
    graphs = _make_synthetic_dataset(n_graphs=4)
    loader = DataLoader(graphs, batch_size=4, shuffle=False)
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory
    )
    out = evaluate(model, loader, torch.device("cpu"))
    assert "accuracy" in out
    assert 0.0 <= out["accuracy"] <= 1.0


@pytest.mark.slow
def test_pipeline_can_learn_separable_task_over_many_epochs():
    """30 epochs on tiny separable dataset: val accuracy should exceed 0.6.

    Marked slow but still runs by default. Sanity check that training
    plumbing actually moves weights in a useful direction.
    """
    torch.manual_seed(0)
    graphs = _make_synthetic_dataset(n_graphs=32, nodes_per_graph=6)
    loader = DataLoader(graphs, batch_size=8, shuffle=True)
    model = QDBGCN(
        n_blocks=2, in_channels=4, n_qubits=8, n_classes=2, conv_factory=_stub_factory,
        freeze_class_norm=True,
    )
    cfg = QDBTrainConfig(n_blocks=2, batch_size=8, epochs=30, lr=5e-3, seed=0)
    history = train_qdb(model, loader, loader, cfg, torch.device("cpu"))
    # Tail-average over last 5 epochs to smooth noise.
    tail_acc = np.mean(history["val_accuracy"][-5:])
    assert tail_acc > 0.55, f"learned tail acc only {tail_acc:.3f}"
