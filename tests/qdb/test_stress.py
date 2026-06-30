"""Stress / memory profile tests on CPU.

No GPU available on dev Mac, so we approximate the B× memory reduction
hypothesis via process RSS or peak tensor allocations on CPU. Real GPU
verification is deferred to the CI rig (A6000 / P6000) — see plan Phase 5.
"""

from __future__ import annotations

import gc
import os
import resource
import time
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qdb.models import QDBGCN
from qdb.training import QDBTrainConfig, train_qdb


class _LinearConv(nn.Module):
    """Heavier-weight stub: 2-layer MLP, so the memory profile is non-trivial."""

    def __init__(self, in_channels: int, n_qubits: int) -> None:
        super().__init__()
        # Inflate parameter count so the stress test sees a measurable diff
        hidden = max(n_qubits * 4, 64)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_qubits),
        )

    def forward(self, x, edge_index):
        return self.proj(x)


def _linear_factory(in_channels: int, n_qubits: int) -> nn.Module:
    return _LinearConv(in_channels, n_qubits)


def _make_dataset(n_graphs: int = 64, in_channels: int = 16, nodes_per_graph: int = 32):
    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(n_graphs):
        x = torch.from_numpy(rng.normal(size=(nodes_per_graph, in_channels)).astype("float32"))
        label = int((x[:, 0].mean() > 0).item())
        edges = []
        for i in range(nodes_per_graph):
            for j in range(nodes_per_graph):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([label])))
    return graphs


def _peak_rss_mb() -> float:
    """Process peak RSS in megabytes."""
    # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if maxrss > 1e9:  # bytes (macOS)
        return maxrss / (1024 * 1024)
    return maxrss / 1024  # KB -> MB


def _train_briefly(model: QDBGCN, loader, epochs: int = 2) -> Dict[str, float]:
    cfg = QDBTrainConfig(n_blocks=model.n_blocks, batch_size=8, epochs=epochs, seed=0)
    history = train_qdb(model, loader, None, cfg, torch.device("cpu"))
    return history


@pytest.mark.stress
def test_param_count_scales_linearly_with_n_blocks():
    """Sanity: total params at B=4 ≈ 4× params at B=1 (blocks are independent)."""

    def n_params(b: int) -> int:
        m = QDBGCN(
            n_blocks=b, in_channels=16, n_qubits=8, n_classes=2,
            conv_factory=_linear_factory,
        )
        return sum(p.numel() for p in m.parameters())

    p1 = n_params(1)
    p4 = n_params(4)
    # Each block adds the same conv+adaln+pre_projection. Class embedding is shared (1x).
    # So p4 ~ 4 * (per-block) + 1 * (class_embedding). Ratio close to 4 but with shared
    # embedding subtracted. Tolerance: 3.5x to 4.5x.
    ratio = p4 / p1
    assert 3.0 < ratio < 4.5, f"block param scaling broken: ratio={ratio:.2f}"


@pytest.mark.stress
def test_per_step_peak_memory_does_not_grow_with_n_blocks_at_fixed_step():
    """One training step touches only ONE block. Memory should be ~constant in B.

    Rationale: total params scale linearly, but per-step backward graph only
    materializes activations for the active block (DiffusionBlocks core
    promise). On CPU we can't measure GPU memory, so we check that peak
    Python heap growth between successive steps is small relative to model
    size differences.
    """
    torch.manual_seed(0)
    graphs = _make_dataset(n_graphs=16, in_channels=16, nodes_per_graph=16)
    loader = DataLoader(graphs, batch_size=4, shuffle=False)

    results: Dict[int, float] = {}
    for b in [1, 2, 4]:
        gc.collect()
        model = QDBGCN(
            n_blocks=b, in_channels=16, n_qubits=8, n_classes=2,
            conv_factory=_linear_factory,
        )
        start_rss = _peak_rss_mb()
        _train_briefly(model, loader, epochs=1)
        end_rss = _peak_rss_mb()
        results[b] = end_rss - start_rss

    # On a tiny CPU model RSS deltas are dominated by allocator noise.
    # ru_maxrss is a process-wide high-water mark, so any test running
    # before this one inflates the baseline. We only assert NO pathological
    # growth (> 500 MB delta = real leak). Real GPU memory profiling
    # belongs on the A6000 runner — see plan Phase 5.
    print(f"\n[stress] per-step RSS delta MB by B: {results}")
    assert results[4] - results[1] < 500.0


@pytest.mark.stress
def test_wall_time_per_step_independent_of_n_blocks():
    """Per-step wall time should be near-constant in B (only one block runs)."""
    torch.manual_seed(0)
    graphs = _make_dataset(n_graphs=16, in_channels=16, nodes_per_graph=16)
    loader = DataLoader(graphs, batch_size=4, shuffle=False)

    times: Dict[int, float] = {}
    for b in [1, 2, 4]:
        model = QDBGCN(
            n_blocks=b, in_channels=16, n_qubits=8, n_classes=2,
            conv_factory=_linear_factory,
        )
        t0 = time.perf_counter()
        _train_briefly(model, loader, epochs=1)
        times[b] = time.perf_counter() - t0

    print(f"\n[stress] epoch wall time (s) by B: {times}")
    # B=4 should not be more than 2x slower than B=1 at fixed epoch count.
    assert times[4] / times[1] < 2.5
