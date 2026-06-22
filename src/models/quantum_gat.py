"""Quantum Graph Attention Network (QGAT) for graph classification."""

import torch
import torch.nn as nn
from torch.nn import (
    Dropout, LayerNorm, LeakyReLU, Linear,
    Module, ModuleList, ReLU, Sequential,
)
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

try:
    from .gat_conv_layers import QGATConv
except ImportError:
    from gat_conv_layers import QGATConv


class QGAT(Module):
    """Quantum Graph Attention Network.

    Stacks ``len(q_depths)`` QGATConv layers, pools node embeddings to a
    graph-level vector, then classifies with a Linear or MLP head.

    Args:
        input_dims: Node feature dimension.
        q_depths: Sequence of per-layer VQC depths, e.g. [2, 2].
        output_dims: Number of output logits (1 for binary, C for multi-class).
        attn_dropout: Dropout on attention coefficients inside each QGATConv.
        layer_dropout: Dropout applied between QGATConv layers.
        max_qubits: Hard qubit cap passed through to QGATConv (default 8).
        pool_type: "mean" (default) or "multiscale" (mean + max + add concatenated).
        use_mlp_head: Replace the single Linear classifier with a 2-layer MLP + BN.
        mlp_hidden: Hidden dimension of the MLP head (used when use_mlp_head=True).
        mlp_dropout: Dropout rate inside the MLP head.
        use_residual: Enable skip-connections inside each QGATConv.
    """

    def __init__(
        self,
        input_dims: int,
        q_depths,
        output_dims: int,
        attn_dropout: float = 0.2,
        layer_dropout: float = 0.2,
        max_qubits: int = 8,
        pool_type: str = "multiscale",
        use_mlp_head: bool = True,
        mlp_hidden: int = 64,
        mlp_dropout: float = 0.3,
        use_residual: bool = True,
    ):
        super().__init__()
        self.pool_type = pool_type

        layers = []
        current_dim = input_dims
        for q_depth in q_depths:
            conv = QGATConv(
                in_channels=current_dim,
                n_layers=q_depth,
                dropout=attn_dropout,
                max_qubits=max_qubits,
                use_residual=use_residual,
            )
            layers.append(conv)
            current_dim = conv.n_qubits

        self.layers = ModuleList(layers)
        self.embedding_dim = current_dim
        self.activ_fn = LeakyReLU(0.2)
        self.layer_drop = Dropout(p=layer_dropout)
        self.output_norm = LayerNorm(self.embedding_dim)

        pool_factor = 3 if pool_type == "multiscale" else 1
        pool_out = self.embedding_dim * pool_factor

        if use_mlp_head:
            half = max(mlp_hidden // 2, output_dims)
            self.classifier = Sequential(
                Linear(pool_out, mlp_hidden),
                LayerNorm(mlp_hidden),
                ReLU(),
                Dropout(mlp_dropout),
                Linear(mlp_hidden, half),
                LayerNorm(half),
                ReLU(),
                Dropout(mlp_dropout * 0.5),
                Linear(half, output_dims),
            )
        else:
            self.classifier = Linear(pool_out, output_dims)

    def qdrop_layers(self):
        """Q-Drop interface: expose each QGATConv's node VQC adapter.

        Each adapter wraps the VQC rotation angles so the shared Q-Drop
        runtime can statistic-prune / wire-dropout them, mirroring QGCN.
        """
        quantum_layers = []
        for layer_index, conv in enumerate(self.layers):
            vqc = getattr(conv, "vqc", None)
            if vqc is None or not hasattr(vqc, "qdrop_layer_spec"):
                continue
            vqc.qdrop_name = f"layers.{layer_index}.vqc"
            quantum_layers.append(vqc)
        return quantum_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.activ_fn(h)
            h = self.layer_drop(h)

        h = self.output_norm(h)

        if self.pool_type == "multiscale":
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h_add = global_add_pool(h, batch)
            h = torch.cat([h_mean, h_max, h_add], dim=1)
        else:
            h = global_mean_pool(h, batch)

        return self.classifier(h)
