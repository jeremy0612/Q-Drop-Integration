"""
Quantum Graph Attention Convolution layer.

Architecture per layer:
  1. VQC  — project node features into quantum embedding (per node)
  2. Quantum Attention — HEA circuit produces scalar weight per edge
  3. Classical aggregation — attention-weighted neighbourhood sum

Key differences from the original QGAT repo:
  - Uses default.qubit + backprop (GPU-native via PyTorch autograd) instead of
    lightning.gpu + adjoint (crashes on current cuStateVec/driver combo).
  - Adds LayerNorm + tanh*π pre-processing before every quantum input so the
    angle distribution stays stationary across training.
  - Removes the global device cache (not needed with backprop; each TorchLayer
    owns its qnode closure and re-uses the device implicitly).
"""
import math

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


def _make_device(n_qubits: int):
    """Return (device, diff_method) for the fastest working backend."""
    return qml.device("default.qubit", wires=n_qubits), "backprop"


def _select_qubits(dim: int, max_qubits: int = 8) -> int:
    """Bucket dim to the nearest valid power-of-2 qubit count: 4, 8, or 16."""
    n = min(dim, max_qubits)
    if n > 8:
        return min(16, max_qubits)
    if n > 4:
        return min(8, max_qubits)
    return 4


def _build_vqc(n_qubits: int, n_layers: int) -> nn.Module:
    """Build the variational quantum circuit for node embedding.

    Circuit: AngleEmbedding(Y) → [RY, RZ, CZ-ring] × n_layers → PauliZ expectations.
    Weight shape: (n_layers, n_qubits, 2)  — 2 rotation angles per qubit per layer.
    """
    dev, diff = _make_device(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def _circuit(inputs, q_weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(q_weights[layer, q, 0], wires=q)
                qml.RZ(q_weights[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
            if n_qubits > 2:
                qml.CZ(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return qml.qnn.TorchLayer(_circuit, {"q_weights": (n_layers, n_qubits, 2)})


def _build_attention_circuit(n_qubits: int, n_layers: int) -> nn.Module:
    """Build the HEA attention circuit that maps edge features → scalar.

    Circuit: AngleEmbedding(Y) → [RY, RZ, CNOT-ladder] × n_layers → PauliZ on last qubit.
    Returns a scalar per edge for use as raw attention logit.
    Weight shape: (n_layers * n_qubits * 2 + 1,)  — flat parameter vector.
    """
    dev, diff = _make_device(n_qubits)
    n_params = n_layers * n_qubits * 2 + 1

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def _circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        idx = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[idx], wires=q)
                idx += 1
            for q in range(n_qubits):
                qml.RZ(weights[idx], wires=q)
                idx += 1
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        qml.RY(weights[-1], wires=n_qubits - 1)
        return [qml.expval(qml.PauliZ(n_qubits - 1))]

    return qml.qnn.TorchLayer(_circuit, {"weights": n_params})


class QGATConv(MessagePassing):
    """Quantum Graph Attention Convolution (one layer).

    Args:
        in_channels: Input node feature dimension.
        n_layers: Depth of the VQC and attention circuits.
        n_qubits: Target qubit count (bucketed to 4/8/16, default auto).
        attn_layers: Depth of the attention circuit (default 1).
        dropout: Attention dropout probability.
        max_qubits: Hard cap on qubit count (default 8).
        use_residual: Add skip-connection when input/output dims match (default True).
    """

    def __init__(
        self,
        in_channels: int,
        n_layers: int,
        n_qubits: int = None,
        attn_layers: int = 1,
        dropout: float = 0.0,
        max_qubits: int = 8,
        use_residual: bool = True,
    ):
        super().__init__(aggr="add")
        self.dropout = dropout
        self.use_residual = use_residual

        # ── VQC path ──────────────────────────────────────────────────────────
        self.n_qubits = _select_qubits(
            in_channels if n_qubits is None else n_qubits,
            max_qubits=max_qubits,
        )
        self.in_channels = in_channels

        if in_channels != self.n_qubits:
            self.feature_reduction = Linear(in_channels, self.n_qubits, bias=False)
        else:
            self.feature_reduction = None

        self.vqc_norm = LayerNorm(self.n_qubits)
        self.vqc = _build_vqc(self.n_qubits, n_layers)
        self.bias = Parameter(torch.zeros(self.n_qubits))

        # ── Attention path ────────────────────────────────────────────────────
        attn_input_dim = self.n_qubits * 2
        self.attn_qubits = _select_qubits(attn_input_dim, max_qubits=max_qubits)
        self.attn_reduction = Linear(attn_input_dim, self.attn_qubits, bias=False)
        self.attn_norm = LayerNorm(self.attn_qubits)
        self.attn_circuit = _build_attention_circuit(self.attn_qubits, attn_layers)
        self.attn_readout = Linear(1, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.feature_reduction is not None:
            nn.init.xavier_uniform_(self.feature_reduction.weight)
        nn.init.xavier_uniform_(self.attn_reduction.weight)
        nn.init.xavier_uniform_(self.attn_readout.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Feature projection + normalize + bound for quantum angle embedding
        if self.feature_reduction is not None:
            x_q = self.feature_reduction(x)
        else:
            x_q = x

        x_q = self.vqc_norm(x_q)
        x_q = torch.tanh(x_q) * math.pi

        # VQC: quantum node embedding
        node_emb = self.vqc(x_q).float()
        node_emb = torch.nan_to_num(node_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # Message passing with quantum attention
        out = self.propagate(edge_index, x=node_emb)
        out = out + self.bias

        if self.use_residual and node_emb.shape == out.shape:
            out = out + node_emb

        return out

    def message(self, x_i, x_j, index, ptr, size_i):
        # Concatenate target and source embeddings → attention input
        x_cat = torch.cat([x_i, x_j], dim=-1)

        # Project + normalize + bound for attention circuit input
        x_attn = self.attn_reduction(x_cat)
        x_attn = self.attn_norm(x_attn)
        x_attn = torch.tanh(x_attn) * math.pi

        # Quantum attention: HEA circuit → scalar per edge
        alpha = self.attn_circuit(x_attn).float()
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)

        alpha = self.attn_readout(alpha)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha * x_j
