"""
Quantum Graph Attention Network (QATN) Layer.

Three-part design:
  1. VQC  – transform node features via variational quantum circuit (RY/RZ + CZ ring)
  2. Quantum Attention – compute edge attention weights via HEA circuit (RY/RZ + CNOT ladder)
  3. Classical Aggregation – softmax-weighted neighbour aggregation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter, LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import pennylane as qml
from pennylane.exceptions import DeviceError

try:
    from ..QNN_Node_Embedding_QGAT import quantum_net
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from QNN_Node_Embedding_QGAT import quantum_net

_ATTN_DEVICE_CACHE = {}


def _get_attn_device(n_qubits: int):
    if n_qubits in _ATTN_DEVICE_CACHE:
        return _ATTN_DEVICE_CACHE[n_qubits]
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits, shots=None)
    except (DeviceError, ImportError, Exception):
        dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)
    _ATTN_DEVICE_CACHE[n_qubits] = dev
    return dev


def HEA_Attention(n_qubits: int, n_layers: int = 1):
    """HEA attention circuit: RY/RZ rotations + CNOT ladder entanglement."""
    dev = _get_attn_device(n_qubits)
    n_params = n_layers * n_qubits * 2 + 1

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        idx = 0
        for _ in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[idx], wires=i)
                idx += 1
            for i in range(n_qubits):
                qml.RZ(weights[idx], wires=i)
                idx += 1
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        qml.RY(weights[-1], wires=n_qubits - 1)
        return [qml.expval(qml.PauliZ(n_qubits - 1))]

    return qml.qnn.TorchLayer(circuit, {"weights": n_params}), circuit


class QGATConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        n_layers: int,
        attn_model: str = "HEA",
        n_qubits: int = None,
        attn_qubits: int = None,
        dropout: float = 0.0,
        residual: bool = True,
        max_qubits: int = 8,
        attn_layers: int = 1,
        use_layer_norm: bool = True,
    ):
        super().__init__(aggr="add")
        self.dropout = dropout
        self.residual = residual
        self.max_qubits = max_qubits
        self.use_layer_norm = use_layer_norm

        self.n_qubits = self._select_qubits(in_channels if n_qubits is None else n_qubits, max_qubits)
        self.in_channels = in_channels

        self.feature_reduction = (
            Linear(in_channels, self.n_qubits, bias=False) if in_channels != self.n_qubits else None
        )
        self.vqc = quantum_net(self.n_qubits, n_layers, max_qubits=max_qubits)
        self.bias = Parameter(torch.empty(self.n_qubits))
        self.layer_norm = LayerNorm(self.n_qubits) if use_layer_norm else None

        attn_input_dim = self.n_qubits * 2
        self.attn_qubits = self._select_qubits(
            attn_input_dim if attn_qubits is None else attn_qubits, max_qubits
        )
        self.attn_feature_reduction = Linear(attn_input_dim, self.attn_qubits, bias=False)

        if attn_model != "HEA":
            raise ValueError(f"Unknown attn_model: {attn_model}. Only 'HEA' is supported.")
        self.quantum_attention, _ = HEA_Attention(self.attn_qubits, n_layers=attn_layers)
        self.attn_readout = Linear(1, 1)

        self.reset_parameters()

    @staticmethod
    def _select_qubits(dim: int, max_qubits: int = 8) -> int:
        n = min(dim, max_qubits)
        if n > 8:
            return min(16, max_qubits)
        if n > 4:
            return min(8, max_qubits)
        return 4

    def reset_parameters(self):
        nn.init.zeros_(self.bias)
        if self.feature_reduction is not None:
            nn.init.xavier_uniform_(self.feature_reduction.weight)
        nn.init.xavier_uniform_(self.attn_feature_reduction.weight)
        nn.init.xavier_uniform_(self.attn_readout.weight)
        if self.layer_norm is not None:
            self.layer_norm.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x_reduced = self.feature_reduction(x) if self.feature_reduction is not None else x
        h = self.vqc(x_reduced).float()
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        out = self.propagate(edge_index, x=h) + self.bias
        if self.residual and h.size(-1) == out.size(-1):
            out = out + h
        return out

    def message(self, x_i, x_j, index, ptr, size_i):
        x_attn = self.attn_feature_reduction(torch.cat((x_i, x_j), dim=-1))
        alpha = self.quantum_attention(x_attn).float()
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        alpha = self.attn_readout(alpha)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha * x_j
