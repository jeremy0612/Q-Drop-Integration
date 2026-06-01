import math

import torch
from torch.nn import LayerNorm, Linear, Parameter
from torch.nn.utils.parametrizations import orthogonal as orthogonal_parametrization
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

try:
    from ..qnn_node_embedding import quantum_net
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from qnn_node_embedding import quantum_net


class QGCNConv(MessagePassing):
    def __init__(self, in_channels, n_layers, n_qubits=None, use_strongly_entangling=False, weight_init=None):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        # Limit qubits to at most 16 for practical quantum simulation
        # Ensure it's a power of 2
        import numpy as np
        if n_qubits is None:
            n_qubits = min(in_channels, 16)
        else:
            n_qubits = min(n_qubits, 16)
        
        # Ensure power of 2
        if n_qubits > 8:
            n_qubits = 16
        elif n_qubits > 4:
            n_qubits = 8
        elif n_qubits > 2:
            n_qubits = 4
        else:
            n_qubits = 2
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.in_channels = in_channels
        
        # Add linear layer to reduce dimensions if needed
        if in_channels != n_qubits:
            self.feature_reduction = Linear(in_channels, n_qubits, bias=False)
        else:
            self.feature_reduction = None

        # LayerNorm before the tanh*pi bound. The pre-tanh logits drift in
        # scale across epochs (especially when Q-Drop masks reshape the
        # effective circuit), so tanh oscillates between linear and
        # saturated regimes and the quantum input distribution becomes
        # non-stationary. Normalizing across the qubit axis stabilizes the
        # angle distribution fed into BasicEntanglerLayers without changing
        # tensor shape.
        self.input_norm = LayerNorm(n_qubits)

        self.quantum_layer = quantum_net(
            self.n_qubits,
            self.n_layers,
            use_strongly_entangling=use_strongly_entangling,
            weight_init=weight_init,
        )
        self.qc = self.quantum_layer
        self.bias = Parameter(torch.empty(n_qubits))
        self.reset_parameters()

        # Cayley-parametrized orthogonality on the quantum input projection.
        # Forces feature_reduction.weight to remain semi-orthogonal across
        # training, so the projection from in_channels -> n_qubits stays
        # near-isometric and the angle distribution fed into the quantum
        # circuit keeps roughly unit-norm energy per direction. Registered
        # after reset_parameters so the underlying tensor is initialized
        # first and the parametrization just projects it.
        if self.feature_reduction is not None:
            orthogonal_parametrization(
                self.feature_reduction,
                name="weight",
                orthogonal_map="cayley",
            )

    def reset_parameters(self):
        self.bias.data.zero_()
        if self.feature_reduction is not None:
            self.feature_reduction.reset_parameters()
        self.input_norm.reset_parameters()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Reduce feature dimensions if needed
        if self.feature_reduction is not None:
            x_reduced = self.feature_reduction(x)
        else:
            x_reduced = x

        # Stabilize the quantum input distribution before bounding it.
        # LayerNorm centers and scales per-node across the qubit axis so
        # tanh stays in its informative range epoch-to-epoch.
        x_reduced = self.input_norm(x_reduced)

        # Bound quantum-circuit input to a well-defined angle range so
        # BasicEntanglerLayers rotations stay within [-pi, pi] regardless of
        # the upstream feature scale. Unbounded inputs wrap the rotation
        # angles and turn the quantum gradient surface chaotic, which shows
        # up downstream as simultaneous gradient spikes and underfitting.
        x_reduced = torch.tanh(x_reduced) * math.pi

        # Step 3: Apply quantum circuit
        q_out = self.quantum_layer(x_reduced).float()

        # Step 4: Compute normalization.
        row, col = edge_index
        deg = degree(col, q_out.size(0), dtype=q_out.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 5: Start propagating messages.
        out = self.propagate(edge_index, x=q_out, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Normalize node features.
        return norm.view(-1, 1) * x_j
