import torch
import torch.nn as nn
from torch.nn import LayerNorm, Dropout, LeakyReLU, Linear, Module, ModuleList, ReLU, Sequential
from torch.nn.init import orthogonal_
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

try:
    from .gcn_conv_layers import QGCNConv
except ImportError:
    from gcn_conv_layers import QGCNConv


class QGCN(Module):
    """QGCN with optional multi-scale pooling, MLP head, and residual connections."""

    def __init__(
        self,
        input_dims,
        q_depths,
        output_dims,
        activ_fn=LeakyReLU(0.2),
        classifier=None,
        readout=False,
        n_qubits=None,
        pool_type="mean",
        use_mlp_head=False,
        mlp_hidden=64,
        mlp_dropout=0.5,
        use_residual=False,
        use_strongly_entangling=False,
        weight_init=None,
    ):
        super().__init__()
        max_qubits = 16
        if n_qubits is None:
            n_qubits = min(input_dims, max_qubits)
        else:
            n_qubits = min(n_qubits, max_qubits)

        if n_qubits > 8:
            n_qubits = 16
        else:
            n_qubits = 8
        self.n_qubits = n_qubits
        self.pool_type = pool_type
        self.use_residual = use_residual

        layers = []
        for index, q_depth in enumerate(q_depths):
            layer_input_dims = input_dims if index == 0 else n_qubits
            layers.append(QGCNConv(
                layer_input_dims, q_depth,
                n_qubits=n_qubits,
                use_strongly_entangling=use_strongly_entangling,
                weight_init=weight_init,
            ))

        self.layers = ModuleList(layers)
        self.activ_fn = activ_fn

        if readout:
            self.readout = Linear(1, 1)
        else:
            self.readout = None

        pool_factor = 3 if pool_type == "multiscale" else 1
        pool_out_dims = n_qubits * pool_factor

        if use_mlp_head:
            half_hidden = max(mlp_hidden // 2, output_dims)
            self.classifier = Sequential(
                Linear(pool_out_dims, mlp_hidden),
                LayerNorm(mlp_hidden),
                ReLU(),
                Dropout(mlp_dropout),
                Linear(mlp_hidden, half_hidden),
                LayerNorm(half_hidden),
                ReLU(),
                Dropout(mlp_dropout * 0.5),
                Linear(half_hidden, output_dims),
            )
        else:
            self.classifier = Linear(pool_out_dims, output_dims)
            orthogonal_(self.classifier.weight)

    def qdrop_layers(self):
        quantum_layers = []
        for layer_index, layer in enumerate(self.layers):
            quantum_layer = getattr(layer, "quantum_layer", None)
            if quantum_layer is None:
                continue
            quantum_layer.qdrop_name = f"layers.{layer_index}.quantum_layer"
            quantum_layers.append(quantum_layer)
        return quantum_layers

    def forward(self, x, edge_index, batch):
        h = x
        for idx, layer in enumerate(self.layers):
            h_new = layer(h, edge_index)
            h_new = self.activ_fn(h_new)
            if self.use_residual and idx > 0:
                h = h + h_new
            else:
                h = h_new

        if self.pool_type == "multiscale":
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h_add = global_add_pool(h, batch)
            h = torch.cat([h_mean, h_max, h_add], dim=1)
        else:
            h = global_mean_pool(h, batch)

        h = self.classifier(h)

        if self.readout is not None:
            h = self.readout(h)

        return h
