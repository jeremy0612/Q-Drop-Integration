import torch
from torch.nn import Linear, LeakyReLU, Module, ModuleList
from torch_geometric.nn import global_mean_pool

try:
    from .gcn_conv_layers import QGCNConv
except ImportError:
    from gcn_conv_layers import QGCNConv


class QGCN(Module):
    """QGCN with Linear classifier only."""

    def __init__(
        self,
        input_dims,
        q_depths,
        output_dims,
        activ_fn=LeakyReLU(0.2),
        classifier=None,
        readout=False,
        n_qubits=None,
    ):
        super().__init__()
        layers = []
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

        for index, q_depth in enumerate(q_depths):
            layer_input_dims = input_dims if index == 0 else n_qubits
            qgcn_conv = QGCNConv(layer_input_dims, q_depth, n_qubits=n_qubits)
            layers.append(qgcn_conv)

        self.layers = ModuleList(layers)
        self.activ_fn = activ_fn

        if readout:
            self.readout = Linear(1, 1)
        else:
            self.readout = None

        self.classifier = Linear(self.n_qubits, output_dims)

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
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.activ_fn(h)

        h = global_mean_pool(h, batch)
        h = self.classifier(h)

        if self.readout is not None:
            h = self.readout(h)

        return h
