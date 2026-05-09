import torch
from torch.nn import Module, ModuleList, Linear, LeakyReLU, Dropout, LayerNorm
from torch_geometric.nn import global_mean_pool

try:
    from .QATN_Layers import QGATConv
except ImportError:
    from QATN_Layers import QGATConv


class QGAT(Module):
    """
    Quantum Graph Attention Network (QATN).

    Architecture: Input → [QGATConv × L] → GlobalMeanPool → Linear → Output
    Each QGATConv layer uses: VQC (RY/RZ+CZ) → HEA Attention → Aggregation.
    """

    def __init__(
        self,
        input_dims: int,
        q_depths: list,
        output_dims: int,
        attn_model: str = "HEA",
        activ_fn=None,
        dropout: float = 0.2,
        readout: bool = False,
        max_qubits: int = 8,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if activ_fn is None:
            activ_fn = LeakyReLU(0.2)

        layers = []
        current_dim = input_dims
        for q_depth in q_depths:
            layer = QGATConv(
                in_channels=current_dim,
                n_layers=q_depth,
                attn_model=attn_model,
                dropout=dropout,
                residual=True,
                max_qubits=max_qubits,
                use_layer_norm=use_layer_norm,
            )
            layers.append(layer)
            current_dim = layer.n_qubits

        self.layers = ModuleList(layers)
        self.embedding_dim = current_dim
        self.activ_fn = activ_fn
        self.dropout = Dropout(p=dropout)
        self.output_norm = LayerNorm(self.embedding_dim) if use_layer_norm else None
        self.readout = Linear(1, 1) if readout else None
        self.classifier = Linear(self.embedding_dim, output_dims)

    def forward(self, x, edge_index, batch):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.activ_fn(h)
            h = self.dropout(h)

        h = global_mean_pool(h, batch)
        if self.output_norm is not None:
            h = self.output_norm(h)
        h = self.classifier(h)
        if self.readout is not None:
            h = self.readout(h)
        return h
