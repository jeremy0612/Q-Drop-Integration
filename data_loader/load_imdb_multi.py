"""
IMDB-MULTI dataset loader via PyTorch Geometric.

1,500 social ego-networks, 3-class classification (Comedy / Romance / Sci-Fi).
No node features — node degree is one-hot encoded (89 classes, matching GRDL).
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree

_ROOT = "/tmp/pyg_data"
_MAX_DEGREE = 88  # one-hot width = 89, matching GRDL Transform_imdbm


def load_imdb_multi(root: str = _ROOT) -> list[Data]:
    """Load IMDB-MULTI with degree one-hot node features (dim=89)."""
    raw = TUDataset(root=root, name="IMDB-MULTI")
    return [_add_degree_feature(g) for g in raw]


def _add_degree_feature(g: Data) -> Data:
    num_nodes = g.num_nodes
    deg = degree(g.edge_index[0], num_nodes=num_nodes, dtype=torch.long)
    x = F.one_hot(deg.clamp(max=_MAX_DEGREE), num_classes=_MAX_DEGREE + 1).float()
    return Data(x=x, edge_index=g.edge_index, y=g.y, num_nodes=num_nodes)
