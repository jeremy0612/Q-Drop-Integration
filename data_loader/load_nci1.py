"""
NCI1 dataset loader via PyTorch Geometric.

4,110 chemical compound graphs, binary classification (active vs inactive
against non-small-cell lung cancer).
Node features: 37-dim one-hot atom type encoding.
"""

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset


_ROOT = "/tmp/pyg_data"


def load_nci1(root: str = _ROOT) -> list[Data]:
    """Load NCI1; returns graphs with 37-dim node features."""
    raw = TUDataset(root=root, name="NCI1")
    return list(raw)
