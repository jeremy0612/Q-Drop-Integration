"""
PROTEINS dataset loader.

Primary path: HuggingFace ``graphs-datasets/PROTEINS``.
Fallback path: PyG ``torch_geometric.datasets.TUDataset(name="PROTEINS")``.

The fallback exists because the HuggingFace ``datasets`` library has shipped
cache-incompatible schema changes that crash ``Features.from_dict`` with
``TypeError: must be called with a dataclass type or instance`` when an
older cache is read by a newer client.

Dataset: graphs-datasets/PROTEINS
  - 1,113 protein graphs
  - Binary: enzyme (1) vs non-enzyme (0)
  - Node features: 3-dim (degree + 2 biochemical attributes)
"""

from __future__ import annotations

import os
import warnings
from typing import List

import torch
from torch_geometric.data import Data


_HF_CACHE = os.environ.get(
    "HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")
)


def load_proteins(cache_dir: str = _HF_CACHE) -> List[Data]:
    """Load PROTEINS. HF first, PyG TUDataset fallback."""
    try:
        from datasets import load_dataset

        raw = load_dataset("graphs-datasets/PROTEINS", cache_dir=cache_dir)
        return _convert(raw)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"HuggingFace PROTEINS loader failed ({type(exc).__name__}: {exc}); "
            "falling back to torch_geometric.datasets.TUDataset.",
            stacklevel=2,
        )
        return _load_tudataset_proteins(cache_dir)


def _load_tudataset_proteins(cache_dir: str) -> List[Data]:
    """Native PyG loader. PROTEINS: 1,113 graphs, 3-dim node features."""
    from torch_geometric.datasets import TUDataset

    root = os.path.join(cache_dir, "TUDataset")
    os.makedirs(root, exist_ok=True)
    ds = TUDataset(root=root, name="PROTEINS")
    graphs: List[Data] = []
    for g in ds:
        y = g.y.view(-1).long() if g.y is not None else torch.tensor([0], dtype=torch.long)
        graphs.append(
            Data(
                x=g.x.float() if g.x is not None else None,
                edge_index=g.edge_index,
                edge_attr=g.edge_attr.float() if g.edge_attr is not None else None,
                y=y[:1],
                num_nodes=g.num_nodes,
            )
        )
    return graphs


def _convert(raw) -> list:
    graphs = []
    for split in raw.values():
        for item in split:
            node_feat = item.get('node_feat')
            if node_feat is not None:
                x = torch.tensor(node_feat, dtype=torch.float)
            else:
                # Fallback: use degree as single feature if node_feat absent
                num_nodes = item.get('num_nodes', 1)
                x = torch.ones(num_nodes, 1, dtype=torch.float)

            edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
            if edge_index.dim() == 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.t().contiguous()

            edge_attr = (torch.tensor(item['edge_attr'], dtype=torch.float)
                         if item.get('edge_attr') is not None else None)

            y_val = item['y']
            if isinstance(y_val, list):
                y_val = y_val[0]
            y = torch.tensor([int(y_val)], dtype=torch.long)

            graphs.append(Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                num_nodes=item.get('num_nodes', x.size(0)),
            ))
    return graphs
