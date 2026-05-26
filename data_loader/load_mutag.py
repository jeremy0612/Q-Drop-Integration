"""
MUTAG dataset loader from HuggingFace.
Converts to PyTorch Geometric Data objects.
Caches to ~/.cache/huggingface/datasets/ (default HF behavior).
"""

import os
import torch
from torch_geometric.data import Data
from datasets import load_dataset


_HF_CACHE = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))


def load_mutag(cache_dir: str = _HF_CACHE):
    """
    Load MUTAG from HuggingFace (cached after first download).
    Returns: list of torch_geometric.data.Data (187 graphs)
    """
    raw = load_dataset("graphs-datasets/MUTAG", cache_dir=cache_dir)
    return _convert(raw)


def _convert(raw) -> list:
    graphs = []
    for split in raw.values():
        for item in split:
            x = torch.tensor(item['node_feat'], dtype=torch.float)

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
