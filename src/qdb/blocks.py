"""QDB diffusion blocks wrapping a graph-conv module.

Each block fuses node features with a per-graph noisy latent ``z_sigma``,
runs the wrapped graph conv, applies AdaLN modulation conditioned on
``log_sigma``, and pools to a per-graph denoised prediction.

The pre-projection's z half is zero-initialized so the block is an identity
function of ``z_sigma`` at construction time — combined with AdaLN's zero
init, the block initially predicts what the conv would output on ``x``
alone. This is critical for stable diffusion training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from qdb.adaln import AdaLNModulation


class QDBGCNBlock(nn.Module):
    """QDB block wrapping an arbitrary graph-conv module.

    Args:
        conv_module: an ``nn.Module`` mapping
            ``(x [N, in_channels], edge_index [2, E]) -> [N, n_qubits]``.
        in_channels: input node feature dimension expected by ``conv_module``.
        n_qubits: latent/output dimension produced by ``conv_module``.
    """

    def __init__(
        self,
        conv_module: nn.Module,
        in_channels: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_qubits = n_qubits
        self.conv_module = conv_module

        # Fuse (x || z_per_node) -> in_channels.
        self.pre_projection = nn.Linear(
            in_channels + n_qubits, in_channels, bias=False
        )
        self._init_pre_projection()

        # AdaLN modulates the conv output (n_qubits-dim).
        self.adaln = AdaLNModulation(n_qubits)

    def _init_pre_projection(self) -> None:
        """Init the x half with orthogonal weights; zero out the z half.

        ``pre_projection.weight`` has shape ``[in_channels, in_channels + n_qubits]``.
        Slice ``[:, :in_channels]`` is the x half; ``[:, in_channels:]`` is the z half.
        """
        with torch.no_grad():
            x_half = self.pre_projection.weight[:, : self.in_channels]
            nn.init.orthogonal_(x_half)
            self.pre_projection.weight[:, self.in_channels :].zero_()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        z_sigma: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Run one QDB block.

        Args:
            x: ``[N, in_channels]`` node features.
            edge_index: ``[2, E]`` graph connectivity.
            batch: ``[N]`` long tensor mapping each node to a graph index.
            z_sigma: ``[batch_size, n_qubits]`` noisy per-graph latent.
            log_sigma: scalar or ``[batch_size]`` log noise level.

        Returns:
            ``[batch_size, n_qubits]`` denoised per-graph prediction.
        """
        # 1. Broadcast z_sigma to nodes via batch index.
        z_per_node = z_sigma.index_select(0, batch)  # [N, n_qubits]

        # 2. Fuse and project back to in_channels.
        fused = torch.cat([x, z_per_node], dim=-1)  # [N, in_channels + n_qubits]
        x_fused = self.pre_projection(fused)         # [N, in_channels]

        # 3. Wrapped graph conv.
        h = self.conv_module(x_fused, edge_index)    # [N, n_qubits]

        # 4. AdaLN modulation conditioned on log_sigma.
        h = self.adaln(h, log_sigma, batch)          # [N, n_qubits]

        # 5. Per-graph mean pool.
        return global_mean_pool(h, batch)            # [batch_size, n_qubits]


class QDBGATBlock(QDBGCNBlock):
    """QDB block wrapping a GAT-style conv module.

    Exposed as a distinct class so callers can request it by name. The
    forward contract is identical to :class:`QDBGCNBlock`; only the wrapped
    conv differs in practice.
    """

    pass
