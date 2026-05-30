"""QDB model containers: B independently-trainable blocks + class embedding."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from qdb.blocks import QDBGCNBlock, QDBGATBlock
from qdb.class_embedding import ClassEmbedding
from qdb.noise_schedule import equiprob_boundaries


class QDBGCN(nn.Module):
    """Stack of ``n_blocks`` QDB-wrapped graph-conv blocks.

    Each block is independent (no shared parameters across blocks) and
    handles its own slice of the noise schedule. Training samples one
    block per step; inference walks all blocks sequentially via Euler.
    """

    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        n_qubits: int,
        n_classes: int,
        conv_factory: Callable[[int, int], nn.Module],
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        freeze_class_norm: bool = True,
        block_cls: type = QDBGCNBlock,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std

        boundaries = equiprob_boundaries(n_blocks, sigma_min, sigma_max, p_mean, p_std)
        # boundaries shape (n_blocks+1,), descending: [sigma_max, ..., sigma_min]
        self.register_buffer("sigma_boundaries", torch.as_tensor(boundaries, dtype=torch.float32))

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            conv = conv_factory(in_channels, n_qubits)
            self.blocks.append(block_cls(conv, in_channels, n_qubits))

        self.class_embedding = ClassEmbedding(n_classes, n_qubits, freeze_norm=freeze_class_norm)

    def block_sigma_range(self, block_idx: int) -> tuple[float, float]:
        """Return (sigma_low, sigma_high) ascending for the block."""
        if not 0 <= block_idx < self.n_blocks:
            raise IndexError(f"block_idx {block_idx} out of range [0, {self.n_blocks})")
        hi = float(self.sigma_boundaries[block_idx].item())
        lo = float(self.sigma_boundaries[block_idx + 1].item())
        return lo, hi

    def forward_block(
        self,
        block_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        z_sigma: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a single block. Returns [batch_size, n_qubits] denoised pred."""
        return self.blocks[block_idx](x, edge_index, batch, z_sigma, log_sigma)

    @torch.no_grad()
    def denoise(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Inference: Euler-step from sigma_max to sigma_min through all blocks.

        Returns z_T of shape [batch_size, n_qubits], the final denoised latent.
        """
        device = next(self.parameters()).device
        z = torch.randn(
            batch_size, self.n_qubits, device=device, generator=generator
        ) * self.sigma_max
        # Walk blocks high -> low noise.
        for b in range(self.n_blocks):
            sigma_hi = float(self.sigma_boundaries[b].item())
            sigma_lo = float(self.sigma_boundaries[b + 1].item())
            sigma_mid = float(np.sqrt(sigma_hi * sigma_lo))  # geometric mid
            log_sigma = torch.full(
                (batch_size,), float(np.log(sigma_mid)), device=device
            )
            y_hat = self.forward_block(b, x, edge_index, batch, z, log_sigma)
            # Reverse-ODE Euler step (probability-flow): going from sigma_hi to
            # sigma_lo (delta > 0), z drifts toward the denoised prediction y_hat:
            #   z_next = z + (delta / sigma_hi) * (y_hat - z)
            # Wrong sign here makes inference diverge to noise.
            delta = sigma_hi - sigma_lo
            z = z + (delta / max(sigma_hi, 1e-8)) * (y_hat - z)
        return z

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return predicted class indices [batch_size] via nearest-class on z_T."""
        z_t = self.denoise(x, edge_index, batch, batch_size)
        return self.class_embedding.nearest_class(z_t)

    def block_parameters(self, block_idx: int) -> List[nn.Parameter]:
        """Parameters belonging only to one block. Excludes class_embedding."""
        return list(self.blocks[block_idx].parameters())


class QDBGAT(QDBGCN):
    """QDB stack over QGAT-style conv blocks. Same API; different default block_cls."""

    def __init__(self, *args, block_cls: type = QDBGATBlock, **kwargs) -> None:
        super().__init__(*args, block_cls=block_cls, **kwargs)
