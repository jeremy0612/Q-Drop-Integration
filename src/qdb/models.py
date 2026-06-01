"""QDB model containers: B independently-trainable blocks + class embedding."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from qdb.blocks import QDBGCNBlock, QDBGATBlock
from qdb.class_embedding import ClassEmbedding
from qdb.noise_schedule import equiprob_boundaries


def _edm_preconditioning(
    sigma: torch.Tensor, sigma_data: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Karras 2022 EDM preconditioning coefficients.

    Without these, the network sees inputs whose scale spans ~80 at sigma=80
    down to ~1e-3 at sigma=0.002. No LayerNorm composition can absorb that
    dynamic range, so the block degenerates to a near-constant predictor
    and validation accuracy parks at chance. With preconditioning, F_theta
    always sees roughly unit-norm inputs and emits a unit-norm correction.

    Returns (c_skip, c_out, c_in, c_noise) shaped like sigma.
    """
    sigma2 = sigma ** 2
    denom = sigma2 + sigma_data ** 2
    c_skip = (sigma_data ** 2) / denom
    c_out = (sigma * sigma_data) / torch.sqrt(denom)
    c_in = 1.0 / torch.sqrt(denom)
    c_noise = 0.25 * torch.log(sigma.clamp_min(1e-8))
    return c_skip, c_out, c_in, c_noise


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
        sigma_data: float = 0.5,
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
        self.sigma_data = sigma_data

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
        """Apply a single block with EDM preconditioning.

        Wraps the raw block as:
            D_theta(x, z_sigma, sigma) = c_skip*z_sigma + c_out*F_theta(x, c_in*z_sigma, c_noise)
        where (c_skip, c_out, c_in, c_noise) come from Karras 2022 eq.(8).

        Returns [batch_size, n_qubits] denoised prediction.
        """
        sigma = torch.exp(log_sigma)
        if sigma.dim() == 0:
            sigma = sigma.expand(z_sigma.shape[0])
        c_skip, c_out, c_in, c_noise = _edm_preconditioning(sigma, self.sigma_data)
        # Reshape coefficients for broadcasting against [batch_size, n_qubits].
        c_skip_b = c_skip.view(-1, 1)
        c_out_b = c_out.view(-1, 1)
        c_in_b = c_in.view(-1, 1)
        scaled_z = c_in_b * z_sigma
        f_out = self.blocks[block_idx](x, edge_index, batch, scaled_z, c_noise)
        return c_skip_b * z_sigma + c_out_b * f_out

    @torch.no_grad()
    def denoise(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Inference: Euler steps from sigma_max to sigma_min.

        Args:
            inference_steps: total number of Euler steps. Default 4*n_blocks.
                Each step picks the block whose [sigma_b, sigma_{b-1}] range
                contains the current sigma. More steps = finer refinement at
                the cost of B-fold-additional forward passes per step.

        Returns z_T of shape [batch_size, n_qubits], the final denoised latent.
        """
        device = next(self.parameters()).device
        if inference_steps is None:
            inference_steps = max(self.n_blocks * 4, 4)

        # Geometric schedule of sigma values from sigma_max down to sigma_min.
        sigmas = torch.logspace(
            float(np.log10(self.sigma_max)),
            float(np.log10(self.sigma_min)),
            inference_steps + 1,
            device=device,
            dtype=torch.float32,
        )

        z = torch.randn(
            batch_size, self.n_qubits, device=device, generator=generator
        ) * self.sigma_max

        boundaries = self.sigma_boundaries  # descending [sigma_max, ..., sigma_min]
        for i in range(inference_steps):
            sigma_now = float(sigmas[i].item())
            sigma_next = float(sigmas[i + 1].item())
            # Pick block whose [low, high] range contains sigma_now.
            block_idx = self.n_blocks - 1
            for b in range(self.n_blocks):
                hi = float(boundaries[b].item())
                lo = float(boundaries[b + 1].item())
                if lo <= sigma_now <= hi:
                    block_idx = b
                    break
            log_sigma = torch.full(
                (batch_size,), float(np.log(max(sigma_now, 1e-8))), device=device
            )
            y_hat = self.forward_block(block_idx, x, edge_index, batch, z, log_sigma)
            # Reverse-ODE Euler step: z_next = z + (delta / sigma_now) * (y_hat - z)
            delta = sigma_now - sigma_next
            z = z + (delta / max(sigma_now, 1e-8)) * (y_hat - z)
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
