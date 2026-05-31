"""Per-block training loop for QDB models with score-matching objective."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm import tqdm
except ImportError:  # tqdm absent in some lightweight envs
    def tqdm(iterable, **kwargs):
        return iterable

from qdb.models import QDBGCN
from qdb.noise_schedule import edm_weight, sample_sigma_in_block


@dataclass
class QDBTrainConfig:
    n_blocks: int = 2
    epochs: int = 100
    lr: float = 5e-3
    weight_decay: float = 1e-3
    batch_size: int = 16
    n_qubits: Optional[int] = None
    n_folds: int = 10
    early_stop_patience: int = 15
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_data: float = 0.5
    algorithm: str = "pruning"  # pruning | dropout | both — no baseline
    seed: int = 42
    # Wall-time guard: abort if a single train_one_step takes longer than
    # this many seconds. Default 600s (10 min). 0 disables. Catches runaway
    # quantum-sim backends without burning runner hours.
    step_timeout_seconds: float = 600.0


def score_matching_loss(
    y_hat: torch.Tensor,
    y_target: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """EDM-weighted MSE between predicted and target embedding.

    y_hat, y_target: [batch_size, dim]
    sigma: [batch_size] (current noise levels)
    Returns scalar loss.
    """
    weight = edm_weight(sigma.detach().cpu().numpy(), sigma_data)
    weight_t = torch.as_tensor(weight, device=y_hat.device, dtype=y_hat.dtype)
    per_sample = ((y_hat - y_target) ** 2).mean(dim=-1)
    return (weight_t * per_sample).mean()


def build_per_block_optimizers(
    model: QDBGCN,
    lr: float,
    weight_decay: float = 0.0,
) -> List[optim.Optimizer]:
    """One AdamW per block. Block independence requires independent optimizer state."""
    opts: List[optim.Optimizer] = []
    for b in range(model.n_blocks):
        params = model.block_parameters(b)
        if not params:
            opts.append(None)  # type: ignore
            continue
        if weight_decay > 0:
            opts.append(optim.AdamW(params, lr=lr, weight_decay=weight_decay))
        else:
            opts.append(optim.Adam(params, lr=lr))
    return opts


def train_one_step(
    model: QDBGCN,
    block_idx: int,
    optimizer: optim.Optimizer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    labels: torch.Tensor,
    config: QDBTrainConfig,
    rng: np.random.Generator,
) -> float:
    """One forward/backward pass on a single block. Returns loss value."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch_size = int(labels.shape[0])
    device = x.device

    # Target y = class embedding lookup
    y = model.class_embedding(labels)  # [batch_size, n_qubits]

    # Sample sigma in this block's range
    sigma_lo, sigma_hi = model.block_sigma_range(block_idx)
    sigma_np = sample_sigma_in_block(
        rng, sigma_lo, sigma_hi, config.p_mean, config.p_std, size=batch_size
    )
    sigma = torch.as_tensor(sigma_np, device=device, dtype=y.dtype)
    log_sigma = torch.log(sigma)

    # Noisy target
    noise = torch.randn_like(y)
    z_sigma = y + sigma.unsqueeze(-1) * noise

    # Forward + loss
    y_hat = model.forward_block(block_idx, x, edge_index, batch, z_sigma, log_sigma)
    loss = score_matching_loss(y_hat, y, sigma, config.sigma_data)

    loss.backward()
    optimizer.step()
    return float(loss.item())


def pick_block(rng: random.Random, n_blocks: int) -> int:
    return rng.randrange(n_blocks)


def evaluate(
    model: QDBGCN,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    """Run denoise + nearest-class on a DataLoader. Returns accuracy + count."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for graph_batch in loader:
            graph_batch = graph_batch.to(device)
            n_graphs = int(graph_batch.num_graphs)
            preds = model.predict(
                graph_batch.x, graph_batch.edge_index, graph_batch.batch, n_graphs
            )
            labels = graph_batch.y.view(-1).long()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    acc = correct / max(total, 1)
    return {"accuracy": acc, "total": total, "correct": correct}


def train_qdb(
    model: QDBGCN,
    train_loader,
    val_loader,
    config: QDBTrainConfig,
    device: torch.device,
    epoch_callback: Optional[Callable[[int, Dict], None]] = None,
    desc: str = "qdb",
) -> Dict:
    """Drive the full training loop: per-step random block selection.

    Args:
        epoch_callback: optional fn(epoch, history) called after each epoch
            for incremental persistence / progress reporting.
        desc: short string used in tqdm progress bar.

    Returns a summary dict with per-block loss curves and val accuracy history.
    """
    optimizers = build_per_block_optimizers(model, config.lr, config.weight_decay)
    rng_py = random.Random(config.seed)
    rng_np = np.random.default_rng(config.seed)

    history = {
        "block_loss": [[] for _ in range(model.n_blocks)],
        "val_accuracy": [],
        "config": vars(config),
    }

    epoch_iter = tqdm(range(1, config.epochs + 1), desc=desc, leave=False)
    for epoch in epoch_iter:
        epoch_block_losses: List[List[float]] = [[] for _ in range(model.n_blocks)]
        for graph_batch in train_loader:
            graph_batch = graph_batch.to(device)
            b = pick_block(rng_py, model.n_blocks)
            if optimizers[b] is None:
                continue
            labels = graph_batch.y.view(-1).long()
            step_t0 = time.perf_counter()
            loss = train_one_step(
                model=model,
                block_idx=b,
                optimizer=optimizers[b],
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                batch=graph_batch.batch,
                labels=labels,
                config=config,
                rng=rng_np,
            )
            step_dt = time.perf_counter() - step_t0
            if config.step_timeout_seconds and step_dt > config.step_timeout_seconds:
                raise TimeoutError(
                    f"train_one_step exceeded {config.step_timeout_seconds:.0f}s "
                    f"(actual {step_dt:.1f}s) at epoch {epoch}, block {b}. "
                    f"Likely a quantum backend regression. Set "
                    f"QDB_QUANTUM_BACKEND=lightning or lower n_qubits/batch_size."
                )
            epoch_block_losses[b].append(loss)
        for b in range(model.n_blocks):
            avg = float(np.mean(epoch_block_losses[b])) if epoch_block_losses[b] else float("nan")
            history["block_loss"][b].append(avg)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            history["val_accuracy"].append(val_metrics["accuracy"])
        # Lightweight per-epoch tqdm summary so the runner log shows progress.
        summary_bits = []
        for b, losses in enumerate(history["block_loss"]):
            if losses:
                summary_bits.append(f"b{b}={losses[-1]:.3f}")
        if history["val_accuracy"]:
            summary_bits.append(f"val={history['val_accuracy'][-1]:.3f}")
        try:
            epoch_iter.set_postfix_str(" ".join(summary_bits))
        except AttributeError:
            # tqdm fallback (plain iterable) doesn't have set_postfix_str.
            print(f"[{desc}] epoch {epoch}: " + " ".join(summary_bits), flush=True)
        if epoch_callback is not None:
            epoch_callback(epoch, history)
    return history
