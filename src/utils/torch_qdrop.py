"""Backward-compatible exports for Torch-side notebook helpers."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import List, Tuple

from qdrop import QDropDropoutState, QDropLayerSpec, QDropSpecFactory, QDropTensorSpec, SupportsQDropSpec
from qdrop.backends.torch_runtime import TorchQDropRuntime
from qdrop.types import QDropConfig

QuantumDropCompatible = SupportsQDropSpec
QuantumDropoutState = QDropDropoutState


@dataclass
class TorchQDropConfig:
    """Backward-compatible config with old field names used by training scripts.

    Translates to QDropConfig internally:  drop_prob → dropout_prob.
    quantum_param_patterns is used to auto-discover parameters when the model
    does not implement SupportsQDropSpec.
    """
    algorithm: str = "baseline"  # baseline | pruning | dropout | both
    accumulate_window: int = 10
    prune_window: int = 8
    prune_ratio: float = 0.8
    schedule: bool = True
    drop_prob: float = 0.5        # maps to QDropConfig.dropout_prob
    n_drop_wires: int = 1
    # Substrings that identify quantum parameters in model.named_parameters().
    # QGCN uses ".qc.weights"; QGAT uses ("vqc.q_weights", "quantum_attention.weights").
    quantum_param_patterns: tuple = (".qc.weights",)

    def to_qdrop_config(self) -> QDropConfig:
        return QDropConfig(
            algorithm=self.algorithm,
            accumulate_window=self.accumulate_window,
            prune_window=self.prune_window,
            prune_ratio=self.prune_ratio,
            schedule=self.schedule,
            dropout_prob=self.drop_prob,
            n_drop_wires=self.n_drop_wires,
        )


class QuantumParameterMetadata(QDropTensorSpec):
    @property
    def parameter_name(self) -> str:
        return self.tensor_id


@dataclass
class DiscoveredQuantumLayer:
    module_name: str
    module: object
    parameter_specs: List[QuantumParameterMetadata]


def _num_wires_from_shape(shape: Tuple[int, ...]) -> int:
    """Infer qubit count from a quantum parameter shape."""
    if len(shape) >= 2:
        return shape[1]   # (n_layers, n_qubits) or (n_layers, n_qubits, 2)
    return max(shape[0], 1)  # flat 1D weight vector


def _make_mask_builder(param: torch.nn.Parameter):
    """Return a mask_builder callable for a single quantum parameter."""
    shape = tuple(param.shape)

    def mask_builder(dropped_wires: Tuple[int, ...]):
        mask = torch.zeros(shape, dtype=torch.bool, device=param.device)
        n_wires = _num_wires_from_shape(shape)
        for wire in dropped_wires:
            if wire >= n_wires:
                continue
            if len(shape) == 1:
                mask[wire] = True
            elif len(shape) == 2:
                mask[:, wire] = True        # (n_layers, n_qubits)
            else:
                mask[:, wire, :] = True     # (n_layers, n_qubits, 2)
        return mask

    return mask_builder


def _layer_specs_from_model(
    model: torch.nn.Module,
    quantum_param_patterns: tuple,
) -> List[QDropLayerSpec]:
    """Build QDropLayerSpec objects by scanning model.named_parameters()."""
    quantum_params = [
        (name, p)
        for name, p in model.named_parameters()
        if any(pat in name for pat in quantum_param_patterns)
    ]
    if not quantum_params:
        return []

    tensor_specs = [
        QDropTensorSpec(
            tensor_id=name,
            parameter=param,
            num_wires=_num_wires_from_shape(tuple(param.shape)),
            mask_builder=_make_mask_builder(param),
        )
        for name, param in quantum_params
    ]
    return [QDropLayerSpec(layer_id="model", tensor_specs=tensor_specs)]


def discover_quantum_layers(model) -> List[DiscoveredQuantumLayer]:
    if hasattr(model, "qdrop_layers"):
        return model.qdrop_layers()
    return []


class TorchQDropManager(TorchQDropRuntime):
    """
    Manage Q-Drop masking over quantum gradients in a PyTorch model.

    Quantum parameters are identified by name substrings in quantum_param_patterns.
    """

    def __init__(self, model=None, config: TorchQDropConfig | None = None, quantum_layers=None):
        if config is None:
            config = TorchQDropConfig()

        # Translate old TorchQDropConfig → QDropConfig
        qdrop_config = config.to_qdrop_config() if isinstance(config, TorchQDropConfig) else config

        # Build layer specs from the model when not explicitly provided
        if quantum_layers is None:
            if model is None:
                layer_specs = []
            elif hasattr(model, "qdrop_layers"):
                layer_specs = QDropSpecFactory.resolve(model.qdrop_layers())
            else:
                layer_specs = _layer_specs_from_model(model, config.quantum_param_patterns)
        else:
            layer_specs = QDropSpecFactory.resolve(quantum_layers)

        super().__init__(layer_specs, qdrop_config)


__all__ = [
    "DiscoveredQuantumLayer",
    "QuantumDropCompatible",
    "QuantumDropoutState",
    "QuantumParameterMetadata",
    "TorchQDropConfig",
    "TorchQDropManager",
    "discover_quantum_layers",
]
