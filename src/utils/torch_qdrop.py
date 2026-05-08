"""Backward-compatible exports for Torch-side notebook helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from qdrop import QDropConfig, QDropDropoutState, QDropSpecFactory, QDropTensorSpec, SupportsQDropSpec
from qdrop.backends.torch_runtime import TorchQDropRuntime

TorchQDropConfig = QDropConfig
QuantumDropCompatible = SupportsQDropSpec
QuantumDropoutState = QDropDropoutState


@dataclass
class QuantumParameterMetadata(QDropTensorSpec):
    @property
    def parameter_name(self) -> str:
        return self.tensor_id


@dataclass
class DiscoveredQuantumLayer:
    module_name: str
    module: object
    parameter_specs: List[QuantumParameterMetadata]


def discover_quantum_layers(model) -> List[DiscoveredQuantumLayer]:
    if hasattr(model, "qdrop_layers"):
        quantum_layers = model.qdrop_layers()
    else:
        quantum_layers = model

    discovered_layers = []
    for layer_spec in QDropSpecFactory.resolve(quantum_layers):
        module = next((layer for layer in quantum_layers if layer.qdrop_layer_spec().layer_id == layer_spec.layer_id), None)
        parameter_specs = [
            QuantumParameterMetadata(
                tensor_id=tensor_spec.tensor_id,
                parameter=tensor_spec.parameter,
                num_wires=tensor_spec.num_wires,
                supports_gradient_mask=tensor_spec.supports_gradient_mask,
                supports_forward_mask=tensor_spec.supports_forward_mask,
                wire_masks=tensor_spec.wire_masks,
                mask_builder=tensor_spec.mask_builder,
            )
            for tensor_spec in layer_spec.tensor_specs
        ]
        discovered_layers.append(
            DiscoveredQuantumLayer(
                module_name=layer_spec.layer_id,
                module=module,
                parameter_specs=parameter_specs,
            )
        )
    return discovered_layers


class TorchQDropManager(TorchQDropRuntime):
    def __init__(self, model=None, config: TorchQDropConfig | None = None, quantum_layers=None):
        if config is None:
            config = TorchQDropConfig()

        if quantum_layers is None:
            if model is None:
                quantum_layers = []
            elif hasattr(model, "qdrop_layers"):
                quantum_layers = model.qdrop_layers()
            else:
                quantum_layers = model

        super().__init__(QDropSpecFactory.resolve(quantum_layers), config)


__all__ = [
    "DiscoveredQuantumLayer",
    "QuantumDropCompatible",
    "QuantumDropoutState",
    "QuantumParameterMetadata",
    "TorchQDropConfig",
    "TorchQDropManager",
    "discover_quantum_layers",
]
