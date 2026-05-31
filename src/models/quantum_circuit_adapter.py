"""Q-Drop-compatible wrapper around a PennyLane TorchLayer."""

from __future__ import annotations

import os

import pennylane as qml
import torch
import torch.nn as nn

from qdrop.specs.pennylane_torch import PennyLaneTorchSpecFactory
from qdrop.types import QDropDropoutState


# Threshold at which default.qubit + backprop stops being competitive. At
# n_qubits >= 12 the state vector (2^n complex64) broadcast across PyG's
# batched ~600-node tensors crosses ~1 GB per intermediate, and backprop
# stores every intermediate -> tens of GB per training step. Switching to
# lightning.qubit + adjoint differentiation drops backward memory to O(1)
# in circuit depth at the cost of CPU-only simulation. Empirically much
# faster than burning hours in default.qubit + GPU OOM thrash.
_BACKEND_SWITCH_NQUBITS = 12


def _select_device_and_diff(n_qubits: int):
    """Return (device, diff_method) auto-selected by qubit count.

    - n_qubits < 12: default.qubit + backprop. State vector fits, GPU autograd
      is fastest.
    - n_qubits >= 12: lightning.qubit + adjoint differentiation. C++ CPU
      backend, O(1) memory in circuit depth, no intermediate storage.
    - Override via env var QDB_QUANTUM_BACKEND=lightning|default to force.
    """
    override = os.environ.get("QDB_QUANTUM_BACKEND", "").lower().strip()
    use_lightning = (
        override == "lightning"
        or (override != "default" and n_qubits >= _BACKEND_SWITCH_NQUBITS)
    )

    if use_lightning:
        try:
            return qml.device("lightning.qubit", wires=n_qubits), "adjoint"
        except qml.DeviceError:
            # pennylane-lightning not installed; fall through to default.qubit.
            pass

    return qml.device("default.qubit", wires=n_qubits), "backprop"


class QuantumCircuitAdapter(nn.Module):
    """Own a PennyLane TorchLayer and expose lazy Q-Drop specs for it."""

    def __init__(self, n_qubits: int, n_layers: int, use_strongly_entangling: bool = False):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_strongly_entangling = use_strongly_entangling
        self.qdrop_name = self.__class__.__name__

        device, diff_method = _select_device_and_diff(n_qubits)

        if use_strongly_entangling:
            @qml.qnode(device, interface="torch", diff_method=diff_method)
            def qnode(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(wire_index)) for wire_index in range(n_qubits)]

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        else:
            @qml.qnode(device, interface="torch", diff_method=diff_method)
            def qnode(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(wire_index)) for wire_index in range(n_qubits)]

            weight_shapes = {"weights": (n_layers, n_qubits)}

        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.register_buffer("forward_output_mask", torch.ones(n_qubits, dtype=torch.float32))

    @property
    def weights(self) -> nn.Parameter:
        """Expose the trainable quantum weights for compatibility and testing."""
        return self.quantum_layer.weights

    def mask_builder(self, wire_ids):
        if self.use_strongly_entangling:
            mask = torch.zeros(self.n_layers, self.n_qubits, 3, dtype=torch.bool, device=self.weights.device)
            for wire_index in wire_ids:
                if 0 <= wire_index < self.n_qubits:
                    mask[:, wire_index, :] = True
        else:
            mask = torch.zeros(self.n_layers, self.n_qubits, dtype=torch.bool, device=self.weights.device)
            for wire_index in wire_ids:
                if 0 <= wire_index < self.n_qubits:
                    mask[:, wire_index] = True
        return mask

    def set_forward_mask(self, dropout_state: QDropDropoutState | None) -> None:
        self.forward_output_mask.fill_(1.0)
        if dropout_state is None or not dropout_state.enabled:
            return

        for wire_index in dropout_state.dropped_wires:
            if 0 <= wire_index < self.n_qubits:
                self.forward_output_mask[wire_index] = 0.0

    def qdrop_layer_spec(self):
        return PennyLaneTorchSpecFactory.from_adapter(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantum_outputs = self.quantum_layer(inputs)
        quantum_outputs = torch.nan_to_num(quantum_outputs, nan=0.0, posinf=0.0, neginf=0.0)
        return quantum_outputs * self.forward_output_mask.to(quantum_outputs.device)
