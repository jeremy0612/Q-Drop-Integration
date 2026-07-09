"""Data-averaged Quantum Fisher Information for a QuantumCircuitAdapter.

The QFIM is the Fubini-Study metric of the node-embedding circuit, averaged over a
batch of bounded node-feature angle vectors. Probed on a ``default.qubit`` shadow so it
is independent of the training backend (lightning/adjoint), with the embedding angles
baked in as constants -- the block-diagonal metric_tensor is only well-defined when the
embedding angles are NOT tape parameters (verified empirically; full/approx=None needs an
auxiliary wire the device lacks for n_layers>=2).
"""
from __future__ import annotations

import pennylane as qml
import torch


def _shadow_metric(n_qubits, n_layers, use_se, x, weights, embedding_rotation="X"):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(w):
        qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation=embedding_rotation)
        template = (qml.templates.StronglyEntanglingLayers if use_se
                    else qml.templates.BasicEntanglerLayers)
        template(w, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    p = weights.numel()
    F = qml.metric_tensor(circuit, approx="block-diag")(weights)
    return torch.as_tensor(F, dtype=torch.float64).reshape(p, p)


def compute_qfim(adapter, probe_inputs: torch.Tensor, max_probe: int = 32) -> torch.Tensor:
    """probe_inputs: (B, n_qubits) bounded angle vectors. Returns (p, p) mean QFIM, p = weights.numel()."""
    n = adapter.n_qubits
    use_se = getattr(adapter, "use_strongly_entangling", False)
    w = adapter.weights.detach().double().requires_grad_(True)
    rows = probe_inputs.detach().double()[:max_probe]
    # ponytail: rebuilds a shadow qnode per probe row (block-diag needs the embedding angles
    # as constants, not tape params). Fine at once/epoch, B<=32, n<=16; revisit if it dominates.
    rot = getattr(adapter, "embedding_rotation", "X")
    mats = [_shadow_metric(n, adapter.n_layers, use_se, rows[i], w, rot) for i in range(rows.shape[0])]
    return torch.stack(mats).mean(dim=0)
