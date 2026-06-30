"""QFI-Drop wiring: real adapter -> factory runtime -> probe -> grad surgery."""
import torch

from qdrop.types import QDropConfig
from qdrop.factories import QDropRuntimeFactory
from models.quantum_circuit_adapter import QuantumCircuitAdapter


def test_qfi_runtime_probes_and_transforms_grad():
    adapter = QuantumCircuitAdapter(n_qubits=8, n_layers=1)
    rt = QDropRuntimeFactory.create_torch(
        quantum_layers=[adapter], config=QDropConfig(algorithm="qfi")
    )
    rt.register_adapters([(adapter.qdrop_name, "weights", adapter)])
    rt.set_probe_batch((torch.randn(4, 8)).tanh() * 3.14159)

    rt.start_epoch(1)                                   # probes the QFIM -> cache filled
    assert len(rt.session._qfi_cache) == 1

    w = adapter.quantum_layer.weights
    w.grad = torch.randn_like(w)
    before = w.grad.clone()
    rt.after_backward()                                 # the qfi grad surgery runs here
    assert w.grad.shape == before.shape
    assert torch.isfinite(w.grad).all()
    assert not torch.allclose(w.grad, before)           # gradient was transformed


def test_non_qfi_path_untouched():
    """Guard: a baseline runtime ignores all the QFI machinery."""
    adapter = QuantumCircuitAdapter(n_qubits=8, n_layers=1)
    rt = QDropRuntimeFactory.create_torch(
        quantum_layers=[adapter], config=QDropConfig(algorithm="baseline")
    )
    rt.start_epoch(1)
    w = adapter.quantum_layer.weights
    w.grad = torch.ones_like(w)
    rt.after_backward()
    assert torch.allclose(w.grad, torch.ones_like(w))   # baseline leaves grads alone
