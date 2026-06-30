"""QFI-Drop core: spectral leverage, freeze+precondition, and a real-adapter probe."""
import torch

from qdrop.qfi import spectral_leverage, prune_and_precondition


def test_leverage_identity_is_all_ones():
    e = spectral_leverage(torch.eye(4), spectral_ratio=1e-3)
    assert torch.allclose(e, torch.ones(4), atol=1e-6)


def test_leverage_one_dominant_direction():
    F = torch.diag(torch.tensor([1.0, 1e-9, 1e-9, 1e-9]))
    e = spectral_leverage(F, spectral_ratio=1e-3)
    assert e[0] > 0.99
    assert torch.allclose(e[1:], torch.zeros(3), atol=1e-6)


def test_leverage_bounded_unit():
    torch.manual_seed(0)
    A = torch.randn(6, 6)
    e = spectral_leverage(A @ A.T, spectral_ratio=1e-3)
    assert (e >= -1e-6).all() and (e <= 1 + 1e-6).all()


def test_prune_freezes_low_leverage_and_preconditions_survivors():
    F = torch.eye(4)
    e = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = prune_and_precondition(torch.tensor([2.0, 2.0, 2.0, 2.0]), F, e, 0.5, 1e-4)
    assert torch.allclose(out[1:], torch.zeros(3))
    assert abs(out[0].item() - 2.0 / 1.0001) < 1e-4


def test_prune_all_frozen_is_safe():
    out = prune_and_precondition(torch.ones(2), torch.eye(2), torch.zeros(2), 0.5)
    assert torch.allclose(out, torch.zeros(2))


def test_prune_preserves_shape_and_dtype():
    g = torch.randn(2, 4)                     # (n_layers, n_qubits)
    out = prune_and_precondition(g, torch.eye(8), torch.ones(8))
    assert out.shape == g.shape and out.dtype == g.dtype


def test_end_to_end_on_real_adapter():
    """Probe a real QuantumCircuitAdapter -> leverage -> prune a gradient. The whole pipeline."""
    from models.quantum_circuit_adapter import QuantumCircuitAdapter
    from qdrop.qfi_metric import compute_qfim

    adapter = QuantumCircuitAdapter(n_qubits=8, n_layers=1)
    p = adapter.weights.numel()
    probe = (torch.randn(4, 8)).tanh() * 3.14159            # bounded like the model's input
    F = compute_qfim(adapter, probe)
    assert F.shape == (p, p)
    assert bool((torch.linalg.eigvalsh(0.5 * (F + F.T)) >= -1e-6).all())   # PSD

    e = spectral_leverage(F)
    assert e.shape == (p,) and (e >= -1e-6).all() and (e <= 1 + 1e-6).all()

    grad = torch.randn(adapter.weights.shape)
    out = prune_and_precondition(grad, F, e)
    assert out.shape == grad.shape
    assert torch.isfinite(out).all()
    frozen = e < 0.5
    assert torch.allclose(out.reshape(-1)[frozen], torch.zeros(int(frozen.sum())))
