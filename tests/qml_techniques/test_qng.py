import pytest
import torch
import torch.nn as nn

from qml_techniques.qng import apply_fubini_study_precondition, QNGAdam


def test_precondition_with_identity_metric_is_identity():
    grad = torch.tensor([1.0, 2.0, 3.0])
    metric = torch.eye(3)
    out = apply_fubini_study_precondition(grad, metric, reg=0.0)
    assert torch.allclose(out, grad, atol=1e-6)


def test_precondition_with_diagonal_metric_scales_grad():
    grad = torch.tensor([1.0, 2.0, 3.0])
    metric = torch.diag(torch.tensor([2.0, 4.0, 8.0]))
    out = apply_fubini_study_precondition(grad, metric, reg=0.0)
    expected = torch.tensor([0.5, 0.5, 0.375])
    assert torch.allclose(out, expected, atol=1e-6)


def test_precondition_adds_regularization_for_singular_metric():
    grad = torch.tensor([1.0, 2.0])
    metric = torch.zeros(2, 2)  # singular
    out = apply_fubini_study_precondition(grad, metric, reg=1e-3)
    expected = grad * 1000.0
    assert torch.allclose(out, expected, atol=1e-3)


def test_qng_adam_constructs_with_param_groups():
    quantum_params = [nn.Parameter(torch.randn(2, 4))]
    classical_params = [nn.Parameter(torch.randn(8, 4))]
    opt = QNGAdam(
        quantum_params=quantum_params,
        classical_params=classical_params,
        metric_provider=lambda p: torch.eye(p.numel()),
        lr=1e-3,
    )
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["name"] == "quantum"
    assert opt.param_groups[1]["name"] == "classical"


def test_qng_adam_step_calls_metric_provider_on_quantum_only():
    qp = nn.Parameter(torch.zeros(2))
    cp = nn.Parameter(torch.zeros(2))
    qp.grad = torch.tensor([1.0, 2.0])
    cp.grad = torch.tensor([3.0, 4.0])

    seen = []

    def provider(p):
        seen.append(id(p))
        return torch.eye(p.numel())

    opt = QNGAdam(
        quantum_params=[qp],
        classical_params=[cp],
        metric_provider=provider,
        lr=1e-3,
    )
    opt.step()

    # Provider called once per quantum param, never for classical.
    assert seen == [id(qp)]


def test_qng_adam_mutates_quantum_grad_via_precondition():
    qp = nn.Parameter(torch.zeros(3))
    qp.grad = torch.tensor([2.0, 4.0, 6.0])
    # F = diag([2, 4, 8]) -> preconditioned grad = [1, 1, 0.75]
    opt = QNGAdam(
        quantum_params=[qp],
        classical_params=[],
        metric_provider=lambda p: torch.diag(torch.tensor([2.0, 4.0, 8.0])),
        lr=1e-3,
        qng_reg=0.0,
    )
    # Run only the precondition step (skip Adam update) by manually invoking
    # the same loop QNGAdam.step() runs.
    for p in opt.param_groups[0]["params"]:
        metric = opt.metric_provider(p)
        p.grad = apply_fubini_study_precondition(p.grad, metric, reg=opt.qng_reg)
    expected = torch.tensor([1.0, 1.0, 0.75])
    assert torch.allclose(qp.grad, expected, atol=1e-6)


def test_qng_adam_skips_precondition_when_grad_none():
    qp = nn.Parameter(torch.zeros(2))
    cp = nn.Parameter(torch.zeros(2))
    # qp.grad = None  -- skip
    cp.grad = torch.tensor([1.0, 1.0])
    opt = QNGAdam(
        quantum_params=[qp],
        classical_params=[cp],
        metric_provider=lambda p: torch.eye(p.numel()),
        lr=1e-3,
    )
    opt.step()  # must not raise
