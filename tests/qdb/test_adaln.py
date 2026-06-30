"""Tests for qdb.adaln.AdaLNModulation (TDD - written before implementation)."""

from __future__ import annotations

import torch

from qdb.adaln import AdaLNModulation


def test_module_can_be_constructed():
    module = AdaLNModulation(dim=8)
    assert isinstance(module, torch.nn.Module)


def test_module_default_hidden_is_32():
    module = AdaLNModulation(dim=8)
    # Expose hidden as attribute, AND validate parameter count.
    assert module.hidden == 32
    n_params = sum(p.numel() for p in module.parameters())
    # Linear(1,32): 32 + 32 = 64; Linear(32, 16): 32*16 + 16 = 528; total = 592
    assert n_params == 592


def test_zero_init_produces_identity_at_construction():
    torch.manual_seed(0)
    module = AdaLNModulation(dim=8)
    x = torch.randn(4, 8)
    log_sigma = torch.tensor(0.0)
    y = module(x, log_sigma)
    assert torch.allclose(y, x, atol=1e-6)


def test_output_shape_matches_input_shape():
    module = AdaLNModulation(dim=8)
    x = torch.randn(10, 8)
    log_sigma = torch.tensor(0.5)
    y = module(x, log_sigma)
    assert y.shape == (10, 8)


def test_batch_index_routes_per_sample_modulation():
    torch.manual_seed(42)
    module = AdaLNModulation(dim=4)
    # Perturb weights away from zero so gamma/beta become non-trivial.
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    n_per = 3
    x = torch.randn(2 * n_per, 4)
    log_sigma = torch.tensor([0.5, -0.5])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    # Sanity: nodes in the same sample receive the same modulation.
    # Construct gamma/beta directly from the MLP, then verify routing.
    raw = module.mlp(log_sigma.unsqueeze(-1))  # [2, 2*dim]
    gamma, beta = raw.chunk(2, dim=-1)         # [2, dim] each

    y = module(x, log_sigma, batch=batch)
    expected = torch.empty_like(x)
    for i in range(x.shape[0]):
        expected[i] = (1 + gamma[batch[i]]) * x[i] + beta[batch[i]]
    assert torch.allclose(y, expected, atol=1e-6)

    # Verify the per-sample modulation actually differs between groups.
    assert not torch.allclose(gamma[0], gamma[1])
    assert not torch.allclose(beta[0], beta[1])


def test_differentiable_wrt_log_sigma():
    torch.manual_seed(0)
    module = AdaLNModulation(dim=8)
    # Break zero-init so gradient is non-trivial.
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(4, 8)
    log_sigma = torch.tensor(0.3, requires_grad=True)
    y = module(x, log_sigma)
    y.sum().backward()
    assert log_sigma.grad is not None
    assert torch.isfinite(log_sigma.grad).all()


def test_silu_activation_used_not_relu():
    """SiLU lets negative inputs leak through; ReLU would zero them.

    With a strongly negative log_sigma fed into Linear(1, hidden) whose bias is
    zero at init and weights are small, the post-Linear-1 pre-activation is
    nearly entirely negative. SiLU produces non-zero (small negative) outputs;
    ReLU would produce all zeros, making the final Linear output (and thus
    gamma/beta) identically zero even if its weights are non-zero.
    """
    torch.manual_seed(0)
    module = AdaLNModulation(dim=4)
    # Force first Linear weights positive so input -10 makes pre-activation negative.
    with torch.no_grad():
        module.mlp[0].weight.fill_(0.5)
        module.mlp[0].bias.zero_()
        # Make final Linear non-zero so any non-zero hidden propagates.
        module.mlp[2].weight.fill_(0.1)
        module.mlp[2].bias.zero_()
    log_sigma = torch.tensor(-10.0)
    raw = module.mlp(log_sigma.unsqueeze(-1))
    # Under ReLU this would be all zeros; under SiLU it is non-zero.
    assert not torch.allclose(raw, torch.zeros_like(raw), atol=1e-8)
