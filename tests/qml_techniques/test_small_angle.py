import torch

from qml_techniques.small_angle import make_small_angle_init


def test_returns_callable():
    init = make_small_angle_init(std=0.1)
    assert callable(init)


def test_inits_to_normal_distribution_with_given_std():
    torch.manual_seed(0)
    init = make_small_angle_init(std=0.1)
    t = torch.empty(10000)
    init(t)
    assert abs(float(t.mean().item())) < 0.01
    assert abs(float(t.std().item()) - 0.1) < 0.02


def test_default_std_is_small():
    init = make_small_angle_init()
    t = torch.empty(1000)
    init(t)
    assert float(t.std().item()) < 0.5  # nothing like uniform[0, 2pi]


def test_init_is_deterministic_with_seed():
    init = make_small_angle_init(std=0.1)
    torch.manual_seed(42)
    a = torch.empty(100)
    init(a)
    torch.manual_seed(42)
    b = torch.empty(100)
    init(b)
    assert torch.allclose(a, b)
