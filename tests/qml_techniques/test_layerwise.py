import pytest
import torch
import torch.nn as nn

from qml_techniques.layerwise import LayerwiseSchedule


class _StubQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(2, 4))


class _StubConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum_layer = _StubQuantumLayer()


class _StubModel(nn.Module):
    def __init__(self, n_convs: int):
        super().__init__()
        self.layers = nn.ModuleList([_StubConv() for _ in range(n_convs)])


def test_schedule_constructs_with_n_phases_equal_to_n_layers():
    model = _StubModel(n_convs=2)
    sched = LayerwiseSchedule(model, total_epochs=100, n_phases=2)
    assert sched.n_phases == 2


def test_phase_one_freezes_all_but_layer_zero():
    model = _StubModel(n_convs=2)
    sched = LayerwiseSchedule(model, total_epochs=100, n_phases=2)
    sched.apply_for_epoch(0)
    assert model.layers[0].quantum_layer.weights.requires_grad is True
    assert model.layers[1].quantum_layer.weights.requires_grad is False


def test_phase_two_unfreezes_layer_one():
    model = _StubModel(n_convs=2)
    sched = LayerwiseSchedule(model, total_epochs=100, n_phases=2)
    sched.apply_for_epoch(60)  # second half
    assert model.layers[0].quantum_layer.weights.requires_grad is True
    assert model.layers[1].quantum_layer.weights.requires_grad is True


def test_three_phase_schedule_unfreezes_progressively():
    model = _StubModel(n_convs=3)
    sched = LayerwiseSchedule(model, total_epochs=90, n_phases=3)
    sched.apply_for_epoch(0)
    assert [c.quantum_layer.weights.requires_grad for c in model.layers] == [True, False, False]
    sched.apply_for_epoch(35)
    assert [c.quantum_layer.weights.requires_grad for c in model.layers] == [True, True, False]
    sched.apply_for_epoch(75)
    assert [c.quantum_layer.weights.requires_grad for c in model.layers] == [True, True, True]


def test_invalid_n_phases_raises():
    model = _StubModel(n_convs=2)
    with pytest.raises(ValueError):
        LayerwiseSchedule(model, total_epochs=100, n_phases=0)
