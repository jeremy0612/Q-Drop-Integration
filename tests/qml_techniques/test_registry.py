import pytest

from qml_techniques.registry import get_technique, TECHNIQUE_NAMES


def test_known_techniques_listed():
    assert set(TECHNIQUE_NAMES) == {"baseline", "small_angle", "qng", "layerwise"}


def test_baseline_provides_no_hooks():
    spec = get_technique("baseline")
    assert spec.weight_init is None
    assert spec.optimizer_factory is None
    assert spec.schedule_factory is None


def test_small_angle_provides_weight_init():
    spec = get_technique("small_angle")
    assert callable(spec.weight_init)


def test_unknown_technique_raises():
    with pytest.raises(ValueError):
        get_technique("nonsense")
