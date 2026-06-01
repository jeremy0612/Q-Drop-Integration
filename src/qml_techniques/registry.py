"""Registry that maps --technique CLI value to (weight_init, optimizer, schedule)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from qml_techniques.small_angle import make_small_angle_init


TECHNIQUE_NAMES = ("baseline", "small_angle", "qng", "layerwise")


@dataclass
class TechniqueSpec:
    name: str
    weight_init: Optional[Callable] = None
    optimizer_factory: Optional[Callable] = None  # (model, config) -> Optimizer
    schedule_factory: Optional[Callable] = None  # (model, config) -> LayerwiseSchedule


def _baseline_spec() -> TechniqueSpec:
    return TechniqueSpec(name="baseline")


def _small_angle_spec() -> TechniqueSpec:
    return TechniqueSpec(name="small_angle", weight_init=make_small_angle_init(std=0.1))


def _qng_spec() -> TechniqueSpec:
    # Optimizer factory wired in training/graph_training.py — needs access
    # to the model's bare qnode for metric-tensor evaluation.
    return TechniqueSpec(name="qng")


def _layerwise_spec() -> TechniqueSpec:
    return TechniqueSpec(name="layerwise")


_BUILDERS = {
    "baseline": _baseline_spec,
    "small_angle": _small_angle_spec,
    "qng": _qng_spec,
    "layerwise": _layerwise_spec,
}


def get_technique(name: str) -> TechniqueSpec:
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown technique {name!r}. Choose from {TECHNIQUE_NAMES}."
        )
    return _BUILDERS[name]()
