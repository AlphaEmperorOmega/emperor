from __future__ import annotations

import pytest

from model_runtime.inspection.preflight import preflight_inspection_configuration
from models.catalog import discover_model_packages

_MAXIMUM_DEFAULT_ESTIMATE_TO_ACTUAL_RATIO = 200


@pytest.mark.parametrize(
    "package",
    discover_model_packages(),
    ids=lambda package: package.catalog_key,
)
def test_default_parameter_estimate_is_safe_and_bounded(package) -> None:
    preset = package.default_preset
    estimate = preflight_inspection_configuration(package, {}, preset)
    configuration = package.build_configuration(preset)
    model = package.build_model(configuration)
    actual_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert actual_parameter_count <= estimate
    assert (
        estimate <= actual_parameter_count * _MAXIMUM_DEFAULT_ESTIMATE_TO_ACTUAL_RATIO
    )
    assert estimate <= package.inspection_construction_limits.maximum_parameter_estimate
