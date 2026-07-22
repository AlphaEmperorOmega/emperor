from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.runtime_defaults import (
    RuntimeDefaultsSpec,
    runtime_defaults_spec,
)
from model_runtime.packages import ModelPackage


def _numeric(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _effective_values(
    spec: RuntimeDefaultsSpec,
    overrides: Mapping[str, Any],
    preset: Enum,
) -> dict[str, tuple[str, int | float]]:
    effective: dict[str, tuple[str, int | float]] = {}
    for config_key in spec.supported_keys:
        model_param = spec.model_parameter(config_key)
        value = overrides.get(model_param, spec.current_value(config_key))
        numeric = _numeric(value)
        if numeric is not None:
            effective[model_param] = (config_key, numeric)

    for model_param, lock in spec.locks_for_preset(preset).items():
        numeric = _numeric(getattr(lock, "value", None))
        if numeric is not None:
            config_key = spec.resolve_key(model_param) or model_param.upper()
            effective[model_param] = (config_key, numeric)
    return effective


def _positive_int(
    effective: Mapping[str, tuple[str, int | float]],
    key: str,
) -> int:
    value = effective.get(key, (key.upper(), 0))[1]
    return int(value) if isinstance(value, int) and value > 0 else 0


def preflight_inspection_configuration(
    package: ModelPackage,
    overrides: Mapping[str, Any],
    preset: Enum,
) -> int:
    """Validate safe bounds and return a conservative parameter estimate."""

    spec = runtime_defaults_spec(package)
    limits = spec.inspection_limits
    effective = _effective_values(spec, overrides, preset)
    for config_key, value in effective.values():
        maximum = limits.maximum_for(config_key)
        if maximum is not None and value > maximum:
            raise InspectionError(
                f"Runtime Defaults field '{config_key}' value {value} exceeds "
                f"the Inspection maximum of {maximum}."
            )

    hidden_dimension = max(
        _positive_int(effective, "hidden_dim"),
        _positive_int(effective, "model_dim"),
    )
    input_dimension = _positive_int(effective, "input_dim")
    output_dimension = _positive_int(effective, "output_dim")
    layer_count = max(
        (
            int(value)
            for model_param, (_, value) in effective.items()
            if model_param.endswith("num_layers")
            and isinstance(value, int)
            and value > 0
        ),
        default=1,
    )
    expert_count = max(1, _positive_int(effective, "num_experts"))
    estimate = hidden_dimension * (input_dimension + output_dimension)
    estimate += hidden_dimension * hidden_dimension * layer_count * expert_count
    if estimate > limits.maximum_parameter_estimate:
        raise InspectionError(
            f"Inspection estimated parameter count {estimate} exceeds the "
            f"Model Package limit of {limits.maximum_parameter_estimate}."
        )
    return estimate


__all__ = ["preflight_inspection_configuration"]
