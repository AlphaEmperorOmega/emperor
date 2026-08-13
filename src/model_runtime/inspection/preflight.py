from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.runtime_defaults import (
    RuntimeDefaultsSpec,
    raise_runtime_defaults_inspection_error,
    runtime_defaults_spec,
)
from model_runtime.packages import ModelPackage, RuntimeDefaultsError
from model_runtime.packages.inspection_limits import (
    ESTIMATED_PARAMETER_RESIDENCY_BYTES,
    InspectionFieldProductLimit,
)

_DENSE_PARAMETER_ALLOWANCE = 6
_PARAMETER_MEMORY_SHARE_DIVISOR = 2


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
        numeric = _numeric(lock.value)
        if numeric is not None:
            config_key = spec.resolve_key(model_param) or model_param.upper()
            effective[model_param] = (config_key, numeric)
    return effective


def _positive_ints_ending_with(
    effective: Mapping[str, tuple[str, int | float]],
    suffixes: str | tuple[str, ...],
) -> tuple[int, ...]:
    return tuple(
        int(value)
        for model_param, (_, value) in effective.items()
        if model_param.endswith(suffixes)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _field_product(
    effective_by_key: Mapping[str, int | float],
    factors: tuple[tuple[str, ...], ...],
) -> int | None:
    product = 1
    for alternatives in factors:
        value = next(
            (effective_by_key[key] for key in alternatives if key in effective_by_key),
            None,
        )
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return None
        product *= value
    return product


class _InspectionPreflight:
    def __init__(
        self,
        package: ModelPackage,
        overrides: Mapping[str, Any],
        preset: Enum,
        memory_limit_bytes: int | None,
    ) -> None:
        self._spec = runtime_defaults_spec(package)
        self._limits = self._spec.inspection_limits
        self._effective = _effective_values(self._spec, overrides, preset)
        self._effective_by_key = {
            config_key.upper(): value for config_key, value in self._effective.values()
        }
        self._supported_keys = {key.upper() for key in self._spec.supported_keys}
        self._memory_limit_bytes = memory_limit_bytes

    def validate(self) -> int:
        self._validate_maximum_declarations()
        self._validate_field_values()
        multiplier = self._dense_parameter_multiplier()
        estimate = self._parameter_estimate(multiplier)
        maximum, maximum_source = self._maximum_parameter_estimate()
        if estimate > maximum:
            raise InspectionError(
                f"Inspection estimated parameter count {estimate} exceeds the "
                f"{maximum_source} of {maximum}."
            )
        return estimate

    def _validate_maximum_declarations(self) -> None:
        unknown_keys = sorted(set(self._limits.field_maximums) - self._supported_keys)
        if unknown_keys:
            raise InspectionError(
                "Inspection field maximums declare unknown Runtime Defaults "
                f"field(s): {', '.join(unknown_keys)}."
            )

    def _validate_field_values(self) -> None:
        for config_key, value in self._effective.values():
            maximum = self._limits.maximum_for(config_key)
            if maximum is not None and value > maximum:
                raise InspectionError(
                    f"Runtime Defaults field '{config_key}' value {value} exceeds "
                    f"the Inspection maximum of {maximum}."
                )

    def _validate_product_keys(
        self,
        product_limit: InspectionFieldProductLimit,
    ) -> None:
        for alternatives in product_limit.factors:
            for field_key in alternatives:
                if field_key not in self._supported_keys:
                    raise InspectionError(
                        f"Inspection product '{product_limit.label}' declares "
                        f"unknown Runtime Defaults field '{field_key}'."
                    )

    def _dense_parameter_multiplier(self) -> int:
        multiplier = 1
        for product_limit in self._limits.field_product_limits:
            self._validate_product_keys(product_limit)
            product = _field_product(
                self._effective_by_key,
                product_limit.factors,
            )
            if product is None:
                raise InspectionError(
                    f"Inspection product '{product_limit.label}' could not resolve "
                    "positive integer Runtime Defaults factors."
                )
            if product > product_limit.maximum:
                raise InspectionError(
                    f"Runtime Defaults {product_limit.label} {product} exceeds "
                    f"the Inspection maximum of {product_limit.maximum}."
                )
            if product_limit.repeats_dense_parameter_estimate:
                multiplier *= product
        return multiplier

    def _parameter_estimate(self, dense_multiplier: int) -> int:
        hidden_dimension = max(
            _positive_ints_ending_with(
                self._effective,
                ("hidden_dim", "model_dim", "embedding_dim"),
            ),
            default=0,
        )
        input_dimensions = _positive_ints_ending_with(self._effective, "input_dim")
        output_dimensions = _positive_ints_ending_with(
            self._effective,
            "output_dim",
        )
        vocabulary_sizes = _positive_ints_ending_with(
            self._effective,
            "vocab_size",
        )
        sequence_lengths = _positive_ints_ending_with(
            self._effective,
            "sequence_length",
        )
        layer_count = max(
            1,
            sum(_positive_ints_ending_with(self._effective, "num_layers")),
        )
        expert_count = max(
            (1, *_positive_ints_ending_with(self._effective, "num_experts"))
        )
        embedding_and_io_width = sum(
            (
                *input_dimensions,
                *output_dimensions,
                *vocabulary_sizes,
                *sequence_lengths,
            )
        )
        shared_estimate = hidden_dimension * embedding_and_io_width
        dense_estimate = (
            _DENSE_PARAMETER_ALLOWANCE
            * hidden_dimension
            * hidden_dimension
            * layer_count
            * expert_count
        )
        return shared_estimate + (dense_estimate * dense_multiplier)

    def _maximum_parameter_estimate(self) -> tuple[int, str]:
        maximum = self._limits.maximum_parameter_estimate
        maximum_source = "Model Package limit"
        if self._memory_limit_bytes is not None:
            memory_derived_maximum = self._memory_limit_bytes // (
                _PARAMETER_MEMORY_SHARE_DIVISOR * ESTIMATED_PARAMETER_RESIDENCY_BYTES
            )
            if memory_derived_maximum < maximum:
                maximum = memory_derived_maximum
                maximum_source = "memory-derived maximum"
        return maximum, maximum_source


def preflight_inspection_configuration(
    package: ModelPackage,
    overrides: Mapping[str, Any],
    preset: Enum,
    *,
    memory_limit_bytes: int | None = None,
) -> int:
    """Validate construction bounds and return a structural parameter estimate."""

    try:
        return _InspectionPreflight(
            package,
            overrides,
            preset,
            memory_limit_bytes,
        ).validate()
    except RuntimeDefaultsError as exc:
        raise_runtime_defaults_inspection_error(exc)


__all__ = ["preflight_inspection_configuration"]
