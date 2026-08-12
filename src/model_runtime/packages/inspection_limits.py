from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import cast

ESTIMATED_PARAMETER_RESIDENCY_BYTES = 8
DEFAULT_INSPECTION_PARAMETER_STORAGE_BUDGET_BYTES = 2 * 1024**3
DEFAULT_MAXIMUM_PARAMETER_ESTIMATE = (
    DEFAULT_INSPECTION_PARAMETER_STORAGE_BUDGET_BYTES
    // ESTIMATED_PARAMETER_RESIDENCY_BYTES
)

_FIELD_MAXIMUM_RULES: tuple[
    tuple[str | tuple[str, ...], str, str | None],
    ...,
] = (
    (
        ("HIDDEN_DIM", "MODEL_DIM", "EMBEDDING_DIM"),
        "maximum_hidden_dimension",
        None,
    ),
    (("INPUT_DIM", "OUTPUT_DIM", "VOCAB_SIZE"), "maximum_io_dimension", None),
    ("SEQUENCE_LENGTH", "maximum_sequence_length", None),
    ("NUM_LAYERS", "maximum_layer_count", None),
    ("NUM_EXPERTS", "maximum_expert_count", None),
    ("NUM_HEADS", "maximum_attention_head_count", None),
    ("MAX_STEPS", "maximum_recurrent_steps", "TRAINER_"),
    ("NUM_INNER_STEPS", "maximum_recurrent_steps", None),
)


def _validate_positive_integer_limits(values: Mapping[str, object]) -> None:
    invalid = [
        name for name, value in values.items() if type(value) is not int or value < 1
    ]
    if invalid:
        raise ValueError(
            "Inspection construction limits must be positive: " + ", ".join(invalid)
        )


def _frozen_limits(value: object) -> Mapping[str, int | float]:
    if not isinstance(value, Mapping):
        raise TypeError("Inspection field maximums must be a mapping.")
    limits: dict[str, int | float] = {}
    for key, maximum in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise TypeError("Inspection field maximum keys must be strings.")
        normalized_key = key.strip().upper()
        if not normalized_key:
            raise ValueError("Inspection field maximum keys must be non-empty.")
        valid_integer = type(maximum) is int and maximum > 0
        valid_float = type(maximum) is float and isfinite(maximum) and maximum > 0
        if not valid_integer and not valid_float:
            raise ValueError(
                "Inspection field maximums must be finite positive numbers."
            )
        limits[normalized_key] = cast(int | float, maximum)
    return MappingProxyType(limits)


def _validated_label(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Inspection field product labels must be non-empty.")
    return value.strip()


def _validated_positive_maximum(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Inspection field product maximums must be positive.")
    return value


def _normalized_factors(value: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("Inspection field products require non-empty factors.")
    normalized_factors: list[tuple[str, ...]] = []
    for alternatives_value in cast(tuple[object, ...], value):
        if not isinstance(alternatives_value, tuple) or not alternatives_value:
            raise ValueError("Inspection field products require non-empty factors.")
        alternatives = cast(tuple[object, ...], alternatives_value)
        if any(not isinstance(key, str) for key in alternatives):
            raise TypeError("Inspection field product keys must be strings.")
        normalized = tuple(cast(str, key).strip().upper() for key in alternatives)
        if any(not key for key in normalized):
            raise ValueError("Inspection field product keys must be non-empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Inspection field product alternatives must be unique.")
        normalized_factors.append(normalized)
    return tuple(normalized_factors)


def _validated_repetition_flag(value: object) -> bool:
    if type(value) is not bool:
        raise TypeError("Inspection dense-estimate repetition flags must be boolean.")
    return value


def _validated_product_limits(
    value: object,
) -> tuple[InspectionFieldProductLimit, ...]:
    if not isinstance(value, tuple):
        raise TypeError(
            "Inspection field product limits must be "
            "InspectionFieldProductLimit values."
        )
    validated: list[InspectionFieldProductLimit] = []
    for limit in cast(tuple[object, ...], value):
        if not isinstance(limit, InspectionFieldProductLimit):
            raise TypeError(
                "Inspection field product limits must be "
                "InspectionFieldProductLimit values."
            )
        validated.append(limit)
    return tuple(validated)


@dataclass(frozen=True, slots=True)
class InspectionFieldProductLimit:
    """Bound a package-owned product of effective Runtime Defaults fields.

    Each factor lists keys in preference order. This supports fields such as an
    initial cluster dimension which falls back to its capacity dimension when
    omitted.
    """

    label: str
    factors: tuple[tuple[str, ...], ...]
    maximum: int
    repeats_dense_parameter_estimate: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _validated_label(self.label))
        object.__setattr__(
            self,
            "maximum",
            _validated_positive_maximum(self.maximum),
        )
        object.__setattr__(self, "factors", _normalized_factors(self.factors))
        object.__setattr__(
            self,
            "repeats_dense_parameter_estimate",
            _validated_repetition_flag(self.repeats_dense_parameter_estimate),
        )


@dataclass(frozen=True)
class InspectionConstructionLimits:
    """Pre-construction limits exposed by a selected Model Package.

    The defaults cover the shared Runtime Defaults vocabulary. A package with
    different safe operating bounds can supply ``field_maximums`` without
    teaching an Adapter about that package's identity.
    """

    maximum_hidden_dimension: int = 16_384
    maximum_io_dimension: int = 1_000_000
    maximum_sequence_length: int = 1_000_000
    maximum_layer_count: int = 256
    maximum_expert_count: int = 1_024
    maximum_attention_head_count: int = 1_024
    maximum_recurrent_steps: int = 4_096
    maximum_parameter_estimate: int = DEFAULT_MAXIMUM_PARAMETER_ESTIMATE
    field_maximums: Mapping[str, int | float] = field(
        default_factory=dict[str, int | float]
    )
    field_product_limits: tuple[InspectionFieldProductLimit, ...] = ()

    def __post_init__(self) -> None:
        _validate_positive_integer_limits(
            {
                "maximum_hidden_dimension": self.maximum_hidden_dimension,
                "maximum_io_dimension": self.maximum_io_dimension,
                "maximum_sequence_length": self.maximum_sequence_length,
                "maximum_layer_count": self.maximum_layer_count,
                "maximum_expert_count": self.maximum_expert_count,
                "maximum_attention_head_count": self.maximum_attention_head_count,
                "maximum_recurrent_steps": self.maximum_recurrent_steps,
                "maximum_parameter_estimate": self.maximum_parameter_estimate,
            }
        )
        object.__setattr__(self, "field_maximums", _frozen_limits(self.field_maximums))
        object.__setattr__(
            self,
            "field_product_limits",
            _validated_product_limits(
                object.__getattribute__(self, "field_product_limits")
            ),
        )

    def maximum_for(self, config_key: str) -> int | float | None:
        """Return the explicit construction bound for one Runtime Defaults field."""

        key = config_key.upper()
        explicit = self.field_maximums.get(key)
        if explicit is not None:
            return explicit
        for suffixes, limit_attribute, excluded_fragment in _FIELD_MAXIMUM_RULES:
            if not key.endswith(suffixes):
                continue
            if excluded_fragment is not None and excluded_fragment in key:
                continue
            return cast(int, getattr(self, limit_attribute))
        return None


DEFAULT_INSPECTION_CONSTRUCTION_LIMITS = InspectionConstructionLimits()


__all__ = [
    "DEFAULT_INSPECTION_CONSTRUCTION_LIMITS",
    "DEFAULT_INSPECTION_PARAMETER_STORAGE_BUDGET_BYTES",
    "DEFAULT_MAXIMUM_PARAMETER_ESTIMATE",
    "ESTIMATED_PARAMETER_RESIDENCY_BYTES",
    "InspectionConstructionLimits",
    "InspectionFieldProductLimit",
]
