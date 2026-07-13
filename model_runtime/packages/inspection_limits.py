from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _frozen_limits(values: Mapping[str, int | float]) -> Mapping[str, int | float]:
    return MappingProxyType({str(key).upper(): value for key, value in values.items()})


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
    maximum_parameter_estimate: int = 1_000_000_000
    field_maximums: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive_limits = {
            "maximum_hidden_dimension": self.maximum_hidden_dimension,
            "maximum_io_dimension": self.maximum_io_dimension,
            "maximum_sequence_length": self.maximum_sequence_length,
            "maximum_layer_count": self.maximum_layer_count,
            "maximum_expert_count": self.maximum_expert_count,
            "maximum_attention_head_count": self.maximum_attention_head_count,
            "maximum_recurrent_steps": self.maximum_recurrent_steps,
            "maximum_parameter_estimate": self.maximum_parameter_estimate,
        }
        invalid = [name for name, value in positive_limits.items() if value < 1]
        if invalid:
            raise ValueError(
                "Inspection construction limits must be positive: "
                + ", ".join(invalid)
            )
        object.__setattr__(self, "field_maximums", _frozen_limits(self.field_maximums))

    @property
    def maximum_dimension(self) -> int:
        """Compatibility name for the primary hidden/model dimension bound."""

        return self.maximum_hidden_dimension

    def maximum_for(self, config_key: str) -> int | float | None:
        """Return the explicit construction bound for one Runtime Defaults field."""

        key = config_key.upper()
        explicit = self.field_maximums.get(key)
        if explicit is not None:
            return explicit
        if key.endswith(("HIDDEN_DIM", "MODEL_DIM", "EMBEDDING_DIM")):
            return self.maximum_hidden_dimension
        if key.endswith(("INPUT_DIM", "OUTPUT_DIM", "VOCAB_SIZE")):
            return self.maximum_io_dimension
        if key.endswith("SEQUENCE_LENGTH"):
            return self.maximum_sequence_length
        if key.endswith("NUM_LAYERS"):
            return self.maximum_layer_count
        if key.endswith("NUM_EXPERTS"):
            return self.maximum_expert_count
        if key.endswith("NUM_HEADS"):
            return self.maximum_attention_head_count
        if key.endswith("MAX_STEPS") and "TRAINER_" not in key:
            return self.maximum_recurrent_steps
        if key.endswith("NUM_INNER_STEPS"):
            return self.maximum_recurrent_steps
        return None


DEFAULT_INSPECTION_CONSTRUCTION_LIMITS = InspectionConstructionLimits()


__all__ = [
    "DEFAULT_INSPECTION_CONSTRUCTION_LIMITS",
    "InspectionConstructionLimits",
]
