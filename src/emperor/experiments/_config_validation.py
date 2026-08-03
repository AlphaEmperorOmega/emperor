import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


@dataclass(frozen=True, slots=True)
class _ResolvedExperimentConfig:
    learning_rate: int | float
    output_dim: int


class _ExperimentConfigValidator:
    @classmethod
    def resolve(
        cls,
        config: "ModelConfig",
        experiment_name: str,
        *,
        minimum_output_dim: int = 1,
    ) -> _ResolvedExperimentConfig:
        return _ResolvedExperimentConfig(
            learning_rate=cls._resolve_learning_rate(
                config.learning_rate,
                experiment_name,
            ),
            output_dim=cls._resolve_output_dim(
                config.output_dim,
                experiment_name,
                minimum_output_dim,
            ),
        )

    @staticmethod
    def _resolve_learning_rate(
        value: object,
        experiment_name: str,
    ) -> int | float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"{experiment_name} config.learning_rate must be a finite "
                "real number greater than or equal to 0."
            )
        return value

    @staticmethod
    def _resolve_output_dim(
        value: object,
        experiment_name: str,
        minimum_output_dim: int,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum_output_dim
        ):
            requirement = "a positive integer"
            if minimum_output_dim > 1:
                requirement = (
                    f"an integer greater than or equal to {minimum_output_dim}"
                )
            raise ValueError(
                f"{experiment_name} config.output_dim must be {requirement}."
            )
        return value
