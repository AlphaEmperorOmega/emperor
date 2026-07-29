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
    def __init__(
        self,
        experiment_name: str,
        *,
        minimum_output_dim: int = 1,
    ) -> None:
        self._experiment_name = experiment_name
        self._minimum_output_dim = minimum_output_dim

    def resolve(self, config: "ModelConfig") -> _ResolvedExperimentConfig:
        return _ResolvedExperimentConfig(
            learning_rate=self._resolve_learning_rate(config.learning_rate),
            output_dim=self._resolve_output_dim(config.output_dim),
        )

    def _resolve_learning_rate(self, value: object) -> int | float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"{self._experiment_name} config.learning_rate must be a finite "
                "real number greater than or equal to 0."
            )
        return value

    def _resolve_output_dim(self, value: object) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < self._minimum_output_dim
        ):
            requirement = "a positive integer"
            if self._minimum_output_dim > 1:
                requirement = (
                    f"an integer greater than or equal to {self._minimum_output_dim}"
                )
            raise ValueError(
                f"{self._experiment_name} config.output_dim must be {requirement}."
            )
        return value
