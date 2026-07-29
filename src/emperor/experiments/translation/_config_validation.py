import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.experiments._config_validation import _ExperimentConfigValidator

if TYPE_CHECKING:
    from emperor.config import ModelConfig


_DEFAULT_PAD_TOKEN_ID = 0
_DEFAULT_LABEL_SMOOTHING = 0.1
_DEFAULT_WARMUP_STEPS = 4_000
_DEFAULT_GENERATION_METRICS_FLAG = True


@dataclass(frozen=True, slots=True)
class _ResolvedTranslationConfig:
    learning_rate: float
    vocab_size: int
    model_dim: int
    pad_token_id: int
    label_smoothing: float
    warmup_steps: int
    generation_metrics_flag: bool


class _TranslationConfigValidator:
    def __init__(self) -> None:
        self._experiment_validator = _ExperimentConfigValidator("Translation")

    def resolve(self, config: "ModelConfig") -> _ResolvedTranslationConfig:
        common_config = self._experiment_validator.resolve(config)
        experiment_config = config.experiment_config
        return _ResolvedTranslationConfig(
            learning_rate=float(common_config.learning_rate),
            vocab_size=common_config.output_dim,
            model_dim=self._resolve_positive_integer(
                config.hidden_dim,
                "hidden_dim",
            ),
            pad_token_id=self._resolve_integer(
                getattr(
                    experiment_config,
                    "pad_token_id",
                    _DEFAULT_PAD_TOKEN_ID,
                ),
                "pad_token_id",
            ),
            label_smoothing=self._resolve_label_smoothing(
                getattr(
                    experiment_config,
                    "label_smoothing",
                    _DEFAULT_LABEL_SMOOTHING,
                )
            ),
            warmup_steps=self._resolve_positive_integer(
                getattr(
                    experiment_config,
                    "warmup_steps",
                    _DEFAULT_WARMUP_STEPS,
                ),
                "warmup_steps",
            ),
            generation_metrics_flag=self._resolve_generation_metrics_flag(
                getattr(
                    experiment_config,
                    "generation_metrics_flag",
                    _DEFAULT_GENERATION_METRICS_FLAG,
                )
            ),
        )

    @staticmethod
    def _resolve_positive_integer(value: object, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Translation config.{field_name} must be a positive integer."
            )
        return value

    @staticmethod
    def _resolve_integer(value: object, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Translation config.{field_name} must be an integer.")
        return value

    @staticmethod
    def _resolve_label_smoothing(value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                "Translation config.label_smoothing must be a finite real number "
                "between 0 and 1 inclusive."
            )
        return float(value)

    @staticmethod
    def _resolve_generation_metrics_flag(value: object) -> bool:
        if not isinstance(value, bool):
            raise ValueError(
                "Translation config.generation_metrics_flag must be a boolean."
            )
        return value
