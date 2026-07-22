from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from emperor.experiments import ExperimentTask


class SyntheticInputError(ValueError):
    """An Experiment Task cannot create a safe synthetic input."""


@dataclass(frozen=True, slots=True)
class DatasetArgument:
    name: str
    configuration_path: tuple[str, ...]

    def value_from(self, configuration: Any) -> Any:
        value = configuration
        for attribute in self.configuration_path:
            value = getattr(value, attribute)
        return value


@dataclass(frozen=True, slots=True)
class RankingMetric:
    tier: float
    keys: tuple[str, ...]
    direction: float = 1.0


@dataclass(frozen=True, slots=True)
class ExperimentTaskBehavior:
    task: ExperimentTask
    synthetic_input_builder: Callable[[type, Any], tuple[Any, ...]]
    dataset_arguments: tuple[DatasetArgument, ...]
    ranking_metrics: tuple[RankingMetric, ...]
    missing_ranking_score: tuple[float, float]

    def synthetic_inputs(self, dataset: type, configuration: Any) -> tuple[Any, ...]:
        return self.synthetic_input_builder(dataset, configuration)

    def dataset_constructor_kwargs(self, configuration: Any) -> dict[str, Any]:
        return {
            argument.name: argument.value_from(configuration)
            for argument in self.dataset_arguments
        }

    def ranking_score(self, result: Mapping[str, Any]) -> tuple[float, float]:
        metrics = result.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        for preference in self.ranking_metrics:
            value = next(
                (
                    metrics[key]
                    for key in preference.keys
                    if metrics.get(key) is not None
                ),
                None,
            )
            if value is not None:
                return preference.tier, preference.direction * float(value)
        return self.missing_ranking_score


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SyntheticInputError(
            f"Cannot build a shape-trace input because {label} is not a positive "
            f"integer: {value!r}."
        )
    return value


def _token_batch(length: int, vocabulary_size: int, token_id: int = 1) -> Any:
    import torch

    if not 0 <= token_id < vocabulary_size:
        token_id = 1 if vocabulary_size > 1 else 0
    return torch.full((1, length), token_id, dtype=torch.long)


def _image_inputs(dataset: type, _configuration: Any) -> tuple[Any, ...]:
    import torch

    channels = _positive_integer(
        getattr(dataset, "num_channels", None),
        f"{dataset.__name__}.num_channels",
    )
    width = _positive_integer(
        getattr(dataset, "default_width", None),
        f"{dataset.__name__}.default_width",
    )
    height = _positive_integer(
        getattr(dataset, "default_height", None),
        f"{dataset.__name__}.default_height",
    )
    return (torch.zeros((1, channels, height, width), dtype=torch.float32),)


def _token_inputs(_dataset: type, configuration: Any) -> tuple[Any, ...]:
    sequence_length = _positive_integer(
        getattr(configuration, "sequence_length", None),
        "configuration.sequence_length",
    )
    vocabulary_size = _positive_integer(
        getattr(configuration, "input_dim", None),
        "configuration.input_dim",
    )
    return (_token_batch(sequence_length, vocabulary_size),)


def _translation_inputs(_dataset: type, configuration: Any) -> tuple[Any, ...]:
    experiment_config = getattr(configuration, "experiment_config", None)
    source_length = _positive_integer(
        getattr(experiment_config, "source_sequence_length", None),
        "configuration.experiment_config.source_sequence_length",
    )
    target_length = _positive_integer(
        getattr(experiment_config, "target_sequence_length", None),
        "configuration.experiment_config.target_sequence_length",
    )
    vocabulary_size = _positive_integer(
        getattr(experiment_config, "vocab_size", None),
        "configuration.experiment_config.vocab_size",
    )
    source_token = int(getattr(experiment_config, "bos_token_id", 1))
    return (
        _token_batch(source_length, vocabulary_size, source_token),
        _token_batch(max(1, target_length - 1), vocabulary_size, source_token),
    )


_BATCH_SIZE = DatasetArgument("batch_size", ("batch_size",))
_SEQUENCE_LENGTH = DatasetArgument("sequence_length", ("sequence_length",))
_SOURCE_SEQUENCE_LENGTH = DatasetArgument(
    "source_sequence_length",
    ("experiment_config", "source_sequence_length"),
)
_TARGET_SEQUENCE_LENGTH = DatasetArgument(
    "target_sequence_length",
    ("experiment_config", "target_sequence_length"),
)
_VALIDATION_ACCURACY = RankingMetric(
    tier=1.0,
    keys=("validation_accuracy", "validation/accuracy"),
)
_VALIDATION_LOSS = RankingMetric(
    tier=1.0,
    keys=("validation/loss", "validation_loss"),
    direction=-1.0,
)
_VALIDATION_BLEU = RankingMetric(
    tier=2.0,
    keys=("validation/bleu", "validation_bleu"),
)

_DECLARED_BEHAVIORS = (
    ExperimentTaskBehavior(
        task=ExperimentTask.IMAGE_CLASSIFICATION,
        synthetic_input_builder=_image_inputs,
        dataset_arguments=(_BATCH_SIZE,),
        ranking_metrics=(_VALIDATION_ACCURACY,),
        missing_ranking_score=(1.0, 0.0),
    ),
    ExperimentTaskBehavior(
        task=ExperimentTask.BERT_PRETRAINING,
        synthetic_input_builder=_token_inputs,
        dataset_arguments=(_BATCH_SIZE,),
        ranking_metrics=(_VALIDATION_ACCURACY,),
        missing_ranking_score=(1.0, 0.0),
    ),
    ExperimentTaskBehavior(
        task=ExperimentTask.TEXT_TRANSLATION,
        synthetic_input_builder=_translation_inputs,
        dataset_arguments=(
            _BATCH_SIZE,
            _SOURCE_SEQUENCE_LENGTH,
            _TARGET_SEQUENCE_LENGTH,
        ),
        ranking_metrics=(_VALIDATION_BLEU, _VALIDATION_LOSS),
        missing_ranking_score=(0.0, float("-inf")),
    ),
    ExperimentTaskBehavior(
        task=ExperimentTask.CAUSAL_LANGUAGE_MODELING,
        synthetic_input_builder=_token_inputs,
        dataset_arguments=(_BATCH_SIZE, _SEQUENCE_LENGTH),
        ranking_metrics=(_VALIDATION_LOSS,),
        missing_ranking_score=(0.0, float("-inf")),
    ),
)


def _compile_registry() -> Mapping[ExperimentTask, ExperimentTaskBehavior]:
    task_counts = Counter(behavior.task for behavior in _DECLARED_BEHAVIORS)
    duplicates = sorted(task.name for task, count in task_counts.items() if count > 1)
    missing = sorted(task.name for task in ExperimentTask if task not in task_counts)
    if duplicates or missing:
        details = []
        if duplicates:
            details.append(f"duplicates: {', '.join(duplicates)}")
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        raise RuntimeError(
            "Invalid Experiment Task behavior registry (" + "; ".join(details) + ")"
        )
    return MappingProxyType(
        {behavior.task: behavior for behavior in _DECLARED_BEHAVIORS}
    )


EXPERIMENT_TASK_BEHAVIORS = _compile_registry()


def experiment_task_behavior(task: ExperimentTask) -> ExperimentTaskBehavior:
    try:
        return EXPERIMENT_TASK_BEHAVIORS[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported Experiment Task behavior: {task!r}.") from exc


__all__ = [
    "EXPERIMENT_TASK_BEHAVIORS",
    "ExperimentTaskBehavior",
    "SyntheticInputError",
    "experiment_task_behavior",
]
