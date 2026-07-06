from dataclasses import dataclass

from models.bert._base_experiment_config import (
    ExperimentConfig as BaseExperimentConfig,
)


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    pass

__all__ = ["ExperimentConfig"]
