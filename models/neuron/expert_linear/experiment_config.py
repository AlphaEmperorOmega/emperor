from dataclasses import dataclass

from models.neuron.experiment_config import (
    ExperimentConfig as BaseExperimentConfig,
    HiddenBlockConfig,
)


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    pass

__all__ = ["ExperimentConfig", "HiddenBlockConfig"]
