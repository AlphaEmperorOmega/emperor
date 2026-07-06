from typing import TYPE_CHECKING

from models.neuron._model import Model as BaseModel
from models.neuron.expert_linear_adaptive.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(BaseModel):
    def __init__(self, config: "ModelConfig") -> None:
        if not isinstance(config.experiment_config, ExperimentConfig):
            raise TypeError(
                "config.experiment_config must be a Neuron Expert Linear "
                "Adaptive ExperimentConfig."
            )
        super().__init__(config)
        self.experiment_config: ExperimentConfig = config.experiment_config

__all__ = ["Model"]
