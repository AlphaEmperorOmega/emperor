import torch

from torch import Tensor
from emperor.base.layer.layer import Layer
from emperor.base.layer import LayerStack, LayerStackConfig
from emperor.experiments.classifier import ClassifierExperiment
from models.linear.config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.experiment_config: ExperimentConfig = self.cfg.experiment_config
        self.model_config: LayerStackConfig = self.experiment_config.model_config
        self.model = LayerStack(self.model_config).build()

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        X = Layer.forward_with_state(self.model, X)
        return X
