import torch

from torch import Tensor
from emperor.base.layer.config import LayerConfig
from emperor.base.layer.layer import Layer
from emperor.base.layer import LayerStack, LayerStackConfig
from emperor.experiments.classifier import ClassifierExperiment
from models.linear.experiment_config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.cfg: ExperimentConfig = self.cfg.experiment_config
        self.model_config: LayerStackConfig = self.cfg.model_config
        self.output_model_config: LayerConfig = self.cfg.output_model_config
        self.model = LayerStack(self.model_config).build()
        self.output_model = Layer(self.output_model_config)

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        X = Layer.forward_with_state(self.model, X)
        X = Layer.forward_with_state(self.output_model, X)
        return X
