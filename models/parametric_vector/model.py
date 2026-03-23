import torch

from torch import Tensor
from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStack, LayerStackConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.parametric.utils.stack import ParametricLayerStack
from models.parametric_vector.config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.cfg: ExperimentConfig = self._resolve_main_config(self.cfg, cfg)
        self.input_config: LayerStackConfig = self.cfg.input_model_config
        self.model_config: LayerStackConfig = self.cfg.model_config
        self.output_config: LayerStackConfig = self.cfg.output_model_config

        self.input_model = LayerStack(self.input_config).build_model()
        self.model = ParametricLayerStack(self.model_config).build_model()
        self.output_model = LayerStack(self.output_config).build_model()

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> None:
        if sub_config.override_config is not None:
            return sub_config.override_config
        return main_cfg

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = torch.flatten(X.to(self.device), start_dim=1)
        X = self.input_model(X)
        X, _, _ = self.model(X)
        X = self.output_model(X)
        return X
