from torch import Tensor
from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStack
from emperor.experts.utils.model import MixtureOfExpertsModel
from emperor.experiments.classifier import ClassifierExperiment
from models.experts.config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.main_cfg: ExperimentConfig = self._resolve_main_config(self.cfg, cfg)

        self.experts_config = self.main_cfg.experts_config
        self.output_config = self.main_cfg.output_config

        self.experts = MixtureOfExpertsModel(self.experts_config)
        self.output = LayerStack(self.output_config).build_model()

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> "ExperimentConfig":
        if sub_config.override_config is not None:
            return sub_config.override_config
        return main_cfg

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = self.experts(X)
        return self.output(X)
