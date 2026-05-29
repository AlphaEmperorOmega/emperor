import torch

from torch import Tensor

from emperor.base.layer import LayerStackConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.parametric.core.state import ParametricLayerState
from models.parametric_matrix.config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        if not isinstance(cfg.experiment_config, ExperimentConfig):
            raise TypeError(
                "cfg.experiment_config must be a parametric_matrix ExperimentConfig."
            )

        self.main_cfg: ExperimentConfig = cfg.experiment_config
        self.model_config: LayerStackConfig = self.main_cfg.model_config
        self.model = self.model_config.build()

    def forward(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor]:
        X = torch.flatten(X.to(self.device), start_dim=1)
        state = ParametricLayerState(hidden=X)
        state = self.model(state)
        loss = state.loss if state.loss is not None else state.hidden.new_zeros(())
        return state.hidden, loss
