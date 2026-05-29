import torch

from torch import Tensor

from emperor.base.layer import Layer, LayerStackConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.parametric.core.state import ParametricLayerState
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
        if not isinstance(cfg.experiment_config, ExperimentConfig):
            raise TypeError(
                "cfg.experiment_config must be a parametric_vector ExperimentConfig."
            )

        self.main_cfg: ExperimentConfig = cfg.experiment_config
        self.input_config: LayerStackConfig = self.main_cfg.input_model_config
        self.model_config: LayerStackConfig = self.main_cfg.model_config
        self.output_config: LayerStackConfig = self.main_cfg.output_model_config

        self.input_model = self.input_config.build()
        self.model = self.model_config.build()
        self.output_model = self.output_config.build()

    def forward(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor]:
        X = torch.flatten(X.to(self.device), start_dim=1)
        X = Layer.forward_with_state(self.input_model, X)

        state = ParametricLayerState(hidden=X)
        state = self.model(state)

        logits = Layer.forward_with_state(self.output_model, state.hidden)
        loss = state.loss if state.loss is not None else logits.new_zeros(())
        return logits, loss
