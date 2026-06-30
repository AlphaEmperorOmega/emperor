from typing import TYPE_CHECKING

import torch
from emperor.base.layer import Layer, LayerConfig, LayerStackConfig
from emperor.base.utils import Module
from emperor.experiments.classifier import ClassifierExperiment
from emperor.parametric.core.state import ParametricLayerState
from torch import Tensor

from models.classifier_pipeline import build_from_experiment_config
from models.parametric.parametric_matrix.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        super().__init__(config)
        if not isinstance(config.experiment_config, ExperimentConfig):
            raise TypeError(
                "config.experiment_config must be a parametric_matrix ExperimentConfig."
            )

        self.experiment_config: ExperimentConfig = config.experiment_config
        self.input_config: LayerConfig = self.experiment_config.input_model_config
        self.model_config: LayerStackConfig = self.experiment_config.model_config
        self.output_config: LayerConfig = self.experiment_config.output_model_config

        self.input_model = self._build_input_model()
        self.model = self._build_model()
        self.output_model = self._build_output_model()

    def _build_input_model(self) -> Layer:
        return build_from_experiment_config(
            self.input_config,
            input_dim=self.cfg.input_dim,
            output_dim=self.cfg.hidden_dim,
        )

    def _build_model(self) -> Module:
        return build_from_experiment_config(
            self.model_config,
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
        )

    def _build_output_model(self) -> Layer:
        return build_from_experiment_config(
            self.output_config,
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
        )

    def forward(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor]:
        X = torch.flatten(X.to(self.device), start_dim=1)
        X = Layer.run_model_returning_hidden(self.input_model, X)

        state = ParametricLayerState(hidden=X)
        state = self.model(state)

        logits = Layer.run_model_returning_hidden(self.output_model, state.hidden)
        loss = state.loss if state.loss is not None else logits.new_zeros(())
        return logits, loss
