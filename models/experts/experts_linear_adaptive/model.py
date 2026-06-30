from typing import TYPE_CHECKING

import torch
from emperor.base.layer.layer import Layer
from emperor.experiments.classifier import ClassifierExperiment
from torch import Tensor

from models.classifier_pipeline import build_from_experiment_config
from models.experts.experts_linear_adaptive.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        super().__init__(config)
        self.experiment_config: ExperimentConfig = config.experiment_config
        self.input_model = self._build_input_model()
        self.main_model = self._build_main_model()
        self.output_model = self._build_output_model()

    def _build_input_model(self) -> Layer:
        return build_from_experiment_config(
            self.experiment_config.input_model_config,
            input_dim=self.cfg.input_dim,
            output_dim=self.cfg.hidden_dim,
        )

    def _build_main_model(self) -> "torch.nn.Module":
        return build_from_experiment_config(
            self.experiment_config.model_config,
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
        )

    def _build_output_model(self) -> Layer:
        return build_from_experiment_config(
            self.experiment_config.output_model_config,
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
        )

    def forward(
        self,
        X: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        X = Layer.run_model_returning_hidden(self.input_model, X)
        state = Layer.run_model_returning_state(self.main_model, X)
        logits = Layer.run_model_returning_hidden(self.output_model, state.hidden)
        if state.loss is not None:
            return logits, state.loss
        return logits
