from typing import TYPE_CHECKING

import torch
from emperor.base.layer.layer import Layer
from emperor.base.config import ConfigBase
from emperor.base.module import Module
from emperor.experiments.classifier import ClassifierExperiment
from torch import Tensor

from models.experts.linear.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        super().__init__(config)
        self.experiment_config: ExperimentConfig = config.experiment_config
        self.input_model = self.__build_input_model()
        self.main_model = self.__build_model()
        self.output_model = self.__build_output_model()

    def __build_input_model(self) -> Layer:
        return self.__build_from_experiment_config(
            self.experiment_config.input_model_config,
            input_dim=self.cfg.input_dim,
            output_dim=self.cfg.hidden_dim,
        )

    def __build_model(self) -> Module:
        return self.__build_from_experiment_config(
            self.experiment_config.model_config,
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
        )

    def __build_output_model(self) -> Layer:
        return self.__build_from_experiment_config(
            self.experiment_config.output_model_config,
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
        )

    def __build_from_experiment_config(
        self,
        model_config: ConfigBase,
        *,
        input_dim: int,
        output_dim: int,
    ) -> Module:
        dimension_overrides = type(model_config)(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        return model_config.build(overrides=dimension_overrides)

    def forward(
        self,
        X: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        X = Layer.run_model_returning_hidden(self.input_model, X)
        state = Layer.run_model_returning_state(self.main_model, X)
        classification_logits = Layer.run_model_returning_hidden(
            self.output_model, state.hidden
        )
        if state.loss is not None:
            return classification_logits, state.loss
        return classification_logits
