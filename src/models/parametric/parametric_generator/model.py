from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.config import ConfigBase
from emperor.experiments.classifier import ClassifierExperiment
from emperor.layers import (
    Layer,
    LayerConfig,
    LayerStackConfig,
)
from emperor.nn import Module
from emperor.parametric import ParametricLayerState
from models.parametric.parametric_generator.experiment_config import ExperimentConfig

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
                "config.experiment_config must be a parametric_generator "
                "ExperimentConfig."
            )

        self.experiment_config: ExperimentConfig = config.experiment_config
        self.input_config: LayerConfig = self.experiment_config.input_model_config
        self.model_config: LayerStackConfig = self.experiment_config.model_config
        self.output_config: LayerConfig = self.experiment_config.output_model_config

        self.input_model = self.__build_input_model()
        self.model = self.__build_model()
        self.output_model = self.__build_output_model()

    def __build_input_model(self) -> Layer:
        return self.__build_from_experiment_config(
            self.input_config,
            input_dim=self.cfg.input_dim,
            output_dim=self.cfg.hidden_dim,
        )

    def __build_model(self) -> Module:
        return self.__build_from_experiment_config(
            self.model_config,
            input_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.hidden_dim,
        )

    def __build_output_model(self) -> Layer:
        return self.__build_from_experiment_config(
            self.output_config,
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
    ) -> tuple[Tensor, Tensor]:
        X = torch.flatten(X.to(self.device), start_dim=1)
        input_state = Layer.run_model_from_hidden(self.input_model, X)

        parametric_state = ParametricLayerState(hidden=input_state.hidden)
        parametric_state = self.model(parametric_state)

        output_state = Layer.run_model_from_hidden(
            self.output_model,
            parametric_state.hidden,
        )
        logits = output_state.hidden
        loss = (
            parametric_state.loss
            if parametric_state.loss is not None
            else logits.new_zeros(())
        )
        return logits, loss
