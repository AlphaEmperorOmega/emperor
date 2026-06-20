from typing import TYPE_CHECKING

import torch
from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.layer import Layer
from emperor.base.utils import Module
from emperor.experiments.classifier import ClassifierExperiment
from torch import Tensor

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.model_cfg = cfg
        self.exp_cfg = cfg.experiment_config
        self.input_model = self._build_input_model()
        self.main_model = self._build_model()
        self.output_model = self._build_output_model()

    def _build_input_model(self) -> Layer:
        return self._build_from_experiment_config(
            self.exp_cfg.input_model_config,
            input_dim=self.model_cfg.input_dim,
            output_dim=self.model_cfg.hidden_dim,
        )

    def _build_model(self) -> Module:
        return self._build_from_experiment_config(
            self.exp_cfg.model_config,
            input_dim=self.model_cfg.hidden_dim,
            output_dim=self.model_cfg.hidden_dim,
        )

    def _build_output_model(self) -> Layer:
        return self._build_from_experiment_config(
            self.exp_cfg.output_model_config,
            input_dim=self.model_cfg.hidden_dim,
            output_dim=self.model_cfg.output_dim,
        )

    def _build_from_experiment_config(
        self,
        model_config: LayerConfig | LayerStackConfig | RecurrentLayerConfig,
        input_dim: int,
        output_dim: int,
    ) -> Module:
        override = type(model_config)(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        return model_config.build(overrides=override)

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
