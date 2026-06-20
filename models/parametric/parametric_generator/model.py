import torch

from torch import Tensor

from emperor.base.layer import Layer, LayerConfig, LayerStackConfig
from emperor.base.utils import Module
from emperor.experiments.classifier import ClassifierExperiment
from emperor.parametric.core.state import ParametricLayerState
from models.parametric.parametric_generator.experiment_config import ExperimentConfig

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
                "cfg.experiment_config must be a parametric_generator "
                "ExperimentConfig."
            )

        self.model_cfg = cfg
        self.exp_cfg: ExperimentConfig = cfg.experiment_config
        self.input_config: LayerConfig = self.exp_cfg.input_model_config
        self.model_config: LayerStackConfig = self.exp_cfg.model_config
        self.output_config: LayerConfig = self.exp_cfg.output_model_config

        self.input_model = self._build_input_model()
        self.model = self._build_model()
        self.output_model = self._build_output_model()

    def _build_input_model(self) -> Layer:
        return self._build_from_experiment_config(
            self.input_config,
            input_dim=self.model_cfg.input_dim,
            output_dim=self.model_cfg.hidden_dim,
        )

    def _build_model(self) -> Module:
        return self._build_from_experiment_config(
            self.model_config,
            input_dim=self.model_cfg.hidden_dim,
            output_dim=self.model_cfg.hidden_dim,
        )

    def _build_output_model(self) -> Layer:
        return self._build_from_experiment_config(
            self.output_config,
            input_dim=self.model_cfg.hidden_dim,
            output_dim=self.model_cfg.output_dim,
        )

    def _build_from_experiment_config(
        self,
        model_config: LayerConfig | LayerStackConfig,
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
    ) -> tuple[Tensor, Tensor]:
        X = torch.flatten(X.to(self.device), start_dim=1)
        X = Layer.run_model_returning_hidden(self.input_model, X)

        state = ParametricLayerState(hidden=X)
        state = self.model(state)

        logits = Layer.run_model_returning_hidden(self.output_model, state.hidden)
        loss = state.loss if state.loss is not None else logits.new_zeros(())
        return logits, loss
