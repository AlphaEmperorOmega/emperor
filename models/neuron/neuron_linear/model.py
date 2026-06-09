from dataclasses import fields

import torch

from torch import Tensor
from emperor.base.layer import Layer
from emperor.base.utils import ConfigBase, Module
from emperor.experiments.classifier import ClassifierExperiment
from models.neuron.neuron_linear.experiment_config import ExperimentConfig, HiddenBlockConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class HiddenBlockAdapter(Module):
    def __init__(
        self,
        cfg: HiddenBlockConfig,
        overrides: HiddenBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: HiddenBlockConfig = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.model_config: ConfigBase = self.cfg.model_config
        self.model = self.model_config.build(
            overrides=self._dimension_overrides(self.model_config)
        )

    def _dimension_overrides(self, model_config: ConfigBase):
        declared_fields = {field.name for field in fields(model_config)}
        overrides = {}
        if "input_dim" in declared_fields:
            overrides["input_dim"] = self.input_dim
        if "output_dim" in declared_fields:
            overrides["output_dim"] = self.output_dim
        if not overrides:
            return None
        return type(model_config)(**overrides)

    def forward(self, input: Tensor) -> Tensor:
        state = Layer.run_model_returning_state(self.model, input)
        return state.hidden


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ) -> None:
        super().__init__(cfg)
        if not isinstance(cfg.experiment_config, ExperimentConfig):
            raise TypeError(
                "cfg.experiment_config must be a neuron_linear ExperimentConfig."
            )
        self.model_cfg = cfg
        self.exp_cfg: ExperimentConfig = cfg.experiment_config
        self.input_model = self._build_input_model()
        self.neuron_cluster = self.exp_cfg.neuron_cluster_config.build()
        self.output_model = self._build_output_model()

    def _build_input_model(self):
        return self.exp_cfg.input_model_config.build(
            overrides=type(self.exp_cfg.input_model_config)(
                input_dim=self.model_cfg.input_dim,
                output_dim=self.model_cfg.hidden_dim,
            )
        )

    def _build_output_model(self):
        return self.exp_cfg.output_model_config.build(
            overrides=type(self.exp_cfg.output_model_config)(
                input_dim=self.model_cfg.hidden_dim,
                output_dim=self.model_cfg.output_dim,
            )
        )

    def forward(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor]:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        hidden = Layer.run_model_returning_hidden(self.input_model, X)
        hidden, auxiliary_loss = self.neuron_cluster(hidden)
        logits = Layer.run_model_returning_hidden(self.output_model, hidden)
        return logits, auxiliary_loss
