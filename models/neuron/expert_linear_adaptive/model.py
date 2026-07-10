from typing import TYPE_CHECKING

import torch
from emperor.base.layer import Layer
from emperor.experiments.classifier import ClassifierExperiment
from torch import Tensor

from models.neuron.expert_linear_adaptive.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(self, config: "ModelConfig") -> None:
        if not isinstance(config.experiment_config, ExperimentConfig):
            raise TypeError(
                "config.experiment_config must be a Neuron Expert Linear "
                "Adaptive ExperimentConfig."
            )
        super().__init__(config)
        self.experiment_config: ExperimentConfig = config.experiment_config
        self.input_model = self._build_input_model()
        self.neuron_cluster = self.experiment_config.neuron_cluster_config.build()
        self.output_model = self._build_output_model()

    def _build_input_model(self):
        input_model_config = self.experiment_config.input_model_config
        input_model_config_type = type(input_model_config)
        return input_model_config.build(
            overrides=input_model_config_type(
                input_dim=self.cfg.input_dim,
                output_dim=self.cfg.hidden_dim,
            )
        )

    def _build_output_model(self):
        output_model_config = self.experiment_config.output_model_config
        output_model_config_type = type(output_model_config)
        return output_model_config.build(
            overrides=output_model_config_type(
                input_dim=self.cfg.hidden_dim,
                output_dim=self.cfg.output_dim,
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
