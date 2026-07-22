from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from emperor.experiments.classifier import ClassifierExperiment
from emperor.layers import Layer

from .experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.experiment_config: ExperimentConfig = config.experiment_config
        self.patch = self.experiment_config.patch_config.build()
        self.transformer = self.experiment_config.encoder_config.build()
        self.encoder_layer_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.output = self.experiment_config.output_config.build()

    @property
    def mixer(self):
        return self.transformer

    def forward(self, X: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        patch_tokens = self.patch(X.to(self.device))
        encoder_state = Layer.run_model_returning_state(
            self.transformer,
            patch_tokens,
        )
        normalized_tokens = self.encoder_layer_norm(encoder_state.hidden)
        pooled_hidden = normalized_tokens.mean(dim=1)
        logits = Layer.run_model_returning_hidden(self.output, pooled_hidden)
        if encoder_state.loss is not None and encoder_state.loss.item() != 0.0:
            return logits, encoder_state.loss
        return logits


__all__ = ["Model"]
