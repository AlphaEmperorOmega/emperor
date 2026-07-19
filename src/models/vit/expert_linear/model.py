from typing import TYPE_CHECKING

import torch.nn as nn
from emperor.base.layer import Layer
from emperor.experiments.classifier import ClassifierExperiment
from torch import Tensor

from models.vit.expert_linear.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        experiment_config = self.__validate_experiment_config(config)
        super().__init__(config)
        self.experiment_config: ExperimentConfig = experiment_config
        self.patch = self.__build_patch_model()
        self.positional_embedding = self.__build_positional_embedding()
        self.transformer = self.__build_encoder_model()
        self.encoder_layer_norm = self.__build_encoder_layer_norm()
        self.output = self.__build_output_model()

    @staticmethod
    def __validate_experiment_config(
        config: "ModelConfig",
    ) -> ExperimentConfig:
        if not isinstance(config.experiment_config, ExperimentConfig):
            raise TypeError(
                "config.experiment_config must be a ViT Expert Linear ExperimentConfig."
            )
        return config.experiment_config

    def __build_patch_model(self) -> nn.Module:
        return self.experiment_config.patch_config.build()

    def __build_positional_embedding(self) -> nn.Module:
        return self.experiment_config.positional_embedding_config.build()

    def __build_encoder_model(self) -> nn.Module:
        return self.experiment_config.encoder_config.build()

    def __build_encoder_layer_norm(self) -> nn.LayerNorm:
        return nn.LayerNorm(self.cfg.hidden_dim)

    def __build_output_model(self) -> nn.Module:
        return self.experiment_config.output_config.build()

    def forward(
        self,
        images: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        images = images.to(self.device)
        hidden = self.patch(images)
        hidden = self.positional_embedding(hidden)

        encoder_state = Layer.run_model_returning_state(self.transformer, hidden)
        hidden = self.encoder_layer_norm(encoder_state.hidden)

        cls_hidden = hidden[:, 0, :]
        logits = self.output(cls_hidden)

        if encoder_state.loss is not None and encoder_state.loss.item() != 0.0:
            return logits, encoder_state.loss
        return logits
