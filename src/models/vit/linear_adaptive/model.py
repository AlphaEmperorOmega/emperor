from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from emperor.experiments.classifier import ClassifierExperiment
from emperor.layers import Layer, LayerState
from models.vit.linear_adaptive.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        super().__init__(config)
        self.experiment_config: ExperimentConfig = config.experiment_config
        self.patch = self.__build_patch_model()
        self.positional_embedding = self.__build_positional_embedding()
        self.transformer = self.__build_encoder_model()
        self.encoder_layer_norm = self.__build_encoder_layer_norm()
        self.output = self.__build_output_model()

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
        X: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        X = X.to(self.device)
        token_embeddings = self.__build_token_embeddings(X)
        encoder_state = self.__encode_token_embeddings(token_embeddings)
        classification_logits = self.__build_classification_logits(encoder_state.hidden)

        if encoder_state.loss is not None and encoder_state.loss.item() != 0.0:
            return classification_logits, encoder_state.loss
        return classification_logits

    def __build_token_embeddings(self, X: Tensor) -> Tensor:
        patch_embeddings = self.patch(X)
        return self.positional_embedding(patch_embeddings)

    def __encode_token_embeddings(self, token_embeddings: Tensor) -> LayerState:
        encoder_state = Layer.run_model_returning_state(
            self.transformer, token_embeddings
        )
        encoder_state.hidden = self.encoder_layer_norm(encoder_state.hidden)
        return encoder_state

    def __build_classification_logits(self, encoder_hidden: Tensor) -> Tensor:
        class_token_hidden = encoder_hidden[:, 0, :]
        return Layer.run_model_returning_hidden(self.output, class_token_hidden)
