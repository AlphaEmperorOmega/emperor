from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from emperor.layers import Layer, LayerStackConfig
from emperor.patch._base import PatchBase

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.patch._config import LinearPatchEmbeddingConfig


class PatchEmbeddingLinear(PatchBase):
    def __init__(
        self,
        cfg: "LinearPatchEmbeddingConfig | ModelConfig",
        overrides: "LinearPatchEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.cfg: LinearPatchEmbeddingConfig = self.cfg
        self.stride = self.cfg.stride
        self.padding = self.cfg.padding

        self.patch_model = self.__create_patch_model()
        self.embedding_model = self.__create_embedding_model()

    def __create_embedding_model(self):
        self.patch_dim = self.num_input_channels * (self.patch_size**2)
        overrides = LayerStackConfig(
            input_dim=self.patch_dim, output_dim=self.embedding_dim
        )
        return self.cfg.embedding_stack_config.build(overrides)

    def __create_patch_model(self) -> nn.Unfold:
        return nn.Unfold(
            kernel_size=self.patch_size,
            padding=self.padding,
            stride=self.stride,
        )

    def forward(self, X: Tensor):
        self.VALIDATOR.validate_forward_inputs(self, X)
        X = self.patch_model(X)
        X = X.transpose(1, 2)
        batch_size, sequence_length, patch_dim = X.shape
        X = X.reshape(batch_size * sequence_length, patch_dim)
        X = Layer.run_model_returning_hidden(self.embedding_model, X)
        X = X.reshape(batch_size, sequence_length, self.embedding_dim)
        X = self._concatenate_class_token(X)
        X = self.dropout(X)

        return X
