import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter
from emperor.base.module import Module
from emperor.base.layer import LayerStackConfig
from emperor.base.layer.layer import Layer
from emperor.patch.core._validator import PatchValidator
from emperor.patch.core.config import (
    ConvPatchEmbeddingConfig,
    LinearPatchEmbeddingConfig,
    PatchConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class PatchBase(Module):
    def __init__(
        self,
        cfg: "PatchConfig | ModelConfig",
        overrides: "PatchConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "patch_config", cfg)
        self.cfg: "PatchConfig" = self._override_config(config, overrides)
        self.embedding_dim = self.cfg.embedding_dim
        self.patch_size = self.cfg.patch_size
        self.dropout_probability = self.cfg.dropout_probability
        self.num_input_channels = self.cfg.num_input_channels
        PatchValidator.validate(self)

        self.class_token = self._create_class_token()
        self.dropout = nn.Dropout(self.dropout_probability)

    def _create_class_token(self, shape: tuple | None = None) -> Parameter:
        if shape is None:
            shape = (1, 1, self.embedding_dim)
        class_token_init = torch.randn(shape)
        return Parameter(class_token_init, requires_grad=True)

    def _concatenate_class_token(self, X: Tensor) -> Tensor:
        batch_size = X.size(0)
        class_token = self.class_token.expand(batch_size, -1, -1)
        return torch.cat([class_token, X], dim=1)


class PatchEmbeddingLinear(PatchBase):
    def __init__(
        self,
        cfg: "LinearPatchEmbeddingConfig | ModelConfig",
        overrides: "LinearPatchEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.cfg: "LinearPatchEmbeddingConfig" = self.cfg
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
        PatchValidator.validate_forward_inputs(self, X)
        X = self.patch_model(X)
        X = X.transpose(1, 2)
        batch_size, sequence_length, patch_dim = X.shape
        X = X.reshape(batch_size * sequence_length, patch_dim)
        X = Layer.run_model_returning_hidden(self.embedding_model, X)
        X = X.reshape(batch_size, sequence_length, self.embedding_dim)
        X = self._concatenate_class_token(X)
        X = self.dropout(X)

        return X


class PatchEmbeddingConv(PatchBase):
    def __init__(
        self,
        cfg: "ConvPatchEmbeddingConfig | ModelConfig",
        overrides: "ConvPatchEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.cfg: "ConvPatchEmbeddingConfig" = self.cfg
        self.patch_model = self.__create_patch_extraction_model()

    def __create_patch_extraction_model(self):
        overrides = LayerStackConfig(
            input_dim=self.num_input_channels,
            output_dim=self.embedding_dim,
        )
        return self.cfg.conv_stack_config.build(overrides)

    def forward(self, X: Tensor):
        PatchValidator.validate_forward_inputs(self, X)
        X = Layer.run_model_returning_hidden(self.patch_model, X)
        X = X.flatten(2)
        X = X.transpose(1, 2)
        X = self._concatenate_class_token(X)
        X = self.dropout(X)

        return X
