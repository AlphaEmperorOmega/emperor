import torch.nn as nn

from torch import Tensor
from emperor.base.layer import LayerStack, LayerStackConfig
from emperor.transformer.utils.patch.options.base import PatchBase, PatchConfig

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from emperor.config import ModelConfig


class PatchEmbeddingLinear(PatchBase):
    def __init__(
        self,
        cfg: "PatchConfig | ModelConfig",
        overrides: "PatchConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.patch_model = self.__create_patch_model()
        self.embedding_model = self.__create_embedding_model()

    def __create_embedding_model(self):
        self.patch_dim = self.num_input_channels * (self.patch_size**2)
        overrides = LayerStackConfig(
            input_dim=self.patch_dim, output_dim=self.embedding_dim
        )
        return LayerStack(self.main_cfg, overrides).build_model()

    def __create_patch_model(self) -> nn.Unfold:
        return nn.Unfold(
            kernel_size=self.patch_size,
            padding=self.padding,
            stride=self.stride,
        )

    def forward(self, X: Tensor):
        X = self.patch_model(X)
        X = X.transpose(1, 2)
        X = self.embedding_model(X)
        X = self._concatenate_class_token(X)
        X = self.dropout(X)

        return X
