import torch.nn as nn
from torch import Tensor
from Emperor.transformer.utils.patch.options.base import PatchBase, PatchConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class PatchEmbeddingConv(PatchBase):
    def __init__(
        self,
        cfg: "PatchConfig | ModelConfig",
        overrides: "PatchConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.patch_model = self.__create_patch_extraction_model()

    def __create_patch_extraction_model(self):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=self.cfg.input_channels,
                out_channels=self.cfg.embedding_dim,
                kernel_size=self.cfg.patch_size,
                stride=self.cfg.patch_size,
            ),
            nn.Flatten(2),
        )

    def forward(self, X: Tensor):
        X = self.patch_model(X)
        X = X.transpose(1, 2)
        X = self._concatenate_class_token(X)
        X = self.dropout(X)

        return X
