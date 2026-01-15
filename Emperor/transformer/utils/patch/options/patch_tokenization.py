import torch.nn as nn

from torch import Tensor
from Emperor.transformer.utils.patch.options.base import PatchBase, PatchConfig

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class PatchTokenizer(PatchBase):
    def __init__(
        self,
        cfg: "PatchConfig | ModelConfig",
        overrides: "PatchConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.patch_model = self.__create_patch_tokenizer_model()

    def __create_patch_tokenizer_model(self) -> nn.Unfold:
        return nn.Unfold(
            kernel_size=self.patch_size,
            padding=self.padding,
            stride=self.stride,
        )

    def forward(self, X: Tensor):
        X = self.patch_model(X)
        X = self._add_global_token(X)

        return X
