from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from emperor.nn import Module
from emperor.patch._validation import PatchValidator

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.patch._config import PatchConfig


class PatchBase(Module):
    VALIDATOR = PatchValidator

    def __init__(
        self,
        cfg: "PatchConfig | ModelConfig",
        overrides: "PatchConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "patch_config", cfg)
        self.cfg: PatchConfig = self._override_config(config, overrides)
        self.embedding_dim = self.cfg.embedding_dim
        self.patch_size = self.cfg.patch_size
        self.dropout_probability = self.cfg.dropout_probability
        self.num_input_channels = self.cfg.num_input_channels
        self.VALIDATOR.validate(self)

        self.class_token = self._create_class_token()
        self.dropout = nn.Dropout(self.dropout_probability)

    def _create_class_token(self, shape: tuple | None = None) -> Parameter:
        if shape is None:
            shape = (1, 1, self.embedding_dim)
        class_token_init = torch.randn(shape)
        return Parameter(class_token_init)

    def _concatenate_class_token(self, X: Tensor) -> Tensor:
        batch_size = X.size(0)
        class_token = self.class_token.expand(batch_size, -1, -1)
        return torch.cat([class_token, X], dim=1)
