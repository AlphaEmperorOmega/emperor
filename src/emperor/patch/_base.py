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

        self.class_token_flag = self.cfg.class_token_flag is not False
        if self.class_token_flag:
            self.class_token = self._create_class_token()
        self.dropout = nn.Dropout(self.dropout_probability)

    def _create_class_token(self, shape: tuple | None = None) -> Parameter:
        class_token_shape = shape
        if class_token_shape is None:
            class_token_shape = (1, 1, self.embedding_dim)
        initial_class_token_values = torch.randn(class_token_shape)
        class_token = Parameter(initial_class_token_values)
        return class_token

    def _concatenate_class_token(self, X: Tensor) -> Tensor:
        if not self.class_token_flag:
            return X
        batch_size = X.size(0)
        expanded_class_tokens = self.class_token.expand(batch_size, -1, -1)
        patches_with_class_token = torch.cat([expanded_class_tokens, X], dim=1)
        return patches_with_class_token
