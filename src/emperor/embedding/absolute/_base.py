from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.embedding.absolute._validation import (
    AbsolutePositionalEmbeddingValidator,
)
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.embedding.absolute._config import AbsolutePositionalEmbeddingConfig


class AbsolutePositionalEmbeddingBase(Module):
    VALIDATOR = AbsolutePositionalEmbeddingValidator

    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
        overrides: "AbsolutePositionalEmbeddingConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "absolute_positional_embedding_config", cfg)
        config = getattr(config, "positional_embedding_config", config)
        self.cfg: AbsolutePositionalEmbeddingConfig = self._override_config(
            config, overrides
        )
        self.VALIDATOR.validate(self)

        self.embedding_dim: int = self.cfg.embedding_dim
        self.padding_idx: int | None = self.cfg.padding_idx
        self.num_embeddings: int = self.cfg.num_embeddings
        self.init_size: int = self.cfg.init_size
        self.auto_expand_flag: bool = self.cfg.auto_expand_flag

    def _make_positions(self, input_tokens: Tensor) -> Tensor:
        if self.padding_idx is None:
            return (
                torch.arange(
                    input_tokens.size(1),
                    device=input_tokens.device,
                )
                .unsqueeze(0)
                .expand_as(input_tokens)
            )
        non_padding_mask = input_tokens.ne(self.padding_idx).int()
        cumulative_positions = torch.cumsum(non_padding_mask, dim=1).type_as(
            non_padding_mask
        )
        cumulative_positions = cumulative_positions * non_padding_mask
        return cumulative_positions.long() + self.padding_idx
