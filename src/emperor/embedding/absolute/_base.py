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
        self.cfg: AbsolutePositionalEmbeddingConfig = self._override_config(
            cfg, overrides
        )
        self.VALIDATOR.validate(self)

        self.embedding_dim: int = self.cfg.embedding_dim
        self.num_embeddings: int = self.cfg.num_embeddings

    def _make_positions(
        self,
        input_tokens: Tensor,
        padding_idx: int | None,
    ) -> Tensor:
        if padding_idx is None:
            return (
                torch.arange(
                    input_tokens.size(1),
                    device=input_tokens.device,
                )
                .unsqueeze(0)
                .expand_as(input_tokens)
            )
        non_padding_mask = input_tokens.ne(padding_idx).int()
        cumulative_positions = torch.cumsum(non_padding_mask, dim=1).type_as(
            non_padding_mask
        )
        cumulative_positions = cumulative_positions * non_padding_mask
        return cumulative_positions.long() + padding_idx
