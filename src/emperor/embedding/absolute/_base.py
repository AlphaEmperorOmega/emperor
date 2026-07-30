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
            return self.__make_unpadded_positions(input_tokens)
        return self.__make_padding_aware_positions(input_tokens, padding_idx)

    def __make_unpadded_positions(self, input_tokens: Tensor) -> Tensor:
        sequence_length = input_tokens.size(1)
        position_indices = torch.arange(
            sequence_length,
            device=input_tokens.device,
        )
        single_sequence_positions = position_indices.unsqueeze(0)
        return single_sequence_positions.expand_as(input_tokens)

    def __make_padding_aware_positions(
        self,
        input_tokens: Tensor,
        padding_idx: int,
    ) -> Tensor:
        non_padding_mask = input_tokens.ne(padding_idx).int()
        cumulative_positions = torch.cumsum(non_padding_mask, dim=1).type_as(
            non_padding_mask
        )
        cumulative_positions = cumulative_positions * non_padding_mask
        return cumulative_positions.long() + padding_idx
