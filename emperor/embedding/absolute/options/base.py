import torch

from torch import Tensor
from emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.embedding.options import AbsolutePositionalEmbeddingOptions
    from emperor.embedding.absolute.config import (
        AbsolutePositionalEmbeddingConfig,
    )


class AbsolutePositionalEmbeddingBase(Module):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.positional_embedding_option: "AbsolutePositionalEmbeddingOptions" = (
            self.cfg.positional_embedding_option
        )
        self.class_token_flag: bool = self.cfg.class_token_flag
        self.embedding_dim: int = self.cfg.embedding_dim
        self.padding_idx: int = self.cfg.padding_idx
        self.num_embeddings: int = self.cfg.num_embeddings
        self.init_size: int = self.cfg.init_size
        self.auto_expand_flag: bool = self.cfg.auto_expand_flag

    def _make_positions(self, input: Tensor) -> Tensor:
        non_padding_mask = input.ne(self.padding_idx).int()
        cumulative_positions = torch.cumsum(non_padding_mask, dim=1).type_as(
            non_padding_mask
        )
        cumulative_positions = cumulative_positions * non_padding_mask
        return cumulative_positions.long() + self.padding_idx
