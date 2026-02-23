import torch

from torch import Tensor
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.embedding.relative.options.config import (
        RelativePositionalEmbeddingConfig,
    )


class LearnedPositionalBias(Module):
    def __init__(
        self,
        cfg: "RelativePositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.text_processing_flag = self.cfg.text_processing_flag
        self.embedding_dim = self.cfg.embedding_dim
        self.padding_idx = self.cfg.padding_idx
        self.num_embeddings = self.__get_num_embeddings()
        self.init_size = self.cfg.init_size
        self.auto_expand_flag = self.cfg.auto_expand_flag
        self.max_positions = self.cfg.max_positions
        self.num_heads = self.cfg.num_heads
        self.head_dim = self.embedding_dim // self.num_heads

        embeding_shape = (self.num_heads, self.head_dim, self.max_positions * 2 + 1)
        self.relative_positional_emnbeddings = self._init_parameter_bank(embeding_shape)

    def __get_num_embeddings(self) -> int:
        if self.padding_idx is None:
            return self.cfg.num_embeddings
        return self.cfg.num_embeddings + self.padding_idx + 1

    def forward(
        self,
        query: Tensor,
        length: int,
        last: bool = False,
    ) -> Tensor:
        logits = torch.einsum(
            "nhid,hdj->nhij", query, self.relative_positional_emnbeddings
        )
        embedding_grid = self.__compute_embedding_grid(length, last)
        relative_offsets = self.__compute_relative_offsets(embedding_grid, logits)
        output = logits.gather(-1, relative_offsets)
        return output

    def __compute_embedding_grid(self, length: int, last: bool) -> Tensor:
        indices = torch.arange(length, dtype=torch.long, device=self.device)
        if not last:
            return indices[None, :] - indices[:, None]
        return indices[None, :] - (length - 1)

    def __compute_relative_offsets(
        self, embedding_grid: Tensor, logits: Tensor
    ) -> Tensor:
        min_valid_offset = 1 - self.max_positions
        max_valid_offset = self.max_positions - 1
        bounded_offsets = torch.clamp(
            embedding_grid, min=min_valid_offset, max=max_valid_offset
        )
        table_indices = bounded_offsets + self.max_positions
        broadcastable_indices = table_indices[None, None, :, :]
        batch_size, num_heads = logits.size(0), logits.size(1)
        expanded_indices = broadcastable_indices.expand(batch_size, num_heads, -1, -1)

        return expanded_indices
