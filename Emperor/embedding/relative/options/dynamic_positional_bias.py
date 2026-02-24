import torch

from torch import Tensor
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.embedding.relative.options.config import (
        RelativePositionalEmbeddingConfig,
    )


class DynamicPostionalBias(Module):
    def __init__(
        self,
        cfg: "RelativePositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.text_processing_flag: bool = self.cfg.text_processing_flag
        self.embedding_dim: int = self.cfg.embedding_dim
        self.padding_idx: int = self.cfg.padding_idx
        self.num_embeddings: int = self.__get_num_embeddings()
        self.init_size: int = self.cfg.init_size
        self.auto_expand_flag: bool = self.cfg.auto_expand_flag
        self.max_positions: int = self.cfg.max_positions
        self.num_heads: int = self.cfg.num_heads
        self.head_dim: int = self.embedding_dim // self.num_heads

        embeding_shape = (self.num_heads, self.head_dim, self.max_positions * 2 + 1)
        self.relative_positional_emnbeddings = self._init_parameter_bank(embeding_shape)

    def __get_num_embeddings(self) -> int:
        num_embeddings: int = self.cfg.num_embeddings
        if self.padding_idx is None:
            return num_embeddings
        return num_embeddings + self.padding_idx + 1

    def forward(
        self,
        query: Tensor,
        sequence_length: int,
        last: bool = False,
    ) -> Tensor:
        logits = torch.einsum(
            "nhid,hdj->nhij", query, self.relative_positional_emnbeddings
        )
        embedding_grid = self.__compute_embedding_grid(sequence_length, last)
        relative_offsets = self.__compute_relative_offsets(embedding_grid, logits)
        output = logits.gather(-1, relative_offsets)
        return output

    def __compute_embedding_grid(self, sequence_length: int, last: bool) -> Tensor:
        indices = torch.arange(sequence_length, dtype=torch.long, device=self.device)
        if not last:
            return indices[None, :] - indices[:, None]
        return indices[None, :] - (sequence_length - 1)

    def __compute_relative_offsets(
        self, embedding_grid: Tensor, logits: Tensor
    ) -> Tensor:
        min_offset = 1 - self.max_positions
        max_offset = self.max_positions - 1
        bounded_offsets = torch.clamp(embedding_grid, min=min_offset, max=max_offset)
        table_indices = bounded_offsets + self.max_positions
        broadcastable_indices = table_indices[None, None, :, :]
        batch_size, num_heads = logits.size(0), logits.size(1)
        expanded_indices = broadcastable_indices.expand(batch_size, num_heads, -1, -1)

        return expanded_indices
