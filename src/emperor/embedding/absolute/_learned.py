from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.embedding.absolute._base import AbsolutePositionalEmbeddingBase

if TYPE_CHECKING:
    from emperor.embedding.absolute._config import (
        AbsolutePositionalEmbeddingConfig,
        ImageLearnedPositionalEmbeddingConfig,
    )


class LearnedPositionalEmbedding(AbsolutePositionalEmbeddingBase):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
        overrides: "AbsolutePositionalEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.num_embeddings = self._get_num_embeddings()
        self.embedding_model = self.__initialize_embedding_model()

    def _get_num_embeddings(self) -> int:
        return self.num_embeddings

    def __initialize_embedding_model(self) -> nn.Embedding:
        embeddings = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
        )
        nn.init.normal_(embeddings.weight, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            nn.init.constant_(embeddings.weight[self.padding_idx], 0)
        return embeddings


class TextLearnedPositionalEmbedding(LearnedPositionalEmbedding):
    def _get_num_embeddings(self) -> int:
        padding_offset = self.padding_idx if self.padding_idx is not None else 0
        return self.num_embeddings + padding_offset + 1

    def forward(
        self,
        input_tokens: Tensor,
        incremental_state: dict[str, dict[str, Tensor | None]] | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        self.VALIDATOR.validate_text_tokens(input_tokens)
        if positions is not None:
            self.VALIDATOR.validate_positions(
                positions,
                expected_shape=tuple(input_tokens.shape),
                num_embeddings=self.embedding_model.num_embeddings,
            )
        positions = self.__resolve_positions(input_tokens, incremental_state, positions)
        return self.embedding_model(positions)

    def __resolve_positions(
        self,
        input_tokens: Tensor,
        incremental_state: dict[str, dict[str, Tensor | None]] | None,
        positions: Tensor | None,
    ) -> Tensor:
        if positions is not None:
            return positions
        if incremental_state is not None:
            return self.__make_incremental_position(input_tokens)
        return self._make_positions(input_tokens)

    def __make_incremental_position(self, input_tokens: Tensor) -> Tensor:
        padding_idx = self.embedding_model.padding_idx or 0
        current_decoding_step = int(padding_idx + input_tokens.size(1))
        single_step_position = input_tokens.new_full(
            (input_tokens.size(0), 1),
            current_decoding_step,
            dtype=torch.long,
        )
        return single_step_position


class ImageLearnedPositionalEmbedding(LearnedPositionalEmbedding):
    def __init__(
        self,
        cfg: "ImageLearnedPositionalEmbeddingConfig",
        overrides: "ImageLearnedPositionalEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.class_token_flag: bool = self.cfg.class_token_flag

    def _get_num_embeddings(self) -> int:
        if not self.cfg.class_token_flag:
            return self.num_embeddings
        return self.num_embeddings + 1

    def forward(self, patch_embeddings: Tensor) -> Tensor:
        self.VALIDATOR.validate_patch_embeddings(
            patch_embeddings,
            num_embeddings=self.embedding_model.num_embeddings,
            embedding_dim=self.embedding_dim,
        )
        return self.embedding_model.weight + patch_embeddings
