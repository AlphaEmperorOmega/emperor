import math
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from emperor.embedding.absolute._base import AbsolutePositionalEmbeddingBase

if TYPE_CHECKING:
    from emperor.embedding.absolute._config import (
        AbsolutePositionalEmbeddingConfig,
        ImageSinusoidalPositionalEmbeddingConfig,
    )


class SinusoidalPositionalEmbedding(AbsolutePositionalEmbeddingBase):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
        overrides: "AbsolutePositionalEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.position_offset = self._get_padding_idx()
        self.init_size = self._get_init_size()
        self._register_positional_embedding_tensor()

    def _get_padding_idx(self) -> int:
        return self.padding_idx or 0

    def _get_init_size(self) -> int:
        return self.num_embeddings + self.position_offset + 1

    def _register_positional_embedding_tensor(self):
        embeddings = self._get_embedding(self.init_size)
        self.register_buffer("weights", embeddings, persistent=False)

    def _get_embedding(self, num_embeddings: int) -> Tensor:
        embedding = self._compute_embedding_tensor(self.embedding_dim, num_embeddings)
        embedding = self._maybe_add_odd_dim_padding(
            embedding, self.embedding_dim, num_embeddings
        )
        return self._maybe_mask_padding_index(embedding, self.padding_idx)

    def _compute_embedding_tensor(
        self, embedding_dim: int, num_embeddings: int
    ) -> Tensor:
        half_dim = embedding_dim // 2
        frequency_scale = 0.0 if half_dim <= 1 else math.log(10000) / (half_dim - 1)
        frequency_exponents = torch.arange(half_dim) * -frequency_scale
        frequencies = torch.exp(frequency_exponents)
        position_range = torch.arange(num_embeddings).unsqueeze(1)
        scaled_positions = position_range * frequencies.unsqueeze(0)
        embedding = torch.cat(
            [torch.sin(scaled_positions), torch.cos(scaled_positions)], dim=1
        )
        return embedding.view(num_embeddings, -1)

    def _maybe_add_odd_dim_padding(
        self, embedding: Tensor, embedding_dim: int, num_embeddings: int
    ) -> Tensor:
        if embedding_dim % 2 == 1:
            padding_vector = torch.zeros(num_embeddings, 1)
            embedding = torch.cat([embedding, padding_vector], dim=1)
        return embedding

    def _maybe_mask_padding_index(
        self, embedding: Tensor, padding_idx: int | None
    ) -> Tensor:
        if padding_idx is not None:
            embedding[padding_idx, :] = 0
        return embedding


class TextSinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding):
    def forward(
        self,
        input_tokens: Tensor,
        incremental_state: Any = None,
        timestep: Tensor | None = None,
    ) -> Tensor:
        self.VALIDATOR.validate_text_tokens(input_tokens)
        batch_size, sequence_length = input_tokens.size()
        if incremental_state is not None:
            self.VALIDATOR.validate_incremental_sequence_length(sequence_length)
        self.__maybe_expand_weights(
            input_tokens,
            incremental_state=incremental_state,
            timestep=timestep,
        )

        if incremental_state is not None:
            return self.__forward_incremental(batch_size, sequence_length, timestep)
        return self.__forward_full_sequence(input_tokens, batch_size, sequence_length)

    def __maybe_expand_weights(
        self,
        input_tokens: Tensor,
        *,
        incremental_state: Any,
        timestep: Tensor | None,
    ) -> None:
        _, sequence_length = input_tokens.size()
        if incremental_state is not None and timestep is not None:
            sequence_length = int(timestep.flatten()[0].item()) + 1
        max_positions = self.position_offset + 1 + sequence_length
        if self.auto_expand_flag and max_positions > self.weights.size(0):
            self.weights = self._get_embedding(max_positions).to(
                device=self.weights.device,
                dtype=self.weights.dtype,
            )

    def __forward_incremental(
        self, batch_size: int, sequence_length: int, timestep: Tensor | None
    ) -> Tensor:
        current_position = sequence_length
        if timestep is not None:
            current_position = timestep.flatten()[0] + 1
        single_step_weights = self.weights[self.position_offset + current_position, :]
        return single_step_weights.expand(batch_size, 1, -1)

    def __forward_full_sequence(
        self, input_tokens: Tensor, batch_size: int, sequence_length: int
    ) -> Tensor:
        positions = self._make_positions(input_tokens)
        selected_weights = self.weights.index_select(0, positions.view(-1))
        return selected_weights.view(batch_size, sequence_length, -1).detach()


class ImageSinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding):
    def __init__(
        self,
        cfg: "ImageSinusoidalPositionalEmbeddingConfig",
        overrides: "ImageSinusoidalPositionalEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.class_token_flag: bool = self.cfg.class_token_flag

    def _get_init_size(self) -> int:
        return self.num_embeddings + int(self.cfg.class_token_flag)

    def forward(self, patch_embeddings: Tensor) -> Tensor:
        self.VALIDATOR.validate_patch_embeddings(
            patch_embeddings,
            num_embeddings=self.weights.size(0),
            embedding_dim=self.embedding_dim,
        )
        return patch_embeddings + self.weights
