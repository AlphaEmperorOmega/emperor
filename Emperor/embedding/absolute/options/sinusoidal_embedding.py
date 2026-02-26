import math
import torch

from torch import Tensor
from typing import Any
from Emperor.embedding.absolute.options.base import AbsolutePositionalEmbeddingBase
from Emperor.embedding.absolute.options._validator import (
    SinusoidalPositionalEmbeddingValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.embedding.absolute.config import (
        AbsolutePositionalEmbeddingConfig,
    )


class SinusoidalPositionalEmbedding(AbsolutePositionalEmbeddingBase):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.padding_idx: int = self._get_padding_idx(cfg)
        self.init_size: int = self._get_init_size()

        self._register_positional_embedding_tensor()
        self.validator = SinusoidalPositionalEmbeddingValidator(self)

    def _get_padding_idx(self, cfg: "AbsolutePositionalEmbeddingConfig") -> int:
        is_padding_idx = cfg.padding_idx is not None
        return cfg.padding_idx if is_padding_idx else 0

    def _get_init_size(self) -> int:
        return self.num_embeddings + self.padding_idx + 1

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
        frequency_scale = math.log(10000) / (half_dim - 1)
        frequency_exponents = (
            torch.arange(half_dim, dtype=torch.float) * -frequency_scale
        )
        frequencies = torch.exp(frequency_exponents)
        position_range = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)
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
        self.validator.ensure_propper_input_shape(input_tokens)
        self.validator.ensure_propper_input_type(input_tokens)
        batch_size, sequence_length = input_tokens.size()

        self.__maybe_expand_weights(input_tokens)

        if incremental_state is not None:
            return self.__forward_incremental(batch_size, sequence_length, timestep)
        return self.__forward_full_sequence(input_tokens, batch_size, sequence_length)

    def __maybe_expand_weights(self, input_tokens: Tensor) -> None:
        _, sequence_length = input_tokens.size()
        max_positions = self.padding_idx + 1 + sequence_length
        if self.auto_expand_flag and max_positions > self.weights.size(0):
            self.weights = self._get_embedding(max_positions).to(self.weights.device)

    def __forward_incremental(
        self, batch_size: int, sequence_length: int, timestep: Tensor | None
    ) -> Tensor:
        current_position = sequence_length
        if timestep is not None:
            beam_absolute_position = timestep.view(-1)[0] + 1
            current_position = beam_absolute_position
        single_step_weights = self.weights[self.padding_idx + current_position, :]
        return single_step_weights.expand(batch_size, 1, -1)

    def __forward_full_sequence(
        self, input_tokens: Tensor, batch_size: int, sequence_length: int
    ) -> Tensor:
        positions = self._make_positions(input_tokens)
        selected_weights = self.weights.index_select(0, positions.view(-1))
        return selected_weights.view(batch_size, sequence_length, -1).detach()


class ImageSinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding):
    def forward(self, patch_embeddings: Tensor) -> Tensor:
        return patch_embeddings + self.weights
