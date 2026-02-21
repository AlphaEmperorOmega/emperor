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
    from Emperor.embedding.absolute.options.config import (
        AbsolutePositionalEmbeddingConfig,
    )


class SinusoidalPositionalEmbedding(AbsolutePositionalEmbeddingBase):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim: bool = self.cfg.embedding_dim
        self.num_embeddings: int = self.cfg.num_embeddings
        self.padding_idx: int = self._get_padding_idx(cfg)
        self.init_size: int = self._get_init_size()
        self.auto_expand_flag: bool = self.cfg.auto_expand_flag

        self._register_positional_embedding_tensor()
        self.validator = SinusoidalPositionalEmbeddingValidator(self)

    def _get_padding_idx(self, cfg: "AbsolutePositionalEmbeddingConfig") -> int:
        is_padding_idx = cfg.padding_idx is not None
        return cfg.padding_idx if is_padding_idx else 0

    def _get_init_size(self) -> int:
        return self.num_embeddings + self.padding_idx + 1

    def _register_positional_embedding_tensor(self):
        embeddings = self._get_embedding(
            self.init_size, self.embedding_dim, self.padding_idx
        )
        self.register_buffer("weights", embeddings, persistent=False)

    def _get_embedding(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> Tensor:
        embedding = self._compute_embedding_tensor(embedding_dim, num_embeddings)
        embedding = self._maybe_add_odd_dim_padding(
            embedding, embedding_dim, num_embeddings
        )
        return self._maybe_mask_padding_index(embedding, padding_idx)

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
        input: Tensor,
        incremental_state: Any = None,
        timestep: Tensor | None = None,
    ) -> Tensor:
        self.validator.ensure_propper_input_shape(input)
        self.validator.ensure_propper_input_type(input)
        batch_size, sequence_length = input.size()
        max_positions = self.padding_idx + 1 + sequence_length

        self.__maybe_expand_weights(max_positions)

        if incremental_state is not None:
            return self.__forward_incremental(batch_size, sequence_length, timestep)

        return self.__forward_full_sequence(input, batch_size, sequence_length)

    def __maybe_expand_weights(self, max_positions: int) -> None:
        if max_positions > self.weights.size(0):
            expanded_weights = self._get_embedding(max_positions).to(self.weights)
            if self.auto_expand_flag:
                self.weights = expanded_weights

    def __forward_incremental(
        self, batch_size: int, sequence_length: int, timestep: Tensor | None
    ) -> Tensor:
        is_timestep = timestep is not None
        current_position = timestep.view(-1)[0] + 1 if is_timestep else sequence_length
        single_step_weights = self.weights[self.padding_idx + current_position, :]
        return single_step_weights.expand(batch_size, 1, -1)

    def __forward_full_sequence(
        self, input: Tensor, batch_size: int, sequence_length: int
    ) -> Tensor:
        positions = self.__make_positions(input)
        selected_weights = self.weights.index_select(0, positions.view(-1))
        return selected_weights.view(batch_size, sequence_length, -1).detach()

    def __make_positions(self, tensor: Tensor) -> Tensor:
        non_padding_mask = tensor.ne(self.padding_idx).int()
        cumulative_positions = torch.cumsum(non_padding_mask, dim=1).type_as(
            non_padding_mask
        )
        cumulative_positions = cumulative_positions * non_padding_mask
        embedding_table_indices = cumulative_positions.long() + self.padding_idx
        return embedding_table_indices


class ImageSinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding):
    def forward(self, input: Tensor) -> Tensor:
        return input + self.weights
