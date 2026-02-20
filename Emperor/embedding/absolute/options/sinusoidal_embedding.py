import math
import torch

from torch import Tensor
from typing import Any
from Emperor.base.utils import Module
from Emperor.transformer.utils.embedding.options._validator import (
    SinusoidalPositionalEmbeddingValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.transformer.utils.embedding.selector import PositionalEmbeddingConfig


class SinusoidalPositionalEmbedding(Module):
    def __init__(
        self,
        cfg: "PositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = self.cfg.embedding_dim
        self.num_embeddings = self.cfg.num_embeddings
        self.padding_idx = self.__get_padding_idx(cfg)
        self.init_size = self.__get_init_size()
        self.auto_expand_flag = self.cfg.auto_expand_flag

        self.__register_positional_embedding_tensor()
        self.validator = SinusoidalPositionalEmbeddingValidator(self)

    def __get_padding_idx(self, cfg: "PositionalEmbeddingConfig") -> int:
        is_padding_idx = cfg.padding_idx is not None
        return cfg.padding_idx if is_padding_idx else 0

    def __get_init_size(self) -> int:
        return self.num_embeddings + self.padding_idx + 1

    def __register_positional_embedding_tensor(self):
        embeddings = self.__get_embedding(
            self.init_size, self.embedding_dim, self.padding_idx
        )
        self.register_buffer("weights", embeddings, persistent=False)

    def __get_embedding(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> Tensor:
        embedding = self.__compute_embedding_tensor(embedding_dim, num_embeddings)
        embedding = self.__maybe_add_padding(embedding, embedding_dim, num_embeddings)
        return self.__maybe_mask_padding_index(embedding, padding_idx)

    def __compute_embedding_tensor(
        self, embedding_dim: int, num_embeddings: int
    ) -> Tensor:
        half_dim = embedding_dim // 2
        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.arange(half_dim, dtype=torch.float) * -embedding
        embedding = torch.exp(embedding)
        embedding_range = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)
        embedding = embedding_range * embedding.unsqueeze(0)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        return embedding.view(num_embeddings, -1)

    def __maybe_add_padding(
        self, embedding, embedding_dim: int, num_embeddings: int
    ) -> Tensor:
        if embedding_dim % 2 == 1:
            padding_vector = torch.zeros(num_embeddings, 1)
            embedding = torch.cat([embedding, padding_vector], dim=1)
        return embedding

    def __maybe_mask_padding_index(
        self, embedding: Tensor, padding_idx: int | None
    ) -> Tensor:
        if padding_idx is not None:
            embedding[padding_idx, :] = 0
        return embedding

    def forward(
        self,
        input: Tensor,
        incremental_state: Any = None,
        timestep: Tensor | None = None,
        position: Any | None = None,
    ) -> Tensor:
        self.validator.ensure_propper_input_shape(input)
        self.validator.ensure_propper_input_type(input)
        batch_size, sequence_length = input.size()
        max_positions = self.padding_idx + 1 + sequence_length

        weights = self.__update_weights(max_positions, self.weights)

        if incremental_state is not None:
            is_timestep = timestep is not None
            position = timestep.view(-1)[0] + 1 if is_timestep else sequence_length
            weights = self.weights[self.padding_idx + position, :]
            return weights.expand(batch_size, 1, -1)

        position = self.__make_positions(input)

        weights = self.weights.index_select(0, position.view(-1))
        return weights.view(batch_size, sequence_length, -1).detach()

    def __update_weights(self, max_positions: int, weights: Tensor) -> Tensor:
        if max_positions > self.weights.size(0):
            weights = self.__get_embedding(max_positions).to(self.weights)
            if self.auto_expand:
                self.weights = weights
        return weights

    def __make_positions(self, tensor: Tensor) -> Tensor:
        mask = tensor.ne(self.padding_idx).int()
        mask = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return mask.long() + self.padding_idx
