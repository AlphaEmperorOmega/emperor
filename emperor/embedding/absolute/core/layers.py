import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Any
from emperor.base.utils import Module
from emperor.embedding.absolute.core._validator import (
    AbsolutePositionalEmbeddingValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.embedding.absolute.core.config import (
        AbsolutePositionalEmbeddingConfig,
        ImageLearnedPositionalEmbeddingConfig,
        ImageSinusoidalPositionalEmbeddingConfig,
    )


class AbsolutePositionalEmbeddingBase(Module):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
        overrides: "AbsolutePositionalEmbeddingConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "absolute_positional_embedding_config", cfg)
        config = getattr(config, "positional_embedding_config", config)
        self.cfg: "AbsolutePositionalEmbeddingConfig" = self._override_config(
            config, overrides
        )
        AbsolutePositionalEmbeddingValidator.validate_config(self.cfg)

        self.embedding_dim: int = self.cfg.embedding_dim
        self.padding_idx: int | None = self.cfg.padding_idx
        self.num_embeddings: int = self.cfg.num_embeddings
        self.init_size: int = self.cfg.init_size
        self.auto_expand_flag: bool = self.cfg.auto_expand_flag

    def _make_positions(self, input_tokens: Tensor) -> Tensor:
        if self.padding_idx is None:
            return torch.arange(
                input_tokens.size(1),
                device=input_tokens.device,
                dtype=torch.long,
            ).unsqueeze(0).expand_as(input_tokens)
        non_padding_mask = input_tokens.ne(self.padding_idx).int()
        cumulative_positions = torch.cumsum(non_padding_mask, dim=1).type_as(
            non_padding_mask
        )
        cumulative_positions = cumulative_positions * non_padding_mask
        return cumulative_positions.long() + self.padding_idx


class LearnedPositionalEmbedding(AbsolutePositionalEmbeddingBase):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
        overrides: "AbsolutePositionalEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.num_embeddings = self._get_num_embeddings()
        self.embedding_model = self._initialize_embedding_model()

    def _get_num_embeddings(self) -> int:
        return self.num_embeddings

    def _initialize_embedding_model(self) -> nn.Embedding:
        embeddings = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
        )
        nn.init.normal_(embeddings.weight, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            nn.init.constant_(embeddings.weight[self.padding_idx], 0)
        return embeddings


class TextLearnedPositionalEmbedding(LearnedPositionalEmbedding):
    def _get_num_embeddings(self) -> int:
        return self.num_embeddings + 1

    def forward(
        self,
        input_tokens: Tensor,
        incremental_state: dict[str, dict[str, Tensor | None]] | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        AbsolutePositionalEmbeddingValidator.validate_text_tokens(input_tokens)
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
        single_step_position = torch.zeros(
            (1, 1), device=input_tokens.device, dtype=input_tokens.dtype
        )
        single_step_position = single_step_position.fill_(current_decoding_step)
        return single_step_position.long()


class ImageLearnedPositionalEmbedding(LearnedPositionalEmbedding):
    def __init__(
        self,
        cfg: "ImageLearnedPositionalEmbeddingConfig",
        overrides: "ImageLearnedPositionalEmbeddingConfig | None" = None,
    ):
        config = getattr(cfg, "absolute_positional_embedding_config", cfg)
        config = getattr(config, "positional_embedding_config", config)
        config = self._override_config(config, overrides)
        AbsolutePositionalEmbeddingValidator.validate_image_config(config)
        super().__init__(config)
        self.class_token_flag: bool = self.cfg.class_token_flag

    def _get_num_embeddings(self) -> int:
        if not self.cfg.class_token_flag:
            return self.num_embeddings
        return self.num_embeddings + 1

    def forward(self, patch_embeddings: Tensor) -> Tensor:
        AbsolutePositionalEmbeddingValidator.validate_patch_embeddings(
            patch_embeddings
        )
        return self.embedding_model.weight + patch_embeddings


class SinusoidalPositionalEmbedding(AbsolutePositionalEmbeddingBase):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
        overrides: "AbsolutePositionalEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.padding_idx = self._get_padding_idx()
        self.init_size = self._get_init_size()
        self._register_positional_embedding_tensor()

    def _get_padding_idx(self) -> int:
        return self.padding_idx if self.padding_idx is not None else 0

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
        AbsolutePositionalEmbeddingValidator.validate_text_tokens(input_tokens)
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
            current_position = timestep.view(-1)[0] + 1
        single_step_weights = self.weights[self.padding_idx + current_position, :]
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
        config = getattr(cfg, "absolute_positional_embedding_config", cfg)
        config = getattr(config, "positional_embedding_config", config)
        config = self._override_config(config, overrides)
        AbsolutePositionalEmbeddingValidator.validate_image_config(config)
        super().__init__(config)
        self.class_token_flag: bool = self.cfg.class_token_flag

    def forward(self, patch_embeddings: Tensor) -> Tensor:
        AbsolutePositionalEmbeddingValidator.validate_patch_embeddings(
            patch_embeddings
        )
        return patch_embeddings + self.weights
