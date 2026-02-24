import torch
import torch.nn as nn

from typing import Dict
from torch import Tensor
from Emperor.embedding.absolute.options.base import AbsolutePositionalEmbeddingBase
from Emperor.embedding.absolute.options._validator import (
    LearnedPositionalEmbeddingValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.embedding.absolute.options.config import (
        AbsolutePositionalEmbeddingConfig,
    )


class LearnedPositionalEmbedding(AbsolutePositionalEmbeddingBase):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.num_embeddings = self._get_num_embeddings()
        self.embedding_model = self._initialize_embedding_model()

    def _get_num_embeddings(self) -> int:
        return self.num_embeddings

    def _initialize_embedding_model(self) -> nn.Embedding:
        embeddings = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            device=self.device,
        )
        nn.init.normal_(embeddings.weight, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            nn.init.constant_(embeddings.weight[self.padding_idx], 0)
        return embeddings


class TextLearnedPositionalEmbedding(LearnedPositionalEmbedding):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
    ):
        super().__init__(cfg)
        self.validator = LearnedPositionalEmbeddingValidator(self)

    def _get_num_embeddings(self) -> int:
        return self.num_embeddings + 1

    def forward(
        self,
        input_tokens: Tensor,
        incremental_state: Dict[str, Dict[str, Tensor | None]] | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        self.__validate_inputs(input_tokens, positions)
        positions = self.__resolve_positions(input_tokens, incremental_state, positions)
        return self.embedding_model(positions)

    def __validate_inputs(
        self, input_tokens: Tensor, positions: Tensor | None = None
    ) -> None:
        self.validator.ensure_propper_input_shape(input_tokens)
        self.validator.ensure_propper_input_type(input_tokens)
        self.validator.ensure_padding_index_exists_for_positions(positions)

    def __resolve_positions(
        self,
        input_tokens: Tensor,
        incremental_state: Dict[str, Dict[str, Tensor | None]] | None,
        positions: Tensor | None,
    ) -> Tensor:
        if positions is not None:
            return positions
        if incremental_state is not None:
            return self.__make_incremental_position(input_tokens)
        return self._make_positions(input_tokens)

    def __make_incremental_position(self, input_tokens: Tensor) -> Tensor:
        padding_idx: int = self.embedding_model.padding_idx
        current_decoding_step = int(padding_idx + input_tokens.size(1))
        single_step_position = torch.zeros(
            (1, 1), device=input_tokens.device, dtype=input_tokens.dtype
        )
        single_step_position = single_step_position.fill_(current_decoding_step)
        return single_step_position


class ImageLearnedPositionalEmbedding(LearnedPositionalEmbedding):
    def _get_num_embeddings(self) -> int:
        if not self.class_token_flag:
            return self.num_embeddings
        return self.num_embeddings + 1

    def forward(self, patch_embeddings: Tensor) -> Tensor:
        return self.embedding_model.weight + patch_embeddings
