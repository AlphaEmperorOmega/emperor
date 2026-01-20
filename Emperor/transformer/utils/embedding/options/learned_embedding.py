import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict
from Emperor.base.utils import Module
from Emperor.transformer.utils.embedding.options._validator import (
    LearnedPositionalEmbeddingValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.transformer.utils.embedding.selector import PositionalEmbeddingConfig


class LearnedPositionalEmbedding(Module):
    def __init__(
        self,
        cfg: "PositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.text_processing_flag = self.cfg.text_processing_flag
        self.embedding_dim = self.cfg.embedding_dim
        self.padding_idx = self.cfg.padding_idx
        self.num_embeddings = self.__get_num_embeddings(cfg)
        self.init_size = self.cfg.init_size
        self.auto_expand_flag = self.cfg.auto_expand_flag

        self.embedding_model = self.__initialize_embedding_model()
        self.validator = LearnedPositionalEmbeddingValidator(self)

    def __get_num_embeddings(self, cfg: "PositionalEmbeddingConfig") -> int:
        if self.cfg.padding_idx is None:
            return cfg.num_embeddings
        return cfg.num_embeddings + self.cfg.padding_idx + 1

    def __initialize_embedding_model(self) -> nn.Embedding:
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

    def forward(
        self,
        input: Tensor,
        incremental_state: Dict[str, Dict[str, Tensor | None]] | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        if self.text_processing_flag:
            return self.__process_text(input, incremental_state, positions)
        return self.__process_images(input)

    def __process_text(
        self,
        input: Tensor,
        incremental_state: Dict[str, Dict[str, Tensor | None]] | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        self.validator.ensure_propper_input_shape(input)
        self.validator.ensure_propper_input_type(input)
        self.validator.ensure_padding_index_exists_for_positions(positions)

        positions = self.__resolve_positions(input, incremental_state, positions)
        return self.embedding_model(positions)

    def __process_images(
        self,
        input: Tensor,
    ) -> Tensor:
        return self.embedding_model.weight + input

    def __resolve_positions(
        self,
        input: Tensor,
        incremental_state: Dict[str, Dict[str, Tensor | None]] | None,
        positions: Tensor | None,
    ) -> Tensor:
        if positions is not None:
            return positions
        if incremental_state is not None:
            fill_value = int(self.padding_idx + input.size(1))
            positions = torch.zeros((1, 1), device=input.device, dtype=input.dtype)
            return positions.fill_(fill_value)
        return self.__make_positions(input)

    def __make_positions(self, tensor: Tensor):
        mask = tensor.ne(self.padding_idx).int()
        mask = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return mask.long() + self.padding_idx
