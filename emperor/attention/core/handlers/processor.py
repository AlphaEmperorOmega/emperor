from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor

from emperor.attention.core._validator import AttentionValidatorBase
from emperor.base.module import Module

if TYPE_CHECKING:
    from emperor.attention.core.config import MultiHeadAttentionConfig
    from emperor.attention.core.handlers.projector import ProjectorBase
    from emperor.attention.core.handlers.reshaper import ReshaperBase
    from emperor.attention.core.runtime import QKV, AttentionRuntimeShape


class ProcessorBase(Module):
    VALIDATOR = AttentionValidatorBase

    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        projector: "ProjectorBase",
        reshaper: "ReshaperBase",
    ):
        super().__init__()
        self.cfg = cfg
        object.__setattr__(self, "projector", projector)
        self.reshaper = reshaper
        self.num_heads: int = self.cfg.num_heads
        self.batch_size: int = self.cfg.batch_size
        self.embedding_dim: int = self.cfg.embedding_dim
        self.dropout_probability: float = self.cfg.dropout_probability
        self.target_sequence_length: int = self.cfg.target_sequence_length
        self.query_key_projection_dim: int = (
            self.cfg.query_key_projection_dim or self.embedding_dim
        )
        self.value_projection_dim: int = (
            self.cfg.value_projection_dim or self.embedding_dim
        )
        self.average_attention_weights_flag: bool = (
            self.cfg.average_attention_weights_flag
        )
        self.return_attention_weights_flag: bool = (
            self.cfg.return_attention_weights_flag
        )
        self.qk_head_dim, self.v_head_dim = self.__resolve_qkv_head_dim()
        self.relative_positional_embedding = (
            self.__maybe_initialize_relative_positional_embedding()
        )

    def __maybe_initialize_relative_positional_embedding(self):
        if self.cfg.relative_positional_embedding_config is not None:
            return self.cfg.relative_positional_embedding_config.build()
        return None

    def __resolve_qkv_head_dim(self) -> tuple[int, int]:
        return (
            self.query_key_projection_dim // self.num_heads,
            self.value_projection_dim // self.num_heads,
        )

    def _is_single_batch(self, attention_output: Tensor) -> bool:
        return attention_output.size(0) == 1

    def _compute_relative_position_logits(
        self,
        query: Tensor,
        source_sequence_length: int,
        runtime_shape: "AttentionRuntimeShape | None" = None,
        *,
        query_is_scaled: bool = False,
    ) -> Tensor | None:
        if self.relative_positional_embedding is None:
            return None
        real_source_sequence_length = (
            runtime_shape.real_source_sequence_length
            if runtime_shape is not None
            else source_sequence_length
        )
        scaled_query = query if query_is_scaled else query * query.size(-1) ** -0.5
        prepared_query, restore_shape = self.__prepare_relative_position_query(
            scaled_query, real_source_sequence_length, runtime_shape
        )
        logits = self.relative_positional_embedding(
            prepared_query,
            sequence_length=real_source_sequence_length,
        )
        return self.__restore_and_pad_relative_position_logits(
            logits, restore_shape, source_sequence_length, real_source_sequence_length
        )

    def __prepare_relative_position_query(
        self,
        query: Tensor,
        real_source_sequence_length: int,
        runtime_shape: "AttentionRuntimeShape | None",
    ) -> tuple[Tensor, tuple[int, ...] | None]:
        self.VALIDATOR.validate_standard_relative_position_query_shape(
            query, self.num_heads
        )
        if query.dim() == 3:
            return self.__reshape_flattened_head_query(
                query, real_source_sequence_length, runtime_shape
            )
        return query, None

    def __reshape_flattened_head_query(
        self,
        query: Tensor,
        real_source_sequence_length: int,
        runtime_shape: "AttentionRuntimeShape | None",
    ) -> tuple[Tensor, tuple[int, ...]]:
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        target_sequence_length = query.size(-2)
        branch_count = query.size(0)
        expected_branch_count = batch_size * self.num_heads
        self.VALIDATOR.validate_relative_position_query_branch_count(
            branch_count, expected_branch_count
        )
        query_head_dimension = query.size(-1)
        prepared_query_shape = (
            batch_size,
            self.num_heads,
            target_sequence_length,
            query_head_dimension,
        )
        prepared_query = query.contiguous().view(prepared_query_shape)
        restore_shape = (
            branch_count,
            target_sequence_length,
            real_source_sequence_length,
        )
        return prepared_query, restore_shape

    @staticmethod
    def __restore_and_pad_relative_position_logits(
        logits: Tensor,
        restore_shape: tuple[int, ...] | None,
        source_sequence_length: int,
        real_source_sequence_length: int,
    ) -> Tensor:
        if restore_shape is not None:
            logits = logits.contiguous().view(restore_shape)
        synthetic_position_count = source_sequence_length - real_source_sequence_length
        synthetic_position_padding = (0, synthetic_position_count)
        return F.pad(logits, synthetic_position_padding)

    def _compute_attention_output(
        self,
        weighted_values: Tensor,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        attention_output = self.projector.compute_output_projection(weighted_values)
        embedding_dim = self.embedding_dim
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        target_sequence_length = (
            runtime_shape.target_sequence_length
            if runtime_shape is not None
            else attention_output.size(0) // batch_size
        )
        return attention_output.view(target_sequence_length, batch_size, embedding_dim)

    def compute_attention(
        self,
        qkv: "QKV",
        merged_attention_mask: Tensor | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError("compute_attention must be implemented by subclass.")
