from torch import Tensor
from emperor.embedding.relative.factory import RelativePositionalEmbeddingFactory

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.config import MultiHeadAttentionConfig
    from emperor.attention.core.handlers.projector import ProjectorBase
    from emperor.attention.core.handlers.reshaper import ReshaperBase


class ProcessorBase:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        projector: "ProjectorBase",
        reshaper: "ReshaperBase",
    ):
        self.cfg = cfg
        self.projector = projector
        self.reshaper = reshaper
        self.num_heads: int = self.cfg.num_heads
        self.batch_size: int = self.cfg.batch_size
        self.embedding_dim: int = self.cfg.embedding_dim
        self.dropout_probability: float = self.cfg.dropout_probability
        self.target_sequence_length: int = self.cfg.target_sequence_length
        self.source_sequence_length: int = self.cfg.source_sequence_length
        self.query_key_projection_dim: int = self.cfg.query_key_projection_dim
        if self.query_key_projection_dim == 0:
            self.query_key_projection_dim = self.embedding_dim
        self.value_projection_dim: int = self.cfg.value_projection_dim
        if self.value_projection_dim == 0:
            self.value_projection_dim = self.embedding_dim

        self.causal_attention_mask_flag: bool = self.cfg.causal_attention_mask_flag
        self.average_attention_weights_flag: bool = (
            self.cfg.average_attention_weights_flag
        )
        self.zero_attention_flag: bool = self.cfg.zero_attention_flag
        self.return_attention_weights_flag: bool = (
            self.cfg.return_attention_weights_flag
        )
        self.add_key_value_bias_flag: bool = self.cfg.add_key_value_bias_flag
        self.head_dim: int = self.embedding_dim // self.num_heads
        self.qk_head_dim, self.v_head_dim = self.__resolve_qkv_head_dim()
        self.relative_positional_embedding = (
            self.__maybe_initialize_relative_positional_embedding()
        )

    def __maybe_initialize_relative_positional_embedding(self):
        if self.cfg.relative_positional_embedding_config is not None:
            return RelativePositionalEmbeddingFactory(
                self.cfg.relative_positional_embedding_config
            ).build()
        return None

    def __resolve_qkv_head_dim(self) -> tuple[int, int]:
        qk_head_dim = (
            self.query_key_projection_dim // self.num_heads
            if self.query_key_projection_dim != 0
            else self.head_dim
        )
        v_head_dim = (
            self.value_projection_dim // self.num_heads
            if self.value_projection_dim != 0
            else self.head_dim
        )
        return qk_head_dim, v_head_dim

    def _is_single_batch(self, attention_output: Tensor) -> bool:
        return attention_output.size(0) == 1

    def _compute_attention_output(self, weighted_values: Tensor) -> Tensor:
        attention_output = self.projector.compute_output_projection(weighted_values)
        embedding_dim = attention_output.size(-1)
        target_sequence_length = attention_output.size(0) // self.batch_size
        return attention_output.view(
            target_sequence_length, self.batch_size, embedding_dim
        )

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError("compute_attention must be implemented by subclass.")
