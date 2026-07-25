"""Private self-attention projection implementation."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from emperor.attention._ops.projection import ProjectorBase
from emperor.attention._variants.self_attention.config import (
    SelfAttentionProjectionStrategy,
)

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
    from emperor.attention._runtime import (
        QKV,
        AttentionRuntimeLayout,
        MultiHeadAttentionInputs,
    )
    from emperor.layers import RowLayout


class SelfAttentionProjector(ProjectorBase):
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        super().__init__(cfg)
        self.projection_strategy: SelfAttentionProjectionStrategy = (
            self.cfg.projection_strategy
        )
        self.qkv_model: nn.Module | None = None
        self.query_model: nn.Module | None = None
        self.key_value_model: nn.Module | None = None
        self.key_model: nn.Module | None = None
        self.value_model: nn.Module | None = None
        self.__build_projection_models()

    def _build_output_model(self) -> nn.Module:
        return self._create_model(self.embedding_dim, self.embedding_dim)

    def __build_projection_models(self) -> None:
        match self.projection_strategy:
            case SelfAttentionProjectionStrategy.FUSED:
                self.__build_fused_qkv_projection_model()
            case SelfAttentionProjectionStrategy.FUSED_KEY_VALUE:
                self.__build_query_and_fused_key_value_projection_models()
            case SelfAttentionProjectionStrategy.SEPARATE:
                self.__build_separate_qkv_projection_models()
            case _:
                raise ValueError(
                    "projection_strategy must be FUSED, FUSED_KEY_VALUE, or "
                    "SEPARATE for "
                    f"SelfAttentionProjector, got {self.projection_strategy!r}."
                )

    def __build_fused_qkv_projection_model(self) -> None:
        self.qkv_model = self._create_model(self.embedding_dim, self.embedding_dim * 3)

    def __build_query_and_fused_key_value_projection_models(self) -> None:
        self.query_model = self._create_model(self.embedding_dim, self.embedding_dim)
        self.key_value_model = self._create_model(
            self.embedding_dim, self.embedding_dim * 2
        )

    def __build_separate_qkv_projection_models(self) -> None:
        self.query_model = self._create_model(self.embedding_dim, self.embedding_dim)
        self.key_model = self._create_model(self.embedding_dim, self.embedding_dim)
        self.value_model = self._create_model(self.embedding_dim, self.embedding_dim)

    def compute_qkv_projections(
        self,
        attention_inputs: "MultiHeadAttentionInputs | QKV",
        *,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> "MultiHeadAttentionInputs | QKV":
        runtime_layout = getattr(attention_inputs, "runtime_layout", runtime_layout)
        row_layout = runtime_layout.row_layout if runtime_layout is not None else None
        match self.projection_strategy:
            case SelfAttentionProjectionStrategy.FUSED:
                query, key, value = self.__compute_fused_qkv_projections(
                    attention_inputs.query,
                    row_layout=row_layout,
                )
            case SelfAttentionProjectionStrategy.FUSED_KEY_VALUE:
                query, key, value = self.__compute_fused_key_value_projections(
                    attention_inputs.query,
                    attention_inputs.key,
                    row_layout=row_layout,
                )
            case SelfAttentionProjectionStrategy.SEPARATE:
                query, key, value = self.__compute_separate_qkv_projections(
                    attention_inputs.query,
                    attention_inputs.key,
                    attention_inputs.value,
                    row_layout=row_layout,
                )
            case _:
                raise AssertionError(
                    "projection_strategy was validated during construction."
                )
        return replace(attention_inputs, query=query, key=key, value=value)

    def __compute_fused_qkv_projections(
        self,
        query: Tensor,
        *,
        row_layout: "RowLayout | None",
    ) -> tuple[Tensor, Tensor, Tensor]:
        qkv_projection = self._compute_projection(
            query,
            self.qkv_model,
            row_layout=row_layout,
        )
        return self.__split_self_attention_projection(qkv_projection)

    def __compute_fused_key_value_projections(
        self,
        query: Tensor,
        key: Tensor,
        *,
        row_layout: "RowLayout | None",
    ) -> tuple[Tensor, Tensor, Tensor]:
        q_projection = self._compute_projection(
            query,
            self.query_model,
            row_layout=row_layout,
        )
        key_value_projection = self._compute_projection(
            key,
            self.key_value_model,
            row_layout=row_layout,
        )
        k_projection, v_projection = self.__split_key_value_projection(
            key_value_projection
        )
        return q_projection, k_projection, v_projection

    def __compute_separate_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        row_layout: "RowLayout | None",
    ) -> tuple[Tensor, Tensor, Tensor]:
        q_projection = self._compute_projection(
            query,
            self.query_model,
            row_layout=row_layout,
        )
        k_projection = self._compute_projection(
            key,
            self.key_model,
            row_layout=row_layout,
        )
        v_projection = self._compute_projection(
            value,
            self.value_model,
            row_layout=row_layout,
        )
        return q_projection, k_projection, v_projection

    def __split_self_attention_projection(
        self, qkv_projections: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        query, key, value = qkv_projections.chunk(3, dim=-1)
        return query, key, value

    def __split_key_value_projection(
        self, key_value_projection: Tensor
    ) -> tuple[Tensor, Tensor]:
        key, value = key_value_projection.chunk(2, dim=-1)
        return key, value
