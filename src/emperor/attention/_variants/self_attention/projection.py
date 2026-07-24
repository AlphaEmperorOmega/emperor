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
    from emperor.attention._runtime import QKV, AttentionRuntimeLayout
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
        qkv: "QKV",
        *,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> "QKV":
        row_layout = runtime_layout.row_layout if runtime_layout is not None else None
        match self.projection_strategy:
            case SelfAttentionProjectionStrategy.FUSED:
                return self.__compute_fused_qkv_projections(
                    qkv,
                    row_layout=row_layout,
                )
            case SelfAttentionProjectionStrategy.FUSED_KEY_VALUE:
                return self.__compute_fused_key_value_projections(
                    qkv,
                    row_layout=row_layout,
                )
            case SelfAttentionProjectionStrategy.SEPARATE:
                return self.__compute_separate_qkv_projections(
                    qkv,
                    row_layout=row_layout,
                )
        raise AssertionError("projection_strategy was validated during construction.")

    def __compute_fused_qkv_projections(
        self,
        qkv: "QKV",
        *,
        row_layout: "RowLayout | None",
    ) -> "QKV":
        qkv_projection = self._compute_projection(
            qkv.query,
            self.qkv_model,
            row_layout=row_layout,
        )
        q_projection, k_projection, v_projection = (
            self.__split_self_attention_projection(qkv_projection)
        )
        return replace(qkv, query=q_projection, key=k_projection, value=v_projection)

    def __compute_fused_key_value_projections(
        self,
        qkv: "QKV",
        *,
        row_layout: "RowLayout | None",
    ) -> "QKV":
        q_projection = self._compute_projection(
            qkv.query,
            self.query_model,
            row_layout=row_layout,
        )
        key_value_projection = self._compute_projection(
            qkv.key,
            self.key_value_model,
            row_layout=row_layout,
        )
        k_projection, v_projection = self.__split_key_value_projection(
            key_value_projection
        )
        return replace(qkv, query=q_projection, key=k_projection, value=v_projection)

    def __compute_separate_qkv_projections(
        self,
        qkv: "QKV",
        *,
        row_layout: "RowLayout | None",
    ) -> "QKV":
        q_projection = self._compute_projection(
            qkv.query,
            self.query_model,
            row_layout=row_layout,
        )
        k_projection = self._compute_projection(
            qkv.key,
            self.key_model,
            row_layout=row_layout,
        )
        v_projection = self._compute_projection(
            qkv.value,
            self.value_model,
            row_layout=row_layout,
        )
        return replace(qkv, query=q_projection, key=k_projection, value=v_projection)

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
