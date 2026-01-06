import torch.nn as nn

from torch.types import Tensor
from Emperor.base.utils import Module
from Emperor.base.layer import Layer, LayerStackConfig
from Emperor.attention.utils.handlers._validator import (
    IndependentProjectorValidator,
    SelfAttentionProjectorValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.layer import MultiHeadAttentionConfig


class ProjectorSelector(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
    ):
        super().__init__()
        self.cfg = cfg

        self.embedding_dim = self.cfg.embedding_dim
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.is_self_attention_projector_flag = (
            self.cfg.is_self_attention_projector_flag
        )

    def build_model(self):
        if self.__should_use_self_attention_projector():
            return SelfAttentionProjector(self.cfg)
        return IndependentProjector(self.cfg)

    def __should_use_self_attention_projector(self):
        is_self_attention = self.is_self_attention_projector_flag
        are_qk_dims_same = self.embedding_dim == self.query_key_projection_dim
        are_qv_dims_same = self.embedding_dim == self.value_projection_dim
        return is_self_attention and (are_qk_dims_same and are_qv_dims_same)


class ProjectorBase(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "MultiHeadAttentionConfig" = self._overwrite_config(cfg, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)

        self.model_type = self.cfg.model_type.value
        self.embedding_dim = self.cfg.embedding_dim
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag
        self.is_self_attention_projector_flag = (
            self.cfg.is_self_attention_projector_flag
        )
        self.__resolve_kv_dimensions()
        self.output_model = self._build_output_model()

    def _build_output_model(self) -> tuple:
        return self._create_model(self.value_projection_dim, self.embedding_dim)

    def __resolve_kv_dimensions(self):
        is_qk_dim_zero = self.query_key_projection_dim == 0
        is_v_dim_zero = self.value_projection_dim == 0
        self.query_key_projection_dim = (
            self.embedding_dim if is_qk_dim_zero else self.query_key_projection_dim
        )
        self.value_projection_dim = (
            self.embedding_dim if is_v_dim_zero else self.value_projection_dim
        )

    def _create_model(self, input_dim: int, output_dim: int) -> Layer:
        overrides = LayerStackConfig(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        return self.model_type(self.main_cfg, overrides).build_model()

    def _compute_projection(self, tensor: Tensor, model: nn.Module) -> Tensor:
        sequence_length, batch_size, embedding_dim = tensor.shape
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensor_reshaped = tensor.view(-1, embedding_dim)
        projection = model(tensor_reshaped)
        if isinstance(projection, tuple):
            if len(projection) == 2:
                projection, loss = projection
            elif len(projection) == 3:
                projection, skip_mask, loss = projection

        return projection.view(sequence_length, batch_size, -1)

    def compute_output_projection(self, weighted_values: Tensor) -> Tensor:
        return self._compute_projection(weighted_values, self.output_model)


class SelfAttentionProjector(ProjectorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.qkv_model = self._create_model(self.embedding_dim, self.embedding_dim * 3)
        self.validator = SelfAttentionProjectorValidator(self)

    def _build_output_model(self) -> tuple:
        return self._create_model(self.embedding_dim, self.embedding_dim)

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.ensure_qkv_are_equal_for_self_attention(key, query, value)
        qkv_projection = self._compute_projection(query, self.qkv_model)
        query_projections, key_projections, value_projections = (
            self.__split_self_attention_projection(qkv_projection)
        )
        return query_projections, key_projections, value_projections

    def __split_self_attention_projection(
        self, qkv_projections: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        projections = qkv_projections.unflatten(-1, (3, -1))
        projections = projections.unsqueeze(0)
        projections = projections.transpose(0, -2)
        projections = projections.squeeze(-2)
        projections = projections.contiguous()
        query_projections = projections[0]
        key_projections = projections[1]
        value_projections = projections[2]

        return query_projections, key_projections, value_projections


class IndependentProjector(ProjectorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        qk_dims = (self.embedding_dim, self.query_key_projection_dim)
        v_dims = (self.embedding_dim, self.value_projection_dim)
        self.query_model = self._create_model(*qk_dims)
        self.key_model = self._create_model(*qk_dims)
        self.value_model = self._create_model(*v_dims)

        self.validator = IndependentProjectorValidator(self)

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.ensure_attention_weights_returned_for_self_attention_only()
        self.validator.ensure_propper_kv_shapes_for_independent_projector(key, value)
        query_projections = self._compute_projection(query, self.query_model)
        key_projections = self._compute_projection(key, self.key_model)
        value_projections = self._compute_projection(value, self.value_model)

        return query_projections, key_projections, value_projections
