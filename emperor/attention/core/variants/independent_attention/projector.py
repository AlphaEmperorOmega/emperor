from torch import Tensor
from emperor.attention.core.handlers.projector import ProjectorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.config import MultiHeadAttentionConfig


class IndependentProjector(ProjectorBase):
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        super().__init__(cfg)

        qk_dims = (self.embedding_dim, self.query_key_projection_dim)
        v_dims = (self.embedding_dim, self.value_projection_dim)
        self.query_model = self._create_model(*qk_dims)
        self.key_model = self._create_model(*qk_dims)
        self.value_model = self._create_model(*v_dims)

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_projections = self._compute_projection(query, self.query_model)
        key_projections = self._compute_projection(key, self.key_model)
        value_projections = self._compute_projection(value, self.value_model)
        return query_projections, key_projections, value_projections
