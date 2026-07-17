"""Private independent-attention projection implementation."""

from dataclasses import replace
from typing import TYPE_CHECKING

from emperor.attention._ops.projection import ProjectorBase

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
    from emperor.attention._runtime import QKV


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
        qkv: "QKV",
    ) -> "QKV":
        q_projection = self._compute_projection(qkv.query, self.query_model)
        k_projection = self._compute_projection(qkv.key, self.key_model)
        v_projection = self._compute_projection(qkv.value, self.value_model)
        return replace(qkv, query=q_projection, key=k_projection, value=v_projection)
