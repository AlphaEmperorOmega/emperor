from torch import Tensor
from emperor.attention.core.handlers.projector import ProjectorBase
from emperor.base.layer import Layer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.config import MultiHeadAttentionConfig


class SelfAttentionProjector(ProjectorBase):
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        super().__init__(cfg)
        self.qkv_model = self._create_model(self.embedding_dim, self.embedding_dim * 3)

    def _build_output_model(self) -> Layer:
        return self._create_model(self.embedding_dim, self.embedding_dim)

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        qkv_projection = self._compute_projection(query, self.qkv_model)
        return self.__split_self_attention_projection(qkv_projection)

    def __split_self_attention_projection(
        self, qkv_projections: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        X = qkv_projections.unflatten(-1, (3, -1))
        X = X.unsqueeze(0).transpose(0, -2)
        X = X.squeeze(-2).contiguous()
        return X[0], X[1], X[2]
