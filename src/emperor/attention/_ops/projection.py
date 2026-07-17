"""Private attention projection operations."""

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from emperor.layers import Layer
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
    from emperor.attention._runtime import QKV
    from emperor.layers import LayerStackConfig, RecurrentLayerConfig


class ProjectorBase(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim: int = self.cfg.embedding_dim
        self.query_key_projection_dim: int = self.cfg.query_key_projection_dim
        self.value_projection_dim: int = self.cfg.value_projection_dim
        self.projection_model_config: LayerStackConfig | RecurrentLayerConfig = (
            self.cfg.projection_model_config
        )
        self.__resolve_kv_dimensions()
        self.output_model = self._build_output_model()
        self.auxiliary_loss: Tensor | None = None

    def __resolve_kv_dimensions(self) -> None:
        if not self.query_key_projection_dim:
            self.query_key_projection_dim = self.embedding_dim
        if not self.value_projection_dim:
            self.value_projection_dim = self.embedding_dim

    def _build_output_model(self) -> nn.Module:
        return self._create_model(self.value_projection_dim, self.embedding_dim)

    def _create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        overrides = type(self.projection_model_config)(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        return self.projection_model_config.build(overrides)

    def _compute_projection(self, tensor: Tensor, model: nn.Module) -> Tensor:
        sequence_length, batch_size, embedding_dim = tensor.shape
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensor_reshaped = tensor.view(-1, embedding_dim)
        projection = self._forward_accumulating_loss(model, tensor_reshaped)
        return projection.view(sequence_length, batch_size, -1)

    def _forward_accumulating_loss(self, model: nn.Module, tensor: Tensor) -> Tensor:
        state = Layer.run_model_returning_state(model, tensor)
        if state.loss is not None:
            self._accumulate_auxiliary_loss(state.loss)
        return state.hidden

    def compute_output_projection(self, weighted_values: Tensor) -> Tensor:
        uses_unflattened_sequence_batch_layout = weighted_values.dim() == 3
        if uses_unflattened_sequence_batch_layout:
            return self._compute_projection(weighted_values, self.output_model)
        return self._forward_accumulating_loss(self.output_model, weighted_values)

    def compute_qkv_projections(
        self,
        qkv: "QKV",
    ) -> "QKV":
        raise NotImplementedError(
            "compute_qkv_projections must be implemented by subclass."
        )

    def _accumulate_auxiliary_loss(self, loss: Tensor) -> None:
        if self.auxiliary_loss is None:
            self.auxiliary_loss = loss
        else:
            self.auxiliary_loss = self.auxiliary_loss + loss

    def get_auxiliary_loss_and_clear(self) -> "Tensor | None":
        accumulated_loss = self.auxiliary_loss
        self.auxiliary_loss = None
        return accumulated_loss
