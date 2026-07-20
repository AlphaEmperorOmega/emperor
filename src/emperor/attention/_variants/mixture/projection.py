"""Private mixture-of-attention-heads projection implementation."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from emperor.attention._ops.projection import ProjectorBase
from emperor.experts import (
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from emperor.experts._config import (
    _MixtureOfExpertsMapConfig,
    _MixtureOfExpertsReduceConfig,
)
from emperor.experts._layers.mixture import MixtureOfExperts

if TYPE_CHECKING:
    from emperor.attention._runtime import QKV
    from emperor.attention._variants.mixture.config import (
        MixtureOfAttentionHeadsConfig,
    )
    from emperor.linears import LinearAbstract


class MixtureOfAttentionHeadsProjector(ProjectorBase):
    def __init__(self, cfg: "MixtureOfAttentionHeadsConfig"):
        super().__init__(cfg)
        self.experts_config: MixtureOfExpertsConfig = self.cfg.experts_config
        self.use_kv_expert_models_flag: bool = self.cfg.use_kv_expert_models_flag

        qk_dims = (self.embedding_dim, self.query_key_projection_dim)
        v_dims = (self.embedding_dim, self.value_projection_dim)

        self.query_model = self.__create_q_model(*qk_dims)
        self.key_model = self.__create_kv_model(*qk_dims)
        self.value_model = self.__create_kv_model(*v_dims)
        self.top_k = self.query_model.get_top_k()
        self.sampler = self.__create_sampler()

        self.probabilities = None
        self.indices = None
        self.skip_mask = None

    def clear_routing_state(self) -> None:
        self.probabilities = None
        self.indices = None
        self.skip_mask = None

    def __create_sampler(self):
        sampler_config = self.experts_config.sampler_config
        return sampler_config.build_with_router_input_dim(self.embedding_dim)

    def __create_q_model(self, input_dim: int, output_dim: int) -> MixtureOfExperts:
        overrides = MixtureOfExpertsConfig(input_dim=input_dim, output_dim=output_dim)
        return _MixtureOfExpertsMapConfig(self.experts_config).build(overrides)

    def __create_kv_model(self, input_dim: int, output_dim: int):
        if self.use_kv_expert_models_flag:
            overrides = MixtureOfExpertsConfig(
                input_dim=input_dim, output_dim=output_dim
            )
            return _MixtureOfExpertsMapConfig(self.experts_config).build(overrides)
        return self._create_model(input_dim, output_dim)

    def _build_output_model(self) -> MixtureOfExperts:
        overrides = MixtureOfExpertsConfig(
            input_dim=self.value_projection_dim,
            output_dim=self.embedding_dim,
            routing_initialization_mode=RoutingInitializationMode.DISABLED,
        )
        return _MixtureOfExpertsReduceConfig(self.cfg.experts_config).build(overrides)

    def compute_qkv_projections(
        self,
        qkv: "QKV",
    ) -> "QKV":
        self.__compute_expert_indices(qkv.query)
        q_projection = self.__compute_q_projection(qkv.query, self.query_model)
        k_projection = self.__compute_kv_projection(qkv.key, self.key_model)
        v_projection = self.__compute_kv_projection(qkv.value, self.value_model)
        return replace(
            qkv,
            query=q_projection,
            key=k_projection,
            value=v_projection,
        )

    def __compute_expert_indices(self, X: Tensor) -> None:
        embedding_dim = X.size(-1)
        if not X.is_contiguous():
            X = X.contiguous()
        X_reshaped = X.view(-1, embedding_dim)
        (
            probabilities,
            indices,
            skip_mask,
            sampler_loss,
        ) = self.sampler.sample_probabilities_and_indices(X_reshaped, self.skip_mask)

        self.probabilities = probabilities.view(-1, self.top_k)
        self.indices = indices.view(-1, self.top_k)
        self.skip_mask = skip_mask
        self._accumulate_auxiliary_loss(sampler_loss)

    def __compute_q_projection(self, X: Tensor, model: MixtureOfExperts) -> Tensor:
        sequence_length, batch_size, _ = X.shape
        projection = self._compute_projection(X, model)
        return projection.view(sequence_length, batch_size, self.top_k, -1)

    def __compute_kv_projection(
        self,
        X: Tensor,
        model: "LinearAbstract | MixtureOfExperts",
    ) -> Tensor:
        sequence_length, batch_size, _ = X.shape
        projection = self._compute_projection(X, model)
        if self.use_kv_expert_models_flag:
            return projection.view(sequence_length, batch_size, self.top_k, -1)
        return projection.view(sequence_length, batch_size, -1)

    def _compute_projection(self, X: Tensor, model: nn.Module) -> Tensor:
        embedding_dim = X.size(-1)
        if not X.is_contiguous():
            X = X.contiguous()

        X_reshaped = X.view(-1, embedding_dim)
        if isinstance(model, MixtureOfExperts):
            projection, skip_mask, expert_loss = model(
                X_reshaped, self.probabilities, self.indices, self.skip_mask
            )
            self.skip_mask = skip_mask
            self._accumulate_auxiliary_loss(expert_loss)
        else:
            projection = self._forward_accumulating_loss(model, X_reshaped)

        return projection

    def compute_output_projection(self, weighted_values: Tensor) -> Tensor:
        output_projection = self._compute_projection(weighted_values, self.output_model)
        self.clear_routing_state()
        return output_projection
