import torch.nn as nn

from torch import Tensor
from emperor.attention.core.handlers.projector import ProjectorBase
from emperor.experts.core.options import RoutingInitializationMode
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core.layers import (
    MixtureOfExperts,
    MixtureOfExpertsMap,
    MixtureOfExpertsReduce,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.mixture_of_attention_heads.config import (
        MixtureOfAttentionHeadsConfig,
    )
    from emperor.linears.core.layers import LinearAbstract


class MixtureOfAttentionHeadsProjector(ProjectorBase):
    def __init__(self, cfg: "MixtureOfAttentionHeadsConfig"):
        super().__init__(cfg)
        self.experts_config: "MixtureOfExpertsConfig" = self.cfg.experts_config
        self.use_kv_expert_models_flag: bool = self.cfg.use_kv_expert_models_flag

        qk_dims = (self.embedding_dim, self.query_key_projection_dim)
        v_dims = (self.embedding_dim, self.value_projection_dim)

        self.query_model = self._create_q_model(*qk_dims)
        self.key_model = self._create_kv_model(*qk_dims)
        self.value_model = self._create_kv_model(*v_dims)
        self.top_k = self.query_model.get_top_k()
        self.sampler = self.__create_sampler()

        self.probabilities = None
        self.indices = None
        self.skip_mask = None

    def __create_sampler(self):
        from emperor.sampler.core.config import RouterConfig, SamplerConfig

        sampler_config = self.experts_config.sampler_config
        router_overrides = RouterConfig(input_dim=self.embedding_dim)
        router_config = self._override_config(
            sampler_config.router_config, router_overrides
        )
        sampler_overrides = SamplerConfig(router_config=router_config)
        return sampler_config.build(sampler_overrides)

    def _create_q_model(self, input_dim: int, output_dim: int) -> MixtureOfExperts:
        overrides = MixtureOfExpertsConfig(input_dim=input_dim, output_dim=output_dim)
        return MixtureOfExpertsMap(self.experts_config, overrides)

    def _create_kv_model(self, input_dim: int, output_dim: int):
        if self.use_kv_expert_models_flag:
            overrides = MixtureOfExpertsConfig(
                input_dim=input_dim, output_dim=output_dim
            )
            return MixtureOfExpertsMap(self.experts_config, overrides)
        return self._create_model(input_dim, output_dim)

    def _build_output_model(self) -> MixtureOfExperts:
        overrides = MixtureOfExpertsConfig(
            input_dim=self.value_projection_dim,
            output_dim=self.embedding_dim,
            routing_initialization_mode=RoutingInitializationMode.DISABLED,
        )
        return MixtureOfExpertsReduce(self.cfg.experts_config, overrides)

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._compute_expert_indices(query)
        query_projections = self._compute_q_projection(query, self.query_model)
        key_projections = self._compute_kv_projection(key, self.key_model)
        value_projections = self._compute_kv_projection(value, self.value_model)
        return query_projections, key_projections, value_projections

    def _compute_expert_indices(self, X: Tensor) -> None:
        embedding_dim = X.size(-1)
        if not X.is_contiguous():
            X = X.contiguous()
        X_reshaped = X.view(-1, embedding_dim)
        skip_mask = None
        (
            probabilities,
            indices,
            skip_mask,
            sampler_loss,
        ) = self.sampler.sample_probabilities_and_indices(X_reshaped, skip_mask)

        self.probabilities = probabilities.view(-1, self.top_k)
        self.indices = indices.view(-1, self.top_k)
        self.skip_mask = skip_mask
        self._accumulate_auxiliary_loss(sampler_loss)

    def _compute_q_projection(self, X: Tensor, model: MixtureOfExperts) -> Tensor:
        sequence_length, batch_size, _ = X.shape
        projection = self._compute_projection(X, model)
        return projection.view(sequence_length, batch_size, self.top_k, -1)

    def _compute_kv_projection(
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
            projection, expert_loss = model(X_reshaped, self.probabilities, self.indices)
            self._accumulate_auxiliary_loss(expert_loss)
        else:
            projection = self._forward_accumulating_loss(model, X_reshaped)

        return projection

    def compute_output_projection(self, weighted_values: Tensor) -> Tensor:
        return self._compute_projection(weighted_values, self.output_model)
