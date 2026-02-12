import torch.nn as nn

from torch.types import Tensor
from Emperor.base.utils import Module
from torch.nn.modules import Sequential
from Emperor.sampler.model import SamplerModel
from Emperor.linears.utils.layers import LinearBase
from Emperor.base.layer import Layer, LayerStackConfig
from Emperor.experts.utils.enums import InitSamplerOptions
from Emperor.sampler.utils.routers import RouterConfig, RouterModel
from Emperor.experts.utils.layers import (
    MixtureOfExperts,
    MixtureOfExpertsConfig,
    MixtureOfExpertsMap,
    MixtureOfExpertsReduce,
)
from Emperor.attention.utils.handlers.validators._projector import (
    IndependentProjectorValidator,
    MixtureOfAttentionHeadsProjectorValidator,
    SelfAttentionProjectorValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.layer import MultiHeadAttentionConfig


class ProjectorBuilder(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
    ):
        super().__init__()
        self.cfg = cfg

        self.embedding_dim = self.cfg.embedding_dim
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.attention_option = self.cfg.attention_option

    def build(self) -> "ProjectorBase":
        from Emperor.attention.utils.enums import AttentionOptions

        match self.attention_option:
            case AttentionOptions.SELF_ATTENTION:
                self.__should_use_self_attention_projector()
                return SelfAttentionProjector(self.cfg)
            case AttentionOptions.INDEPENDENT:
                return IndependentProjector(self.cfg)
            case AttentionOptions.MIXTURE_OF_ATTENTION_HEADS:
                return MixtureOfAttentionHeadsProjector(self.cfg)
            case _:
                raise ValueError(
                    f"Attention option not supported or unknown option given: {self.attention_option}"
                )

    def __should_use_self_attention_projector(self):
        are_qk_dims_same = self.embedding_dim == self.query_key_projection_dim
        are_qv_dims_same = self.embedding_dim == self.value_projection_dim
        return are_qk_dims_same and are_qv_dims_same


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
        self.embedding_dim: int = self.cfg.embedding_dim
        self.query_key_projection_dim: int = self.cfg.query_key_projection_dim
        if self.query_key_projection_dim == 0:
            self.query_key_projection_dim = self.embedding_dim
        self.value_projection_dim: int = self.cfg.value_projection_dim
        if self.value_projection_dim == 0:
            self.value_projection_dim = self.embedding_dim
        self.return_attention_weights_flag: bool = (
            self.cfg.return_attention_weights_flag
        )
        self.attention_option: "AttentionOptions" = self.cfg.attention_option
        self.experts_config: "MixtureOfExpertsConfig" = self.cfg.experts_config
        self.use_kv_expert_models_flag: bool = self.cfg.use_kv_expert_models_flag
        self.__resolve_kv_dimensions()
        self.output_model = self._build_output_model()

    def _build_output_model(self) -> Layer:
        return self._create_model(self.value_projection_dim, self.embedding_dim)

    def __resolve_kv_dimensions(self) -> None:
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
        if weighted_values.dim() == 3:
            return self._compute_projection(weighted_values, self.output_model)
        return self.output_model(weighted_values)

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError(
            "`compute_qkv_projections` method must be implemented by subclass"
        )


class SelfAttentionProjector(ProjectorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.qkv_model = self._create_model(self.embedding_dim, self.embedding_dim * 3)
        self.validator = SelfAttentionProjectorValidator(self)

    def _build_output_model(self) -> Layer | Sequential:
        return self._create_model(self.embedding_dim, self.embedding_dim)

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.ensure_qkv_are_same_tensor_for_self_attention(key, query, value)
        qkv_projection = self._compute_projection(query, self.qkv_model)
        query_projections, key_projections, value_projections = (
            self.__split_self_attention_projection(qkv_projection)
        )
        return query_projections, key_projections, value_projections

    def __split_self_attention_projection(
        self, qkv_projections: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        X = qkv_projections.unflatten(-1, (3, -1))
        X = X.unsqueeze(0).transpose(0, -2)
        X = X.squeeze(-2).contiguous()
        query_projections = X[0]
        key_projections = X[1]
        value_projections = X[2]

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


class MixtureOfAttentionHeadsProjector(ProjectorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.router_config = self.experts_config.router_model_config
        self.sampler_config = self.experts_config.sampler_model_config

        qk_dims = (self.embedding_dim, self.query_key_projection_dim)
        v_dims = (self.embedding_dim, self.value_projection_dim)

        self.query_model = self._create_q_model(*qk_dims)
        self.key_model = self._create_kv_model(*qk_dims)
        self.value_model = self._create_kv_model(*v_dims)
        self.top_k = self.query_model.get_top_k()
        self.router, self.sampler = self.__create_router_and_sampler()
        self.validator = MixtureOfAttentionHeadsProjectorValidator(self)

        self.probabilities = None
        self.indices = None
        self.skip_mask = None
        self.sampler_loss = None

    def __create_router_and_sampler(
        self,
    ) -> tuple["RouterModel", "SamplerModel"]:
        router_overrides = RouterConfig(input_dim=self.embedding_dim)
        router = RouterModel(self.experts_config.router_model_config, router_overrides)
        sampler = SamplerModel(self.experts_config.sampler_model_config)
        return router, sampler

    def _create_q_model(self, input_dim: int, output_dim: int):
        overrides = MixtureOfExpertsConfig(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        return MixtureOfExpertsMap(self.experts_config, overrides)

    def _create_kv_model(self, input_dim: int, output_dim: int):
        if self.use_kv_expert_models_flag:
            overrides = MixtureOfExpertsConfig(
                input_dim=input_dim,
                output_dim=output_dim,
            )
            return self._create_experts_model(overrides)
        return self._create_model(input_dim, output_dim)

    def _create_experts_model(
        self,
        overrides: MixtureOfExpertsConfig,
    ) -> MixtureOfExperts:
        return MixtureOfExpertsMap(self.experts_config, overrides)

    def _build_output_model(self) -> MixtureOfExperts:
        overrides = MixtureOfExpertsConfig(
            input_dim=self.value_projection_dim,
            output_dim=self.embedding_dim,
            init_sampler_option=InitSamplerOptions.DISABLED,
        )
        return MixtureOfExpertsReduce(self.experts_config, overrides)

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.ensure_attention_weights_returned_for_self_attention_only()
        self.validator.ensure_propper_kv_shapes_for_independent_projector(key, value)

        self._compute_expert_indices(query)
        query_projections = self._compute_q_projection(query, self.query_model)
        key_projections = self._compute_kv_projection(key, self.key_model)
        value_projections = self._compute_kv_projection(value, self.value_model)

        return query_projections, key_projections, value_projections

    def _compute_expert_indices(self, X: Tensor) -> None:
        logits = self.router.compute_logit_scores(X)
        # TODO: Ensure the skip mask gets used in the future
        skip_mask = None
        (
            probabilities,
            indices,
            skip_mask,
            sampler_loss,
        ) = self.sampler.sample_probabilities_and_indices(logits, skip_mask)

        self.probabilities = probabilities.view(-1, self.top_k)
        self.indices = indices.view(-1, self.top_k)
        self.skip_mask = skip_mask
        self.sampler_loss = sampler_loss

    def _compute_q_projection(self, X: Tensor, model: MixtureOfExperts) -> Tensor:
        sequence_length, batch_size, _ = X.shape
        projection = self._compute_projection(X, model)
        return projection.view(sequence_length, batch_size, self.top_k, -1)

    def _compute_kv_projection(
        self,
        X: Tensor,
        model: LinearBase | MixtureOfExperts,
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
            projection = model(X_reshaped, self.probabilities, self.indices)
        else:
            projection = model(X_reshaped)

        # TODO: In the future return those to make use of them on higher
        # level classes
        if isinstance(projection, tuple):
            if len(projection) == 2:
                projection, loss = projection
            elif len(projection) == 3:
                projection, skip_mask, loss = projection

        return projection

    def compute_output_projection(self, weighted_values: Tensor) -> Tensor:
        sequence_length, batch_size, _, _ = weighted_values.shape
        projection = self._compute_projection(weighted_values, self.output_model)
        return projection.view(sequence_length, batch_size, self.embedding_dim)
