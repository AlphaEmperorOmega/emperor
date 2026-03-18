import torch
import torch.nn as nn

from torch import Tensor, device
from dataclasses import dataclass, field
from emperor.sampler.model import SamplerModel
from emperor.base.layer import LayerStackConfig
from emperor.base.utils import ConfigBase, Module
from emperor.experts.utils._expert_capacity import _ExpertCapacityHandler
from emperor.experts.utils._expert_weighting import _ExpertWeightingHandler
from emperor.experts.utils._validator import _ValidatorHandler
from emperor.experts.utils.enums import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    InitSamplerOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.sampler.utils.samplers import SamplerConfig
    from emperor.linears.options import LinearLayerStackOptions
    from emperor.sampler.utils.routers import RouterConfig, RouterModel


@dataclass
class MixtureOfExpertsConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_stack_option: "LinearLayerStackOptions | None" = field(
        default=None,
        metadata={"help": "Number of layers added to the router"},
    )
    top_k: int | None = field(
        default=None,
        metadata={
            "help": "Top-k probabilities and indices to be selected from a distribution"
        },
    )
    num_experts: int | None = field(
        default=None,
        metadata={"help": "Number of experts in the model"},
    )
    capacity_factor: float | None = field(
        default=None,
        metadata={
            "help": "Limits tokens per expert to prevent load imbalance. Tokens over capacity are dropped. 0.0=disabled, 1.0=fair share, >1.0=buffer."
        },
    )
    dropped_token_behavior: DroppedTokenOptions | None = field(
        default=None,
        metadata={
            "help": "Controls dropped tokens. ZERO: become zero vectors (default). IDENTITY: retain original input."
        },
    )
    compute_expert_mixture_flag: bool | None = field(
        default=None,
        metadata={"help": "When true computes the expert mixture for this layer."},
    )
    weighted_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the sepected parameters will be multiplied by their probs."
        },
    )
    weighting_position_option: ExpertWeightingPositionOptions | None = field(
        default=None,
        metadata={
            "help": "Dictates if the weights are applided before or after the experts."
        },
    )
    init_sampler_option: InitSamplerOptions | None = field(
        default=None,
        metadata={
            "help": "Use `SHARED` for a single router and sampler across all layers, or `LAYER` for one per layer."
        },
    )
    router_model_config: "RouterConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    sampler_model_config: "SamplerConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


@dataclass
class _ExpertInputData:
    expert_index: int
    expert_samples: Tensor
    dropped_samples: Tensor
    expert_routing_positions: Tensor | None
    dropped_routing_positions: Tensor | None
    probabilities: Tensor | None


class MixtureOfExperts(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | ModelConfig",
        overrides: "MixtureOfExpertsConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "mixture_of_experts_config", cfg)
        self.cfg: "MixtureOfExpertsConfig" = self._overwrite_config(config, overrides)
        self.main_cfg: "LayerStackConfig" = self._resolve_main_config(self.cfg, cfg)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.layer_stack_model: "LinearLayerStackOptions" = self.cfg.layer_stack_option
        self.top_k: int = self.cfg.top_k
        self.num_experts: int = self.cfg.num_experts
        self.capacity_factor: float = self.cfg.capacity_factor
        self.dropped_token_behavior: DroppedTokenOptions = (
            self.cfg.dropped_token_behavior or DroppedTokenOptions.ZEROS
        )
        self.compute_expert_mixture_flag: bool = self.cfg.compute_expert_mixture_flag
        self.weighted_parameters_flag: bool = self.cfg.weighted_parameters_flag
        self.init_sampler_option: "InitSamplerOptions" = self.cfg.init_sampler_option
        self.weighting_position_option: "ExpertWeightingPositionOptions" = (
            self.cfg.weighting_position_option
        )
        self.router_model_config: "RouterConfig" = self.cfg.router_model_config
        self.sampler_model_config: "SamplerConfig" = self.cfg.sampler_model_config

        self.validator_handler = _ValidatorHandler(self)
        self.capacity_handler = _ExpertCapacityHandler(self.cfg)
        self.expert_weighting_handler = _ExpertWeightingHandler(self.cfg)
        self.router, self.sampler = self.__maybe_create_router_and_sampler()
        self.expert_modules = self.__create_experts()

    def get_top_k(self) -> int:
        return self.top_k

    def __maybe_create_router_and_sampler(
        self,
    ) -> tuple["RouterModel | None", "SamplerModel | None"]:
        from emperor.sampler.utils.routers import RouterConfig, RouterModel

        if self.init_sampler_option != InitSamplerOptions.LAYER:
            return None, None
        self.validator_handler.ensure_router_config_exists()
        self.validator_handler.ensure_sampler_config_exists()
        router_overrides = RouterConfig(input_dim=self.input_dim)
        router = RouterModel(self.router_model_config, router_overrides)
        sampler = SamplerModel(self.sampler_model_config)
        return router, sampler

    def __create_experts(self) -> nn.ModuleList:
        expert_list = []
        for _ in range(self.num_experts):
            overrides = LayerStackConfig(
                input_dim=self.input_dim, output_dim=self.output_dim
            )
            model_stack = self.layer_stack_model.value(
                self.main_cfg, overrides
            ).build_model()

            expert_list.append(model_stack)
        return nn.ModuleList(expert_list)

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self.validator_handler.ensure_tensor_is_vector_or_matrix(input_batch)
        self.validator_handler.ensure_tensor_is_vector_or_matrix(probabilities)
        self.validator_handler.ensure_tensor_is_vector_or_matrix(indices)
        probabilities, indices, sampler_loss = self._maybe_compute_expert_indices(
            input_batch, probabilities, indices
        )
        expert_input_data = self._split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        expert_outputs, routing_positions, probabilities, expert_loss = (
            self._compute_experts(expert_input_data, probabilities)
        )

        output = self.__compute_expert_mixture(
            expert_outputs, routing_positions, probabilities
        )

        total_loss = sampler_loss + expert_loss
        return output, total_loss

    def _maybe_compute_expert_indices(
        self,
        inputs: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.init_sampler_option != InitSamplerOptions.LAYER or (
            indices is not None or probabilities is not None
        ):
            return probabilities, indices, torch.tensor(0.0)
        self.validator_handler.ensure_sampler_is_initialized()
        self.validator_handler.ensure_external_probabilities_are_not_given(
            probabilities, indices
        )
        logits = self.router.compute_logit_scores(inputs)
        # TODO: In the future see if `skip_mask` needs to be implemented
        skip_mask = None
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, indices, sampler_loss

    def _split_tokens_per_expert(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
        indices: Tensor | None,
    ) -> list[_ExpertInputData]:
        if self.num_experts == self.top_k:
            return self.__build_dense_expert_inputs(input_batch, probabilities)
        return self.__build_routed_expert_inputs(input_batch, probabilities, indices)

    def __build_dense_expert_inputs(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
    ) -> list[_ExpertInputData]:
        empty = torch.tensor([], dtype=input_batch.dtype, device=input_batch.device)
        expert_input_data = []
        for expert_index in range(self.num_experts):
            expert_probabilities = (
                self.expert_weighting_handler.maybe_get_expert_probabilities(
                    None, probabilities, expert_index
                )
            )
            expert_input_data.append(
                _ExpertInputData(
                    expert_index=expert_index,
                    expert_samples=input_batch,
                    dropped_samples=empty,
                    expert_routing_positions=None,
                    dropped_routing_positions=None,
                    probabilities=expert_probabilities,
                )
            )
        return expert_input_data

    def __build_routed_expert_inputs(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
        indices: Tensor,
    ) -> list[_ExpertInputData]:
        expert_input_data = []
        for expert_index in range(self.num_experts):
            expert_sample_indices, dropped_sample_indices = (
                self._get_expert_token_indices(indices, expert_index)
            )
            if expert_sample_indices.numel() == 0:
                continue
            expert_routing_positions, dropped_routing_positions = (
                self._get_expert_routing_positions(indices, expert_index)
            )
            expert_samples, dropped_samples = (
                self.capacity_handler.select_expert_and_dropped_samples(
                    input_batch, expert_sample_indices, dropped_sample_indices
                )
            )
            expert_probabilities = (
                self.expert_weighting_handler.maybe_get_expert_probabilities(
                    expert_routing_positions, probabilities, expert_index
                )
            )
            expert_input_data.append(
                _ExpertInputData(
                    expert_index=expert_index,
                    expert_samples=expert_samples,
                    dropped_samples=dropped_samples,
                    expert_routing_positions=expert_routing_positions,
                    dropped_routing_positions=dropped_routing_positions,
                    probabilities=expert_probabilities,
                )
            )
        return expert_input_data

    def _get_expert_token_indices(
        self,
        indices: Tensor | None,
        expert_index: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size = indices.size(0)
        samples_for_current_expert = indices == expert_index
        if self.top_k > 1:
            samples_for_current_expert = samples_for_current_expert.sum(dim=-1)
        sample_indices_for_expert = samples_for_current_expert.nonzero().flatten()
        return self.capacity_handler.maybe_apply_capacity_limit_token_indices(
            sample_indices_for_expert, batch_size
        )

    def _get_expert_routing_positions(
        self,
        indices: Tensor,
        expert_index: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size = indices.size(0)
        boolean_tensor = indices == expert_index
        expert_sample_indices = boolean_tensor.flatten()
        expert_sample_indices = expert_sample_indices.nonzero().squeeze(dim=-1)
        return self.capacity_handler.maybe_apply_capacity_limit_routing_positions(
            expert_sample_indices, batch_size
        )

    def _compute_experts(
        self,
        experts_data: list[_ExpertInputData],
        full_probabilities: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        expert_outputs_list = []
        sample_indices_for_expert_list = []
        total_loss = torch.tensor(0.0)

        for expert_data in experts_data:
            expert_output, loss = self.__compute_expert_output(expert_data)
            self.__append_expert_output(expert_outputs_list, expert_output, expert_data)
            self.__append_sample_indices(sample_indices_for_expert_list, expert_data)
            total_loss = total_loss + loss

        expert_outputs = torch.cat(expert_outputs_list, dim=0)
        routing_positions, reindexed_probs = self.__aggregate_sample_indices(
            sample_indices_for_expert_list, full_probabilities
        )

        return (
            expert_outputs,
            routing_positions,
            reindexed_probs,
            total_loss,
        )

    def __compute_expert_output(
        self,
        expert_data: _ExpertInputData,
    ) -> tuple[Tensor, Tensor]:
        expert_samples = self.expert_weighting_handler.maybe_apply_probabilities_before(
            expert_data.expert_samples, expert_data.probabilities
        )

        expert_model = self.expert_modules[expert_data.expert_index]
        output = expert_model(expert_samples)
        return output, torch.tensor(0.0)

    def __append_expert_output(
        self,
        expert_outputs_list: list[Tensor],
        expert_output: Tensor,
        expert_data: _ExpertInputData,
    ) -> None:
        if expert_data.dropped_samples.numel() > 0:
            expert_output = torch.cat(
                [expert_output, expert_data.dropped_samples], dim=0
            )
        expert_outputs_list.append(expert_output)

    def __append_sample_indices(
        self,
        sample_indices_for_expert_list: list[Tensor],
        expert_data: _ExpertInputData,
    ) -> None:
        if self.top_k != self.num_experts:
            sample_indices = torch.cat(
                [
                    expert_data.expert_routing_positions,
                    expert_data.dropped_routing_positions,
                ],
                dim=0,
            )
            sample_indices_for_expert_list.append(sample_indices)

    def __aggregate_sample_indices(
        self,
        sample_indices_for_expert_list: list[Tensor],
        full_probabilities: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.top_k == self.num_experts:
            return None, full_probabilities
        routing_positions = torch.cat(sample_indices_for_expert_list)
        reindexed_probs = full_probabilities
        if self.capacity_factor > 0:
            assert full_probabilities is not None
            reindexed_probs = full_probabilities.flatten()[routing_positions]
        return routing_positions, reindexed_probs

    def __compute_expert_mixture(
        self,
        experts_output: Tensor,
        indices: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        output_dim = experts_output.size(-1)
        if self.top_k != self.num_experts and indices is not None:
            _, _index_sorted_indices = indices.sort(dim=0)
            experts_output = experts_output[_index_sorted_indices]

        experts_output = self.expert_weighting_handler.maybe_apply_probabilities_after(
            experts_output, probabilities
        )

        if not self.compute_expert_mixture_flag or self.top_k == 1:
            return experts_output

        if self.top_k > 1:
            experts_output = experts_output.view(-1, self.top_k, output_dim)
        return experts_output.sum(dim=1)


class MixtureOfExpertsMap(MixtureOfExperts):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | ModelConfig",
        overrides: "MixtureOfExpertsConfig | None" = None,
    ):
        overrides = self.__update_overrides(overrides)
        super().__init__(cfg, overrides)
        self.routing_positions = []

        self.routing_positions = None
        self.sample_probabilities = None

    def __update_overrides(
        self, overrides: "MixtureOfExpertsConfig | None" = None
    ) -> MixtureOfExpertsConfig:
        if overrides is None:
            return MixtureOfExpertsConfig(
                weighted_parameters_flag=False,
                compute_expert_mixture_flag=False,
                init_sampler_option=InitSamplerOptions.DISABLED,
            )
        overrides.weighted_parameters_flag = False
        overrides.compute_expert_mixture_flag = False
        overrides.init_sampler_option = InitSamplerOptions.DISABLED

        return overrides


class MixtureOfExpertsReduce(MixtureOfExperts):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | ModelConfig",
        overrides: "MixtureOfExpertsConfig | None" = None,
    ):
        overrides = self.__update_overrides(overrides)
        super().__init__(cfg, overrides)

    def __update_overrides(
        self, overrides: "MixtureOfExpertsConfig | None" = None
    ) -> MixtureOfExpertsConfig:
        if overrides is None:
            return MixtureOfExpertsConfig(
                weighted_parameters_flag=True,
                compute_expert_mixture_flag=True,
                weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
                init_sampler_option=InitSamplerOptions.DISABLED,
            )
        overrides.weighted_parameters_flag = True
        overrides.compute_expert_mixture_flag = True
        overrides.weighting_position_option = (
            ExpertWeightingPositionOptions.AFTER_EXPERTS
        )
        overrides.init_sampler_option = InitSamplerOptions.DISABLED

        return overrides

    def _split_tokens_per_expert(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> list[_ExpertInputData]:
        expert_input_data = []
        empty_dropped = torch.zeros(
            0, dtype=input_batch.dtype, device=input_batch.device
        )
        for expert_index in range(self.num_experts):
            expert_routing_positions, dropped_routing_positions = (
                self._get_expert_routing_positions(indices, expert_index)
            )
            if (
                expert_routing_positions is not None
                and expert_routing_positions.numel() == 0
            ):
                continue
            expert_samples = (
                input_batch[expert_routing_positions]
                if expert_routing_positions is not None
                else input_batch
            )
            expert_input_data.append(
                _ExpertInputData(
                    expert_index=expert_index,
                    expert_samples=expert_samples,
                    dropped_samples=empty_dropped,
                    expert_routing_positions=expert_routing_positions,
                    dropped_routing_positions=dropped_routing_positions,
                    probabilities=None,
                )
            )
        return expert_input_data

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self.validator_handler.ensure_tensor_is_vector_or_matrix(input_batch)
        self.validator_handler.ensure_tensor_is_vector_or_matrix(probabilities)
        self.validator_handler.ensure_tensor_is_vector_or_matrix(indices)

        expert_input_data = self._split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        expert_outputs, routing_positions, probabilities, expert_loss = (
            self._compute_experts(expert_input_data, probabilities)
        )
        output = self.__compute_expert_mixture(
            expert_outputs, routing_positions, probabilities
        )
        return output, expert_loss

    def _compute_experts(
        self,
        expert_input_data: list[_ExpertInputData],
        full_probabilities: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        expert_outputs, routing_positions, _, total_loss = super()._compute_experts(
            expert_input_data, full_probabilities
        )
        expert_outputs, probabilities = (
            self.capacity_handler.maybe_reconstruct_full_batch_from_expert_outputs(
                expert_outputs, full_probabilities, routing_positions
            )
        )
        return expert_outputs, routing_positions, probabilities, total_loss

    def __compute_expert_mixture(
        self,
        experts_output: Tensor,
        indices: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        output_dim = experts_output.size(-1)
        if self.top_k != self.num_experts and indices is not None:
            _, _index_sorted_indices = indices.sort(dim=0)
            experts_output = experts_output[_index_sorted_indices]

        experts_output = self.expert_weighting_handler.maybe_apply_probabilities_after(
            experts_output, probabilities
        )

        if not self.compute_expert_mixture_flag or self.top_k == 1:
            return experts_output

        if self.top_k > 1:
            experts_output = experts_output.view(-1, self.top_k, output_dim)
        return experts_output.sum(dim=1)
