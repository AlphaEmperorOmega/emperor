import torch
import torch.nn as nn

from torch import Tensor
from dataclasses import dataclass
from emperor.base.layer import Layer, LayerStackConfig, RecurrentLayerConfig
from emperor.base.module import Module
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core._expert_capacity import ExpertCapacityHandler
from emperor.experts.core._expert_weighting import ExpertWeightingHandler
from emperor.experts.core._validator import MixtureOfExpertsValidator
from emperor.experts.core.state import MixtureOfExpertsLayerState
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.core.config import RouterConfig, SamplerConfig
    from emperor.sampler.model import SamplerModel


@dataclass
class ExpertInputData:
    expert_index: int
    expert_samples: Tensor
    dropped_samples: Tensor
    expert_routing_positions: Tensor | None
    dropped_routing_positions: Tensor | None
    probabilities: Tensor | None


class MixtureOfExperts(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig",
        overrides: "MixtureOfExpertsConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "MixtureOfExpertsConfig" = self._override_config(cfg, overrides)
        self.main_cfg: "LayerStackConfig | RecurrentLayerConfig" = (
            self.cfg.expert_model_config
        )

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.expert_model_config: "LayerStackConfig | RecurrentLayerConfig" = (
            self.cfg.expert_model_config
        )
        self.top_k: int = self.cfg.top_k
        self.num_experts: int = self.cfg.num_experts
        self.capacity_factor: float = self.cfg.capacity_factor
        self.dropped_token_behavior: DroppedTokenOptions = (
            self.cfg.dropped_token_behavior or DroppedTokenOptions.ZEROS
        )
        self.compute_expert_mixture_flag: bool = self.cfg.compute_expert_mixture_flag
        self.weighted_parameters_flag: bool = self.cfg.weighted_parameters_flag
        self.routing_initialization_mode: "RoutingInitializationMode" = (
            self.cfg.routing_initialization_mode
        )
        self.weighting_position_option: "ExpertWeightingPositionOptions" = (
            self.cfg.weighting_position_option
        )
        self.sampler_config: "SamplerConfig" = self.cfg.sampler_config

        MixtureOfExpertsValidator.validate(self)
        self.capacity_handler = ExpertCapacityHandler(self.cfg)
        self.expert_weighting_handler = ExpertWeightingHandler(self.cfg)
        self.sampler = self.__maybe_create_sampler()
        self.expert_modules = self.__create_experts()

    def get_top_k(self) -> int:
        return self.top_k

    def __maybe_create_sampler(
        self,
    ) -> "SamplerModel | None":
        if self.routing_initialization_mode != RoutingInitializationMode.LAYER:
            return None
        MixtureOfExpertsValidator.validate_sampler_config_exists(self)
        MixtureOfExpertsValidator.validate_router_config_exists(self)
        return self.sampler_config.build_with_router_input_dim(self.input_dim)

    def __create_experts(self) -> nn.ModuleList:
        expert_list = []
        for _ in range(self.num_experts):
            overrides = LayerStackConfig(
                input_dim=self.input_dim, output_dim=self.output_dim
            )
            model_stack = self.main_cfg.build(overrides)

            expert_list.append(model_stack)
        return nn.ModuleList(expert_list)

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.__compute_routed_expert_output(input_batch, probabilities, indices)

    def __compute_routed_expert_output(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        MixtureOfExpertsValidator.validate_forward_inputs(
            self, input_batch, probabilities, indices
        )
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
        if self.routing_initialization_mode != RoutingInitializationMode.LAYER or (
            indices is not None or probabilities is not None
        ):
            return probabilities, indices, inputs.new_zeros(())
        MixtureOfExpertsValidator.validate_sampler_is_initialized(self)
        MixtureOfExpertsValidator.validate_external_probabilities_are_not_given(
            probabilities, indices
        )
        # TODO: In the future see if `skip_mask` needs to be implemented
        skip_mask = None
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(inputs, skip_mask)
        )
        return probabilities, indices, sampler_loss

    def _split_tokens_per_expert(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
        indices: Tensor | None,
    ) -> list[ExpertInputData]:
        if self.num_experts == self.top_k:
            return self.__build_dense_expert_inputs(input_batch, probabilities)
        return self.__build_routed_expert_inputs(input_batch, probabilities, indices)

    def __build_dense_expert_inputs(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
    ) -> list[ExpertInputData]:
        empty = torch.tensor([], dtype=input_batch.dtype, device=input_batch.device)
        expert_input_data = []
        for expert_index in range(self.num_experts):
            expert_probabilities = (
                self.expert_weighting_handler.maybe_get_expert_probabilities(
                    None, probabilities, expert_index
                )
            )
            expert_input_data.append(
                ExpertInputData(
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
    ) -> list[ExpertInputData]:
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
                ExpertInputData(
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
        experts_data: list[ExpertInputData],
        full_probabilities: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        expert_outputs_list = []
        sample_indices_for_expert_list = []
        total_loss = experts_data[0].expert_samples.new_zeros(())

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
        expert_data: ExpertInputData,
    ) -> tuple[Tensor, Tensor]:
        expert_samples = self.expert_weighting_handler.maybe_apply_probabilities_before(
            expert_data.expert_samples, expert_data.probabilities
        )

        expert_model = self.expert_modules[expert_data.expert_index]
        output = Layer.run_model_returning_hidden(expert_model, expert_samples)
        return output, expert_samples.new_zeros(())

    def __append_expert_output(
        self,
        expert_outputs_list: list[Tensor],
        expert_output: Tensor,
        expert_data: ExpertInputData,
    ) -> None:
        if expert_data.dropped_samples.numel() > 0:
            expert_output = torch.cat(
                [expert_output, expert_data.dropped_samples], dim=0
            )
        expert_outputs_list.append(expert_output)

    def __append_sample_indices(
        self,
        sample_indices_for_expert_list: list[Tensor],
        expert_data: ExpertInputData,
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


class MixtureOfExpertsLayer(Layer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: "MixtureOfExpertsLayerState",
    ) -> Tensor:
        output, loss = self.model(main_model_input, state.probabilities, state.indices)
        state.loss = loss if state.loss is None else state.loss + loss
        return output


class MixtureOfExpertsMap(MixtureOfExperts):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig",
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
                routing_initialization_mode=RoutingInitializationMode.DISABLED,
            )
        overrides.weighted_parameters_flag = False
        overrides.compute_expert_mixture_flag = False
        overrides.routing_initialization_mode = RoutingInitializationMode.DISABLED

        return overrides


class MixtureOfExpertsReduce(MixtureOfExperts):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig",
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
                routing_initialization_mode=RoutingInitializationMode.DISABLED,
            )
        overrides.weighted_parameters_flag = True
        overrides.compute_expert_mixture_flag = True
        overrides.weighting_position_option = (
            ExpertWeightingPositionOptions.AFTER_EXPERTS
        )
        overrides.routing_initialization_mode = RoutingInitializationMode.DISABLED

        return overrides

    def _split_tokens_per_expert(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> list[ExpertInputData]:
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
                ExpertInputData(
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
        MixtureOfExpertsValidator.validate_reduce_forward_inputs(
            self, input_batch, probabilities, indices
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
        return output, expert_loss

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
