from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.experts._config import MixtureOfExpertsConfig
from emperor.experts._options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.experts._routing.capacity import ExpertCapacityHandler
from emperor.experts._routing.weighting import ExpertWeightingHandler
from emperor.experts._validation.mixture import MixtureOfExpertsValidator
from emperor.layers import Layer, LayerStackConfig, RecurrentLayerConfig
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.sampler import SamplerConfig, SamplerModel


@dataclass
class ExpertInputData:
    expert_index: int
    expert_samples: Tensor
    dropped_samples: Tensor
    expert_routing_positions: Tensor | None
    dropped_routing_positions: Tensor | None
    probabilities: Tensor | None


class MixtureOfExperts(Module):
    VALIDATOR = MixtureOfExpertsValidator

    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig",
        overrides: "MixtureOfExpertsConfig | None" = None,
    ):
        super().__init__()
        self.cfg: MixtureOfExpertsConfig = self._override_config(cfg, overrides)
        self.main_cfg: LayerStackConfig | RecurrentLayerConfig = (
            self.cfg.expert_model_config
        )

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.expert_model_config: LayerStackConfig | RecurrentLayerConfig = (
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
        self.routing_initialization_mode: RoutingInitializationMode = (
            self.cfg.routing_initialization_mode
        )
        self.weighting_position_option: ExpertWeightingPositionOptions = (
            self.cfg.weighting_position_option
        )
        self.sampler_config: SamplerConfig = self.cfg.sampler_config

        self.VALIDATOR.validate(self)
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
        self.VALIDATOR.validate_sampler_config_exists(self)
        self.VALIDATOR.validate_router_config_exists(self)
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
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        self.VALIDATOR.validate_forward_inputs(
            self, input_batch, probabilities, indices, skip_mask
        )
        probabilities, indices, skip_mask, sampler_loss = (
            self._maybe_compute_expert_indices(
                input_batch, probabilities, indices, skip_mask
            )
        )
        expert_input_data = self._split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        expert_outputs, routing_positions, probabilities, expert_loss = (
            self._compute_experts(expert_input_data, probabilities)
        )
        mixture_output = self.__compute_expert_mixture(
            expert_outputs, routing_positions, probabilities
        )
        total_loss = sampler_loss + expert_loss
        return mixture_output, skip_mask, total_loss

    def _maybe_compute_expert_indices(
        self,
        inputs: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor]:
        if self.__should_skip_layer_routing(probabilities, indices):
            return probabilities, indices, skip_mask, inputs.new_zeros(())
        self.VALIDATOR.validate_sampler_is_initialized(self)
        self.VALIDATOR.validate_external_probabilities_are_not_given(
            probabilities, indices
        )
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(inputs, skip_mask)
        )
        return probabilities, indices, skip_mask, sampler_loss

    def __should_skip_layer_routing(
        self,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> bool:
        layer_routing_is_enabled = (
            self.routing_initialization_mode == RoutingInitializationMode.LAYER
        )
        external_routing_is_provided = indices is not None or probabilities is not None
        return not layer_routing_is_enabled or external_routing_is_provided

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
        if indices.dim() > 1:
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
        outputs_grouped_by_expert = []
        routing_positions_by_expert = []
        total_loss = experts_data[0].expert_samples.new_zeros(())

        for expert_data in experts_data:
            expert_output, expert_loss = self.__compute_expert_output(expert_data)
            self.__append_expert_output(
                outputs_grouped_by_expert,
                expert_output,
                expert_data,
            )
            self.__append_sample_indices(routing_positions_by_expert, expert_data)
            total_loss = total_loss + expert_loss

        expert_outputs = torch.cat(outputs_grouped_by_expert, dim=0)
        if self.top_k == self.num_experts:
            output_dim = expert_outputs.size(-1)
            expert_major_outputs = expert_outputs.reshape(
                self.num_experts, -1, output_dim
            )
            sample_major_outputs = expert_major_outputs.transpose(0, 1)
            flattened_sample_major_outputs = sample_major_outputs.reshape(
                -1, output_dim
            )
            expert_outputs = flattened_sample_major_outputs
        routing_positions, routing_probabilities = self.__aggregate_sample_indices(
            routing_positions_by_expert, full_probabilities
        )

        return (
            expert_outputs,
            routing_positions,
            routing_probabilities,
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
        return routing_positions, full_probabilities

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
