import torch
from torch import Tensor

from emperor.experts._config import MixtureOfExpertsConfig
from emperor.experts._layers.mixture import ExpertInputData, MixtureOfExperts
from emperor.experts._options import (
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)


class MixtureOfExpertsReduce(MixtureOfExperts):
    def __init__(
        self,
        cfg: MixtureOfExpertsConfig,
        overrides: MixtureOfExpertsConfig | None = None,
    ):
        overrides = self.__update_overrides(overrides)
        super().__init__(cfg, overrides)

    def __update_overrides(
        self, overrides: MixtureOfExpertsConfig | None = None
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
        device = input_batch.device
        expert_input_data = []
        empty_dropped = torch.zeros(0, dtype=input_batch.dtype, device=device)
        for expert_index in range(self.num_experts):
            expert_routing_positions, dropped_routing_positions = (
                self.__resolve_expert_routing_positions(
                    input_batch, indices, expert_index, device
                )
            )
            if self.__should_skip_expert_without_routed_samples(
                expert_routing_positions
            ):
                continue
            expert_samples = self.__select_expert_samples(
                input_batch, expert_routing_positions
            )
            expert_input = ExpertInputData(
                expert_index=expert_index,
                expert_samples=expert_samples,
                dropped_samples=empty_dropped,
                expert_routing_positions=expert_routing_positions,
                dropped_routing_positions=dropped_routing_positions,
                probabilities=None,
            )
            expert_input_data.append(expert_input)
        return expert_input_data

    def __resolve_expert_routing_positions(
        self,
        input_batch: Tensor,
        indices: Tensor | None,
        expert_index: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if indices is None:
            return self.__get_dense_expert_routing_positions(
                input_batch, expert_index, device
            )
        return self._get_expert_routing_positions(indices, expert_index)

    def __should_skip_expert_without_routed_samples(
        self,
        expert_routing_positions: Tensor | None,
    ) -> bool:
        return (
            expert_routing_positions is not None
            and expert_routing_positions.numel() == 0
        )

    def __get_dense_expert_routing_positions(
        self,
        input_batch: Tensor,
        expert_index: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        num_routed_samples = input_batch.size(0)
        expert_routing_positions = torch.arange(
            expert_index,
            num_routed_samples,
            self.num_experts,
            device=device,
        )
        dropped_routing_positions = torch.empty(0, dtype=torch.long, device=device)
        return expert_routing_positions, dropped_routing_positions

    def __select_expert_samples(
        self,
        input_batch: Tensor,
        expert_routing_positions: Tensor | None,
    ) -> Tensor:
        if expert_routing_positions is None:
            return input_batch
        return input_batch[expert_routing_positions]

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        self.VALIDATOR.validate_reduce_forward_inputs(
            self, input_batch, probabilities, indices, skip_mask
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
        return output, skip_mask, expert_loss

    def __compute_expert_mixture(
        self,
        experts_output: Tensor,
        routing_positions: Tensor | None,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        output_dim = experts_output.size(-1)
        if self.__should_restore_routing_order(routing_positions):
            _, routing_position_sort_order = routing_positions.sort(dim=0)
            experts_output = experts_output[routing_position_sort_order]

        experts_output = self.expert_weighting_handler.maybe_apply_probabilities_after(
            experts_output, probabilities
        )

        if self.__should_return_expert_outputs_without_reduction():
            return experts_output

        experts_output = experts_output.view(-1, self.top_k, output_dim)
        return experts_output.sum(dim=1)

    def __should_restore_routing_order(self, routing_positions: Tensor | None) -> bool:
        return self.top_k != self.num_experts and routing_positions is not None

    def __should_return_expert_outputs_without_reduction(self) -> bool:
        return not self.compute_expert_mixture_flag or self.top_k == 1
