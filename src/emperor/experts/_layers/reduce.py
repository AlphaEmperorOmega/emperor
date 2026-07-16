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
        self.VALIDATOR.validate_reduce_forward_inputs(
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
