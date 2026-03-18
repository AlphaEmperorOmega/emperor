from torch import Tensor
from emperor.experts.utils.enums import ExpertWeightingPositionOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.experts.utils.layers import MixtureOfExpertsConfig


class _ExpertWeightingHandler:
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig",
    ):
        self.cfg = cfg
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag
        self.weighting_position_option = self.cfg.weighting_position_option
        self.top_k = self.cfg.top_k
        self.num_experts = self.cfg.num_experts

    def maybe_get_expert_probabilities(
        self,
        indices: Tensor | None,
        probabilities: Tensor,
        expert_index: int,
    ) -> Tensor | None:
        if self.__should_probabilities_apply_before():
            if self.top_k == self.num_experts:
                return probabilities[:, expert_index]
            probabilities = probabilities.flatten()
            probabilities = probabilities[indices]
            return probabilities
        return None

    def maybe_apply_probabilities_before(
        self,
        experts_output: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        if self.__should_probabilities_apply_before():
            experts_output = self.__maybe_apply_probabilities(
                experts_output, probabilities
            )
        return experts_output

    def maybe_apply_probabilities_after(
        self,
        experts_output: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        if self.__should_probabilities_apply_after():
            experts_output = self.__maybe_apply_probabilities(
                experts_output, probabilities
            )
        return experts_output

    def __should_probabilities_apply_before(self) -> bool:
        position_option = ExpertWeightingPositionOptions.BEFORE_EXPERTS
        return self.weighting_position_option == position_option

    def __should_probabilities_apply_after(self) -> bool:
        position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS
        return self.weighting_position_option == position_option

    def __maybe_apply_probabilities(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        if not self.weighted_parameters_flag:
            return logits

        if probabilities is None:
            raise ValueError(
                "Missing input: `probabilities` must be supplied when `indices` are used "
                "to ensure accurate weighting and processing of inputs."
            )
        return logits * probabilities.reshape(-1, 1)
