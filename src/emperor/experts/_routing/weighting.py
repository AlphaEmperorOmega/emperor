from typing import TYPE_CHECKING

from torch import Tensor

from emperor.experts._options import ExpertWeightingPositionOptions
from emperor.experts._validation.mixture import MixtureOfExpertsValidator

if TYPE_CHECKING:
    from emperor.experts._config import MixtureOfExpertsConfig


class ExpertWeightingHandler:
    VALIDATOR = MixtureOfExpertsValidator

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
                if probabilities.dim() == 1:
                    return probabilities
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

        self.VALIDATOR.validate_probabilities_exist(probabilities)
        probabilities_as_column_vector = probabilities.reshape(-1, 1)
        return logits * probabilities_as_column_vector
