from torch import Tensor

from emperor.experts.utils._validator import _ValidatorHandler
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
        indices: Tensor,
        probabilities: Tensor,
        expert_index: int,
    ) -> Tensor:
        if self._should_apply_before():
            if self.top_k == self.num_experts:
                return probabilities[:, expert_index]
            probabilities = probabilities.flatten()
            probabilities = probabilities[indices]

        return probabilities

    def maybe_apply_probabilities_before(
        self,
        experts_output: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        if self._should_apply_before():
            experts_output = self._maybe_apply(experts_output, probabilities)
        return experts_output

    def maybe_apply_after(
        self,
        experts_output: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        if self._should_apply_after():
            experts_output = self._maybe_apply(experts_output, probabilities)
        return experts_output

    def _should_apply_before(self) -> bool:
        position_option = ExpertWeightingPositionOptions.BEFORE_EXPERTS
        return self.weighting_position_option == position_option

    def _should_apply_after(self) -> bool:
        position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS
        return self.weighting_position_option == position_option

    def _maybe_apply(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        if not self.weighted_parameters_flag:
            return logits

        self.validator.ensure_probabilities_exist(probabilities)
        return logits * probabilities.reshape(-1, 1)
