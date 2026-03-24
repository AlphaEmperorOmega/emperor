import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from emperor.base.utils import Module
from emperor.base.layer import Layer, LayerStackConfig
from emperor.base.enums import LastLayerBiasOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.halting.config import HaltingConfig


@dataclass
class StickBreakingState:
    halt_mask: Tensor = field(
        metadata={
            "help": "Boolean mask indicating which tokens have accumulated enough halt probability to stop computing"
        },
    )
    log_remaining_stick: Tensor = field(
        metadata={
            "help": "Log of the remaining stick length after all breaks so far; the cumulative log probability of not yet having halted"
        },
    )
    accumulated_hidden: Tensor = field(
        metadata={
            "help": "Weighted sum of hidden states accumulated so far, where each step contributes proportionally to its halt probability"
        },
    )
    accumulated_halt_probabilities: Tensor = field(
        metadata={
            "help": "Total halt probability spent across all steps so far; halting is triggered when this exceeds the threshold"
        },
    )
    step_count: int = field(
        metadata={
            "help": "Current step index, used to compute the expected number of steps for regularisation"
        },
    )
    accumulated_expected_step: Tensor = field(
        metadata={
            "help": "Running sum of halt_prob * step across all steps; used to compute the expected computation depth"
        },
    )


class StickBreaking(Module):
    def __init__(
        self,
        cfg: "HaltingConfig | ModelConfig",
        overrides: "HaltingConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "halting_config", cfg)
        self.cfg: "HaltingConfig" = self._overwrite_config(config, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)

        self.input_dim: int = self.cfg.input_dim
        self.threshold: float = self.cfg.threshold

        self._gate: Layer | Sequential = self.__build_gate()
        self.__init_gate_weights()

    def __build_gate(self) -> "Layer | Sequential":
        from emperor.linears.utils.stack import LinearLayerStack

        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=2,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        )
        return LinearLayerStack(self.main_cfg, overrides).build_model()

    def __init_gate_weights(self) -> None:
        last_linear = None
        for m in self._gate.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)

    def forward(
        self,
        previous_state: StickBreakingState | None,
        previous_output: Tensor,
    ) -> StickBreakingState:
        current_log_gates = self.__compute_gate_logits(previous_output)
        if previous_state is None:
            return self.__init_state(current_log_gates, previous_output)
        return self.__update_state(previous_state, current_log_gates, previous_output)

    def __compute_gate_logits(self, hidden_state: Tensor) -> Tensor:
        logits = self._gate(hidden_state)
        if self.training:
            logits = logits + torch.randn_like(logits)
        return F.log_softmax(logits, dim=-1)

    def __init_state(
        self,
        log_softmax_gates: Tensor,
        previous_output: Tensor,
    ) -> StickBreakingState:
        log_continuation, log_halting = torch.unbind(log_softmax_gates, dim=-1)
        halting_probability = torch.exp(log_halting)
        has_halted = halting_probability >= self.threshold
        weighted_hidden = halting_probability.unsqueeze(-1) * previous_output
        return StickBreakingState(
            halt_mask=has_halted,
            log_remaining_stick=log_continuation,
            accumulated_hidden=weighted_hidden,
            accumulated_halt_probabilities=halting_probability,
            step_count=0,
            accumulated_expected_step=torch.tensor(0.0),
        )

    def __update_state(
        self,
        previous_state: StickBreakingState,
        log_softmax_gates: Tensor,
        previous_output: Tensor,
    ) -> StickBreakingState:
        updated_step_count = previous_state.step_count + 1
        log_continuation, halting_probability = self.__compute_step_halting_probability(
            previous_state, log_softmax_gates
        )
        accumulated_halting_probability = (
            previous_state.accumulated_halt_probabilities + halting_probability
        )
        has_halted = accumulated_halting_probability >= self.threshold
        weighted_hidden = halting_probability.unsqueeze(-1) * previous_output
        updated_accumulated_hidden = previous_state.accumulated_hidden + weighted_hidden
        step_contribution = halting_probability * updated_step_count
        updated_accumulated_expected_step = (
            previous_state.accumulated_expected_step + step_contribution
        )
        return StickBreakingState(
            halt_mask=has_halted,
            log_remaining_stick=log_continuation,
            accumulated_hidden=updated_accumulated_hidden,
            accumulated_halt_probabilities=accumulated_halting_probability,
            step_count=updated_step_count,
            accumulated_expected_step=updated_accumulated_expected_step,
        )

    def __compute_step_halting_probability(
        self,
        previous_state: StickBreakingState,
        current_log_gates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        current_log_halting = (
            previous_state.log_remaining_stick.unsqueeze(-1) + current_log_gates
        )
        log_continuation, log_halting = torch.unbind(current_log_halting, dim=-1)
        halting_probability = torch.exp(log_halting)
        halting_probability = halting_probability.masked_fill(
            previous_state.halt_mask, 0.0
        )
        return log_continuation, halting_probability

    def halt_gating(
        self,
        state: StickBreakingState,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        remaining_probabilities = 1 - state.accumulated_halt_probabilities
        weighted_remaining_hidden = (
            remaining_probabilities.unsqueeze(-1) * current_hidden
        )
        soft_halted_hidden = state.accumulated_hidden + weighted_remaining_hidden
        expected_step = (
            state.accumulated_expected_step + (state.step_count + 1) * remaining_probabilities
        )
        if state.halt_mask.any():
            soft_halted_hidden.masked_scatter_(
                state.halt_mask.unsqueeze(-1),
                state.accumulated_hidden[state.halt_mask],
            )
        return soft_halted_hidden, expected_step
