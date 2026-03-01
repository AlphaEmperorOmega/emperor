import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from dataclasses import dataclass, field
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.halting.config import HaltingConfig


@dataclass
class StickBreakingState:
    halt_mask: Tensor = field(
        metadata={
            "help": "Boolean mask indicating which tokens have accumulated enough halt probability to stop computing"
        },
    )
    log_never_halt: Tensor = field(
        metadata={
            "help": "Log probability of never having halted up to the current step; represents the remaining stick"
        },
    )
    accumulated_hidden: Tensor = field(
        metadata={
            "help": "Weighted sum of hidden states accumulated so far, where each step contributes proportionally to its halt probability"
        },
    )
    accumulated_halt_prob: Tensor = field(
        metadata={
            "help": "Total halt probability spent across all steps so far; halting is triggered when this exceeds the threshold"
        },
    )
    step: int = field(
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

        self.input_dim: int = self.cfg.input_dim
        self.threshold: float = self.cfg.threshold

        self._gate: nn.Sequential = self.__build_gate()
        self.__init_gate_weights()

    def __build_gate(self) -> nn.Sequential:
        return nn.Sequential(nn.Linear(self.input_dim, 2, bias=False))

    def __init_gate_weights(self) -> None:
        nn.init.zeros_(self._gate[-1].weight)

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
        current_log_gates: Tensor,
        previous_output: Tensor,
    ) -> StickBreakingState:
        log_continuation = current_log_gates[..., 0]
        log_halting = current_log_gates[..., 1]
        halting_probability = torch.exp(log_halting)
        return StickBreakingState(
            halt_mask=halting_probability >= self.threshold,
            log_never_halt=log_continuation,
            accumulated_hidden=halting_probability[..., None] * previous_output,
            accumulated_halt_prob=halting_probability,
            step=0,
            accumulated_expected_step=torch.tensor(0.0),
        )

    def __update_state(
        self,
        previous_state: StickBreakingState,
        current_log_gates: Tensor,
        previous_output: Tensor,
    ) -> StickBreakingState:
        updated_step = previous_state.step + 1
        current_log_halting = (
            previous_state.log_never_halt[..., None] + current_log_gates
        )
        halting_probability = torch.exp(current_log_halting[..., 1])
        halting_probability = halting_probability.masked_fill(
            previous_state.halt_mask, 0.0
        )
        accumulated_halting_probability = (
            previous_state.accumulated_halt_prob + halting_probability
        )
        return StickBreakingState(
            halt_mask=accumulated_halting_probability >= self.threshold,
            log_never_halt=current_log_halting[..., 0],
            accumulated_hidden=previous_state.accumulated_hidden
            + halting_probability[..., None] * previous_output,
            accumulated_halt_prob=accumulated_halting_probability,
            step=updated_step,
            accumulated_expected_step=previous_state.accumulated_expected_step
            + halting_probability * updated_step,
        )

    def halt_gating(
        self,
        state: StickBreakingState,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        remaining_probabilities = 1 - state.accumulated_halt_prob
        soft_halted_hidden = (
            state.accumulated_hidden
            + remaining_probabilities[..., None] * current_hidden
        )
        expected_step = (
            state.accumulated_expected_step + (state.step + 1) * remaining_probabilities
        )
        if state.halt_mask.any():
            soft_halted_hidden.masked_scatter_(
                state.halt_mask[..., None],
                state.accumulated_hidden[state.halt_mask],
            )
        return soft_halted_hidden, expected_step
