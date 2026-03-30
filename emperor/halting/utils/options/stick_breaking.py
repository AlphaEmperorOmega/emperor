import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from emperor.base.layer import Layer, LayerStackConfig
from emperor.base.enums import LastLayerBiasOptions
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.halting.utils.options.base import HaltingBase

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
    log_continuation: Tensor = field(
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
    accumulated_ponder_cost: Tensor = field(
        metadata={
            "help": "Running sum of halt_prob * step across all steps; used to compute the expected computation depth"
        },
    )


class StickBreaking(HaltingBase[StickBreakingState]):
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
        self.hidden_state_mode: HaltingHiddenStateModeOptions = (
            self.cfg.hidden_state_mode
        )

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

    def update_halting_state(
        self,
        previous_state: StickBreakingState | None,
        model_hidden_state: Tensor,
    ) -> tuple[StickBreakingState, Tensor]:
        current_log_gates = self.__compute_gate_logits(model_hidden_state)
        if previous_state is None:
            state = self.__init_state(current_log_gates, model_hidden_state)
        else:
            state = self.__update_state(
                previous_state, current_log_gates, model_hidden_state
            )
        if self.hidden_state_mode == HaltingHiddenStateModeOptions.ACCUMULATED:
            return state, state.accumulated_hidden
        return state, model_hidden_state

    def __compute_gate_logits(self, hidden_state: Tensor) -> Tensor:
        logits = self._gate(hidden_state)
        if self.training:
            logits = logits + torch.randn_like(logits)
        return F.log_softmax(logits, dim=-1)

    def __init_state(
        self,
        log_softmax_gates: Tensor,
        model_hidden_state: Tensor,
    ) -> StickBreakingState:
        log_continuation, log_halting = torch.unbind(log_softmax_gates, dim=-1)
        halting_probability = torch.exp(log_halting)
        halt_mask = halting_probability >= self.threshold
        weighted_hidden = halting_probability.unsqueeze(-1) * model_hidden_state
        return StickBreakingState(
            halt_mask=halt_mask,
            log_continuation=log_continuation,
            accumulated_hidden=weighted_hidden,
            accumulated_halt_probabilities=halting_probability,
            step_count=0,
            accumulated_ponder_cost=torch.tensor(0.0),
        )

    def __update_state(
        self,
        previous_state: StickBreakingState,
        log_softmax_gates: Tensor,
        model_hidden_state: Tensor,
    ) -> StickBreakingState:
        updated_step_count = previous_state.step_count + 1
        log_continuation, halting_probability = self.__compute_step_halting_probability(
            previous_state, log_softmax_gates
        )
        accumulated_halting_probability = (
            previous_state.accumulated_halt_probabilities + halting_probability
        )
        halt_mask = accumulated_halting_probability >= self.threshold
        weighted_hidden = halting_probability.unsqueeze(-1) * model_hidden_state
        updated_accumulated_hidden = previous_state.accumulated_hidden + weighted_hidden
        step_contribution = halting_probability * updated_step_count
        updated_accumulated_ponder_cost = (
            previous_state.accumulated_ponder_cost + step_contribution
        )
        return StickBreakingState(
            halt_mask=halt_mask,
            log_continuation=log_continuation,
            accumulated_hidden=updated_accumulated_hidden,
            accumulated_halt_probabilities=accumulated_halting_probability,
            step_count=updated_step_count,
            accumulated_ponder_cost=updated_accumulated_ponder_cost,
        )

    def __compute_step_halting_probability(
        self,
        previous_state: StickBreakingState,
        current_log_gates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        previous_step_log_continuation = previous_state.log_continuation.unsqueeze(-1)
        current_log_halting = previous_step_log_continuation + current_log_gates
        log_continuation, log_halting = torch.unbind(current_log_halting, dim=-1)
        halting_probability = torch.exp(log_halting)
        previous_halt_mask = previous_state.halt_mask
        halting_probability = halting_probability.masked_fill(previous_halt_mask, 0.0)
        return log_continuation, halting_probability

    def finalize_weighted_accumulation(
        self,
        state: StickBreakingState,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        remaining_probabilities = 1 - state.accumulated_halt_probabilities
        remaining_probabilities = remaining_probabilities.masked_fill(
            state.halt_mask, 0.0
        )
        weighted_remaining_hidden = (
            remaining_probabilities.unsqueeze(-1) * current_hidden
        )
        soft_halted_hidden = state.accumulated_hidden + weighted_remaining_hidden
        remaining_step_contribution = remaining_probabilities * (state.step_count + 1)
        ponder_loss = state.accumulated_ponder_cost + remaining_step_contribution
        return soft_halted_hidden, ponder_loss
