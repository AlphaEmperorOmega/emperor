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
class SoftHaltingState:
    step: int = field(
        metadata={
            "help": "Current step index, used to compute the expected number of steps for regularisation"
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
    accumulated_expected_step: Tensor = field(
        metadata={
            "help": "Running sum of halt_prob * step across all steps; used to compute the expected computation depth"
        },
    )


class SoftHalting(Module):
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
        self.halting_dropout: float = self.cfg.halting_dropout

        self._gate: nn.Sequential = self.__build_gate()
        self.__init_gate_weights()

    def __build_gate(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.GELU(),
            nn.Dropout(self.halting_dropout),
            nn.Linear(self.input_dim, 2, bias=False),
        )

    def __init_gate_weights(self) -> None:
        nn.init.zeros_(self._gate[-1].weight)  # type: ignore[union-attr]

    def __compute_gate_logits(self, previous_output: Tensor) -> Tensor:
        logits = self._gate(previous_output)
        return F.log_softmax(logits, dim=-1)

    def __update_halting_probs(
        self,
        current_log_gates: Tensor,
        log_never_halt: Tensor,
    ) -> tuple[Tensor, Tensor]:
        log_halt = log_never_halt[..., None] + current_log_gates
        log_never_halt = log_halt[..., 0]
        halting_probability = torch.exp(log_halt[..., 1])
        return halting_probability, log_never_halt

    def __compute_act_loss(
        self,
        state: SoftHaltingState,
        p_never_halt: Tensor,
        pad_mask: Tensor,
    ) -> Tensor:
        act_loss = (
            state.accumulated_expected_step + p_never_halt * state.step
        ) * pad_mask
        return act_loss.sum() / pad_mask.sum()

    def __blend_attn_input(
        self,
        state: SoftHaltingState,
        current_hidden: Tensor,
        self_attn_input: Tensor,
        p_never_halt: Tensor,
    ) -> Tensor:
        halted_output = (
            state.accumulated_hidden + p_never_halt[..., None] * current_hidden
        ).type_as(self_attn_input)
        return torch.where(
            p_never_halt[..., None] < (1 - self.threshold),
            self_attn_input,
            halted_output,
        )

    def forward(
        self,
        previous_state: SoftHaltingState | None,
        previous_output: Tensor,
        pad_mask: Tensor,
    ) -> tuple[SoftHaltingState, Tensor]:
        if previous_state is None:
            return self.__init_state(previous_output, pad_mask)
        return self.__update_state(previous_state, previous_output, pad_mask)

    def __init_state(
        self,
        previous_output: Tensor,
        pad_mask: Tensor,
    ) -> tuple[SoftHaltingState, Tensor]:
        state = SoftHaltingState(
            step=0,
            log_never_halt=torch.zeros_like(previous_output[..., 0]),
            accumulated_hidden=torch.zeros_like(previous_output),
            accumulated_expected_step=torch.zeros_like(previous_output[..., 0]),
        )
        return state, pad_mask

    def __update_state(
        self,
        previous_state: SoftHaltingState,
        previous_output: Tensor,
        pad_mask: Tensor,
    ) -> tuple[SoftHaltingState, Tensor]:
        current_log_gates = self.__compute_gate_logits(previous_output)
        halting_probability, log_never_halt = self.__update_halting_probs(
            current_log_gates, previous_state.log_never_halt
        )
        p_never_halt = log_never_halt.exp()
        p_never_halt = (
            p_never_halt.masked_fill(p_never_halt < (1 - self.threshold), 0) * pad_mask
        ).contiguous()
        state = SoftHaltingState(
            step=previous_state.step + 1,
            log_never_halt=log_never_halt,
            accumulated_hidden=previous_state.accumulated_hidden
            + halting_probability[..., None] * previous_output,
            accumulated_expected_step=previous_state.accumulated_expected_step
            + previous_state.step * halting_probability,
        )
        return state, p_never_halt

    def compute_output(
        self,
        state: SoftHaltingState,
        current_hidden: Tensor,
        self_attn_input: Tensor,
        p_never_halt: Tensor,
        pad_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if state.step == 0:
            return current_hidden, torch.tensor(0.0)
        self_attn_input = self.__blend_attn_input(
            state, current_hidden, self_attn_input, p_never_halt
        )
        act_loss = self.__compute_act_loss(state, p_never_halt, pad_mask)
        return self_attn_input, act_loss
