import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.halting.config import HaltingConfig


STICK_BREAKING_STATE = tuple[Tensor, Tensor, Tensor, Tensor, int, Tensor]


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
        nn.init.zeros_(self._gate[-1].weight)  # type: ignore[union-attr]

    def _compute_gate_logits(self, hidden: Tensor) -> Tensor:
        logits = self._gate(hidden)
        if self.training:
            logits = logits + torch.randn_like(logits)
        return F.log_softmax(logits, dim=-1)

    def __init_state(
        self,
        curr_log_g: Tensor,
        prev_out: Tensor,
    ) -> tuple[STICK_BREAKING_STATE, Tensor]:
        g = torch.exp(curr_log_g[..., 1])
        halt_mask = g >= self.threshold
        state: STICK_BREAKING_STATE = (
            halt_mask,
            curr_log_g[..., 0],
            g[..., None] * prev_out,
            g,
            0,
            torch.tensor(0.0),
        )
        return state, halt_mask

    def __step_state(
        self,
        state: STICK_BREAKING_STATE,
        curr_log_g: Tensor,
        prev_out: Tensor,
    ) -> tuple[STICK_BREAKING_STATE, Tensor]:
        prev_halt_mask, prev_log_never_halt, prev_acc_h, prev_acc_g, prev_step, prev_acc_expstep = state
        step = prev_step + 1
        curr_log_halt = prev_log_never_halt[..., None] + curr_log_g
        g = torch.exp(curr_log_halt[..., 1])
        g = g.masked_fill(prev_halt_mask, 0.0)
        curr_acc_g = prev_acc_g + g
        halt_mask = curr_acc_g >= self.threshold
        new_state: STICK_BREAKING_STATE = (
            halt_mask,
            curr_log_halt[..., 0],
            prev_acc_h + g[..., None] * prev_out,
            curr_acc_g,
            step,
            prev_acc_expstep + g * step,
        )
        return new_state, halt_mask

    def forward(
        self,
        prev_state: STICK_BREAKING_STATE | None,
        prev_out: Tensor,
    ) -> tuple[STICK_BREAKING_STATE, Tensor]:
        curr_log_g = self._compute_gate_logits(prev_out)
        if prev_state is None:
            return self.__init_state(curr_log_g, prev_out)
        return self.__step_state(prev_state, curr_log_g, prev_out)

    def halt_gating(
        self,
        state: STICK_BREAKING_STATE,
        curr_h: Tensor,
    ) -> tuple[Tensor, Tensor]:
        halt_mask, _, curr_acc_h, curr_acc_g, step, curr_expstep = state
        soft_halted_h = curr_acc_h + (1 - curr_acc_g)[..., None] * curr_h
        expstep = curr_expstep + (step + 1) * (1 - curr_acc_g)
        if halt_mask.any():
            soft_halted_h.masked_scatter_(halt_mask[..., None], curr_acc_h[halt_mask])
        return soft_halted_h, expstep
