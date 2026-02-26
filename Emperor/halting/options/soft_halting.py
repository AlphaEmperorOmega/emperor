import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.halting.config import HaltingConfig


ACT_STATE = tuple[int, Tensor, Tensor, Tensor]


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

    def _compute_gate_logits(self, hidden: Tensor) -> Tensor:
        logits = self._gate(hidden)
        return F.log_softmax(logits, dim=-1)

    def _update_halting_probs(
        self,
        log_g: Tensor,
        log_never_halt: Tensor,
    ) -> tuple[Tensor, Tensor]:
        log_halt = log_never_halt[..., None] + log_g
        log_never_halt = log_halt[..., 0]
        p = torch.exp(log_halt[..., 1])
        return p, log_never_halt

    def __init_state(
        self,
        prev_h: Tensor,
        pad_mask: Tensor,
    ) -> tuple[ACT_STATE, Tensor]:
        log_never_halt = torch.zeros_like(prev_h[..., 0])
        acc_expect_depth = torch.zeros_like(prev_h[..., 0])
        acc_h = torch.zeros_like(prev_h)
        state: ACT_STATE = (0, log_never_halt, acc_h, acc_expect_depth)
        return state, pad_mask

    def __step_state(
        self,
        state: ACT_STATE,
        prev_h: Tensor,
        pad_mask: Tensor,
    ) -> tuple[ACT_STATE, Tensor]:
        step, log_never_halt, acc_h, acc_expect_depth = state
        log_g = self._compute_gate_logits(prev_h)
        p, log_never_halt = self._update_halting_probs(log_g, log_never_halt)
        acc_h = acc_h + p[..., None] * prev_h
        acc_expect_depth = acc_expect_depth + step * p
        p_never_halt = log_never_halt.exp()
        p_never_halt = (
            p_never_halt.masked_fill(p_never_halt < (1 - self.threshold), 0)
            * pad_mask
        )
        p_never_halt = p_never_halt.contiguous()
        new_state: ACT_STATE = (step + 1, log_never_halt, acc_h, acc_expect_depth)
        return new_state, p_never_halt

    def __compute_act_loss(
        self,
        state: ACT_STATE,
        p_never_halt: Tensor,
        pad_mask: Tensor,
    ) -> Tensor:
        step, _, _, acc_expect_depth = state
        act_loss = (acc_expect_depth + p_never_halt * step) * pad_mask
        return act_loss.sum() / pad_mask.sum()

    def __blend_attn_input(
        self,
        state: ACT_STATE,
        curr_h: Tensor,
        self_attn_input: Tensor,
        p_never_halt: Tensor,
    ) -> Tensor:
        _, _, acc_h, _ = state
        halted_output = (acc_h + p_never_halt[..., None] * curr_h).type_as(self_attn_input)
        return torch.where(
            p_never_halt[..., None] < (1 - self.threshold),
            self_attn_input,
            halted_output,
        )

    def forward(
        self,
        prev_state: ACT_STATE | None,
        prev_h: Tensor,
        pad_mask: Tensor,
    ) -> tuple[ACT_STATE, Tensor]:
        if prev_state is None:
            return self.__init_state(prev_h, pad_mask)
        return self.__step_state(prev_state, prev_h, pad_mask)

    def compute_output(
        self,
        state: ACT_STATE,
        curr_h: Tensor,
        self_attn_input: Tensor,
        p_never_halt: Tensor,
        pad_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        step = state[0]
        if step == 0:
            return curr_h, torch.tensor(0.0)
        self_attn_input = self.__blend_attn_input(state, curr_h, self_attn_input, p_never_halt)
        act_loss = self.__compute_act_loss(state, p_never_halt, pad_mask)
        return self_attn_input, act_loss
