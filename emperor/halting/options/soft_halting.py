import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from emperor.base.layer import Layer, LayerStackConfig
from emperor.base.enums import LastLayerBiasOptions
from emperor.halting.options.base import HaltingBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.halting.config import HaltingConfig


@dataclass
class SoftHaltingState:
    step_count: int = field(
        metadata={
            "help": "Current step index, used to compute the expected number of steps for regularisation"
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
    accumulated_ponder_cost: Tensor = field(
        metadata={
            "help": "Running sum of halt_prob * step across all steps; used to compute the expected computation depth"
        },
    )
    continuation_probability: Tensor = field(
        metadata={
            "help": "Masked probability of not having halted at this step; used by compute_output to blend the final representation"
        },
    )


class SoftHalting(HaltingBase[SoftHaltingState]):
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

    def __compute_gate_logits(self, model_hidden_state: Tensor) -> Tensor:
        logits = self._gate(model_hidden_state)
        return F.log_softmax(logits, dim=-1)

    def __compute_step_halting_probability(
        self,
        current_log_gates: Tensor,
        log_continuation: Tensor,
    ) -> tuple[Tensor, Tensor]:
        current_log_halting = log_continuation.unsqueeze(-1) + current_log_gates
        log_continuation, log_halting = torch.unbind(current_log_halting, dim=-1)
        halting_probability = torch.exp(log_halting)
        return halting_probability, log_continuation

    def __compute_act_loss(
        self,
        state: SoftHaltingState,
        continuation_probability: Tensor,
        pad_mask: Tensor,
    ) -> Tensor:
        act_loss = (
            state.accumulated_ponder_cost + continuation_probability * state.step_count
        ) * pad_mask
        return act_loss.sum() / pad_mask.sum()

    def __blend_attn_input(
        self,
        state: SoftHaltingState,
        current_hidden: Tensor,
        self_attn_input: Tensor,
        continuation_probability: Tensor,
    ) -> Tensor:
        halted_output = (
            state.accumulated_hidden
            + continuation_probability.unsqueeze(-1) * current_hidden
        ).type_as(self_attn_input)
        return torch.where(
            continuation_probability.unsqueeze(-1) < (1 - self.threshold),
            self_attn_input,
            halted_output,
        )

    def update_halting_state(
        self,
        previous_state: SoftHaltingState | None,
        model_hidden_state: Tensor,
        pad_mask: Tensor | None = None,
    ) -> SoftHaltingState:
        if previous_state is None:
            return self.__init_state(model_hidden_state, pad_mask)
        return self.__update_state(previous_state, model_hidden_state, pad_mask)

    def __init_state(
        self,
        model_hidden_state: Tensor,
        pad_mask: Tensor | None,
    ) -> SoftHaltingState:
        ones = torch.ones_like(model_hidden_state[..., 0])
        return SoftHaltingState(
            step_count=0,
            log_continuation=torch.zeros_like(model_hidden_state[..., 0]),
            accumulated_hidden=torch.zeros_like(model_hidden_state),
            accumulated_ponder_cost=torch.zeros_like(model_hidden_state[..., 0]),
            continuation_probability=ones if pad_mask is None else pad_mask,
        )

    def __update_state(
        self,
        previous_state: SoftHaltingState,
        model_hidden_state: Tensor,
        pad_mask: Tensor | None,
    ) -> SoftHaltingState:
        current_log_gates = self.__compute_gate_logits(model_hidden_state)
        halting_probability, log_continuation = self.__compute_step_halting_probability(
            current_log_gates, previous_state.log_continuation
        )
        continuation_probability = log_continuation.exp()
        continuation_probability = continuation_probability.masked_fill(
            continuation_probability < (1 - self.threshold), 0
        )
        if pad_mask is not None:
            continuation_probability = (
                continuation_probability * pad_mask
            ).contiguous()
        return SoftHaltingState(
            step_count=previous_state.step_count + 1,
            log_continuation=log_continuation,
            accumulated_hidden=previous_state.accumulated_hidden
            + halting_probability.unsqueeze(-1) * model_hidden_state,
            accumulated_ponder_cost=previous_state.accumulated_ponder_cost
            + previous_state.step_count * halting_probability,
            continuation_probability=continuation_probability,
        )

    def compute_output(
        self,
        state: SoftHaltingState,
        current_hidden: Tensor,
        self_attn_input: Tensor,
        pad_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if state.step_count == 0:
            return current_hidden, torch.tensor(0.0)
        self_attn_input = self.__blend_attn_input(
            state, current_hidden, self_attn_input, state.continuation_probability
        )
        act_loss = self.__compute_act_loss(
            state, state.continuation_probability, pad_mask
        )
        return self_attn_input, act_loss
