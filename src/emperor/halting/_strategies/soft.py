from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from emperor.halting._base import HaltingBase, HaltingStateBase
from emperor.halting._config import HaltingHiddenStateModeOptions
from emperor.halting._strategies._initialization import zero_gate_parameters
from emperor.layers import LayerStack, LayerStackConfig, LayerState

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.halting._config import HaltingConfig


@dataclass
class SoftHaltingState(HaltingStateBase):
    step_count: int = field(
        metadata={
            "help": (
                "Current step index, used to compute the expected number of "
                "steps for regularisation"
            )
        },
    )
    log_continuation: Tensor = field(
        metadata={
            "help": (
                "Log of the remaining stick length after all breaks so far; "
                "the cumulative log probability of not yet having halted"
            )
        },
    )
    accumulated_hidden: Tensor = field(
        metadata={
            "help": (
                "Weighted sum of hidden states accumulated so far, where each "
                "step contributes proportionally to its halt probability"
            )
        },
    )
    accumulated_ponder_cost: Tensor = field(
        metadata={
            "help": (
                "Running sum of halt_prob * step across all steps; used to "
                "compute the expected computation depth"
            )
        },
    )
    continuation_probability: Tensor = field(
        metadata={
            "help": (
                "Masked probability of not having halted at this step; used "
                "by compute_output to blend the final representation"
            )
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
        self.cfg: HaltingConfig = self._override_config(config, overrides)
        self.VALIDATOR.validate(self)

        self.input_dim: int = self.cfg.input_dim
        self.threshold: float = self.cfg.threshold
        self.dropout_probability: float | None = self.cfg.dropout_probability
        self.hidden_state_mode: HaltingHiddenStateModeOptions = (
            self.cfg.hidden_state_mode
        )
        self.halting_gate_config: LayerStackConfig = self.cfg.halting_gate_config

        dropout_probability = (
            0.0 if self.dropout_probability is None else self.dropout_probability
        )
        self.dropout_probability_module = nn.Dropout(dropout_probability)
        self._gate: LayerStack = self.__build_gate()
        self.__init_gate_weights()

    def __build_gate(self) -> LayerStack:
        overrides = type(self.halting_gate_config)(
            input_dim=self.input_dim,
        )
        return self.halting_gate_config.build(overrides=overrides)  # type: ignore[return-value]

    def __init_gate_weights(self) -> None:
        zero_gate_parameters(self._gate[-1].model)

    def __compute_gate_logits(self, model_hidden_state: Tensor) -> Tensor:
        original_shape = model_hidden_state.shape
        flat = model_hidden_state.reshape(-1, original_shape[-1])
        state = LayerState(hidden=flat)
        for layer in self._gate.layers[:-1]:
            state = layer(state)
        state.hidden = self.dropout_probability_module(state.hidden)
        logits = self._gate[-1](state).hidden
        logits = logits.reshape(*original_shape[:-1], 2)
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
        return act_loss.sum() / pad_mask.sum().clamp_min(1)

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
    ) -> tuple[SoftHaltingState, Tensor]:
        self.VALIDATOR.validate_hidden_tensor(
            model_hidden_state,
            self.input_dim,
        )
        self.VALIDATOR.validate_pad_mask(pad_mask, model_hidden_state)
        if pad_mask is not None:
            pad_mask = pad_mask.to(model_hidden_state)
        if previous_state is None:
            state = self.__init_state(model_hidden_state, pad_mask)
        else:
            state = self.__update_state(previous_state, model_hidden_state, pad_mask)
        if self.hidden_state_mode == HaltingHiddenStateModeOptions.ACCUMULATED:
            return state, state.accumulated_hidden
        return state, model_hidden_state

    def __init_state(
        self,
        model_hidden_state: Tensor,
        pad_mask: Tensor | None,
    ) -> SoftHaltingState:
        leading_shape = model_hidden_state.shape[:-1]
        ones = model_hidden_state.new_ones(leading_shape)
        return SoftHaltingState(
            step_count=0,
            log_continuation=model_hidden_state.new_zeros(leading_shape),
            accumulated_hidden=torch.zeros_like(model_hidden_state),
            accumulated_ponder_cost=model_hidden_state.new_zeros(leading_shape),
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
        self.VALIDATOR.validate_hidden_tensor(
            current_hidden,
            self.input_dim,
            "current_hidden",
        )
        self.VALIDATOR.validate_hidden_tensor(
            self_attn_input,
            self.input_dim,
            "self_attn_input",
        )
        expected_shape = state.accumulated_hidden.shape
        self.VALIDATOR.validate_tensor_shape(
            current_hidden,
            expected_shape,
            "current_hidden",
        )
        self.VALIDATOR.validate_tensor_shape(
            self_attn_input,
            expected_shape,
            "self_attn_input",
        )
        self.VALIDATOR.validate_pad_mask(
            pad_mask,
            current_hidden,
            required_by="compute_output",
        )
        pad_mask = pad_mask.to(current_hidden)
        if state.step_count == 0:
            return current_hidden, current_hidden.new_zeros(())
        self_attn_input = self.__blend_attn_input(
            state, current_hidden, self_attn_input, state.continuation_probability
        )
        act_loss = self.__compute_act_loss(
            state, state.continuation_probability, pad_mask
        )
        return self_attn_input, act_loss
