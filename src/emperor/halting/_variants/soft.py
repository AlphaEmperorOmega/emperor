from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from emperor.halting._base import HaltingBase, HaltingStateBase
from emperor.halting._config import HaltingHiddenStateModeOptions
from emperor.halting._validation import SoftHaltingValidator
from emperor.halting._variants._initialization import zero_gate_parameters
from emperor.layers import LayerStackConfig, LayerState

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.halting._config import HaltingConfig


@dataclass(kw_only=True)
class SoftHaltingState(HaltingStateBase):
    raw_hidden: Tensor
    output_hidden: Tensor
    accumulated_hidden: Tensor
    continuation_probability: Tensor
    halt_mask: Tensor
    valid_mask: Tensor
    step_count: Tensor
    log_continuation: Tensor
    accumulated_ponder_cost: Tensor
    halt_probability: Tensor
    gate_input: Tensor | None
    gate_logits: Tensor | None
    advanced_mask: Tensor


@dataclass(frozen=True, kw_only=True)
class _SoftPreparedStep:
    accumulated_hidden: Tensor
    continuation_probability: Tensor
    halt_mask: Tensor
    valid_mask: Tensor
    step_count: Tensor
    log_continuation: Tensor
    accumulated_ponder_cost: Tensor
    halt_probability: Tensor
    gate_input: Tensor | None
    gate_logits: Tensor | None
    advanced_mask: Tensor


class SoftHalting(HaltingBase[SoftHaltingState]):
    VALIDATOR = SoftHaltingValidator

    def __init__(
        self,
        cfg: "HaltingConfig | ModelConfig",
        overrides: "HaltingConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "halting_config", cfg)
        self.cfg: HaltingConfig = self._override_config(config, overrides)

        self.input_dim: int = self.cfg.input_dim
        self.threshold: float = self.cfg.threshold
        self.dropout_probability: float | None = self.cfg.dropout_probability
        self.hidden_state_mode: HaltingHiddenStateModeOptions = (
            self.cfg.hidden_state_mode
        )
        self.halting_gate_config: LayerStackConfig | None = self.cfg.halting_gate_config
        self.VALIDATOR.validate(self)

        self._gate = self.__build_gate()
        self.__initialize_output_projection()

    def __build_gate(self) -> nn.Module:
        dropout_probability = (
            0.0 if self.dropout_probability is None else self.dropout_probability
        )
        if self.halting_gate_config is None:
            return nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim, bias=True),
                nn.GELU(),
                nn.Dropout(dropout_probability),
                nn.Linear(self.input_dim, 2, bias=False),
            )
        overrides = type(self.halting_gate_config)(input_dim=self.input_dim)
        return self.halting_gate_config.build(overrides=overrides)

    def __initialize_output_projection(self) -> None:
        if isinstance(self._gate, nn.Sequential):
            nn.init.zeros_(self._gate[-1].weight)
            return
        zero_gate_parameters(self._gate[-1].model)

    def __compute_gate_logits(self, hidden: Tensor) -> Tensor:
        if isinstance(self._gate, nn.Sequential):
            return F.log_softmax(self._gate(hidden), dim=-1)

        original_shape = hidden.shape
        flat_hidden = hidden.reshape(-1, original_shape[-1])
        state = LayerState(hidden=flat_hidden)
        for layer in self._gate.layers[:-1]:
            state = layer(state)
        dropout_probability = (
            0.0 if self.dropout_probability is None else self.dropout_probability
        )
        state.hidden = F.dropout(
            state.hidden,
            p=dropout_probability,
            training=self.training,
        )
        logits = self._gate[-1](state).hidden
        return F.log_softmax(logits.reshape(*original_shape[:-1], 2), dim=-1)

    def update_halting_state(
        self,
        previous_state: SoftHaltingState | None,
        model_hidden_state: Tensor,
    ) -> tuple[SoftHaltingState, Tensor]:
        self.VALIDATOR.validate_hidden_tensor(
            model_hidden_state,
            self.input_dim,
        )
        update_mask = torch.ones(
            model_hidden_state.shape[:-1],
            dtype=torch.bool,
            device=model_hidden_state.device,
        )
        if previous_state is None:
            prepared = self.__prepare_initial_step(
                model_hidden_state,
                update_mask,
                update_mask,
            )
            state = self.__complete_step(
                prepared,
                None,
                model_hidden_state,
                update_mask,
            )
            return state, state.output_hidden

        self.VALIDATOR.validate_tensor_shape(
            model_hidden_state,
            previous_state.raw_hidden.shape,
            "model_hidden_state",
        )
        previously_advanced = previous_state.advanced_mask.bool()
        prepared = self.__prepare_later_step(
            previous_state,
            previous_state.raw_hidden,
            previously_advanced,
        )
        prepared = self.__mask_gate_diagnostics(
            prepared,
            previously_advanced,
        )
        computation_mask = ~previously_advanced | (
            prepared.continuation_probability >= (1.0 - self.threshold)
        )
        candidate = self.__preserve_uncomputed_hidden(
            model_hidden_state,
            previous_state.raw_hidden,
            computation_mask,
        )
        state = self.__complete_step(
            prepared,
            previous_state,
            candidate,
            update_mask,
        )
        return state, state.output_hidden

    def __prepare_initial_step(
        self,
        raw_hidden: Tensor,
        valid_mask: Tensor,
        update_mask: Tensor,
    ) -> _SoftPreparedStep:
        leading_shape = raw_hidden.shape[:-1]
        return _SoftPreparedStep(
            accumulated_hidden=torch.zeros_like(raw_hidden),
            continuation_probability=valid_mask.to(raw_hidden.dtype),
            halt_mask=torch.zeros_like(valid_mask),
            valid_mask=valid_mask,
            step_count=raw_hidden.new_zeros(leading_shape),
            log_continuation=raw_hidden.new_zeros(leading_shape),
            accumulated_ponder_cost=raw_hidden.new_zeros(leading_shape),
            halt_probability=raw_hidden.new_zeros(leading_shape),
            gate_input=None,
            gate_logits=None,
            advanced_mask=update_mask,
        )

    def __prepare_later_step(
        self,
        previous_state: SoftHaltingState,
        raw_hidden: Tensor,
        update_mask: Tensor,
    ) -> _SoftPreparedStep:
        gate_logits = self.__compute_gate_logits(raw_hidden)
        log_probability_masses = (
            previous_state.log_continuation.unsqueeze(-1) + gate_logits
        )
        candidate_log_continuation, log_halt_probability = torch.unbind(
            log_probability_masses,
            dim=-1,
        )
        halt_probability = log_halt_probability.exp().masked_fill(~update_mask, 0.0)
        log_continuation = torch.where(
            update_mask,
            candidate_log_continuation,
            previous_state.log_continuation,
        )
        continuation_probability = log_continuation.exp().masked_fill(
            log_continuation.exp() < (1.0 - self.threshold),
            0.0,
        )
        continuation_probability = torch.where(
            update_mask,
            continuation_probability,
            previous_state.continuation_probability,
        ).contiguous()
        updated_step_count = torch.where(
            update_mask,
            previous_state.step_count + 1,
            previous_state.step_count,
        )
        return _SoftPreparedStep(
            accumulated_hidden=(
                previous_state.accumulated_hidden
                + halt_probability.unsqueeze(-1) * raw_hidden
            ),
            continuation_probability=continuation_probability,
            halt_mask=(
                previous_state.halt_mask
                | (update_mask & (continuation_probability < (1.0 - self.threshold)))
            ),
            valid_mask=previous_state.valid_mask,
            step_count=updated_step_count,
            log_continuation=log_continuation,
            accumulated_ponder_cost=(
                previous_state.accumulated_ponder_cost
                + previous_state.step_count * halt_probability
            ),
            halt_probability=halt_probability,
            gate_input=raw_hidden,
            gate_logits=gate_logits,
            advanced_mask=previous_state.advanced_mask | update_mask,
        )

    @staticmethod
    def __mask_gate_diagnostics(
        prepared: _SoftPreparedStep,
        gate_mask: Tensor,
    ) -> _SoftPreparedStep:
        def mask_rows(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            expanded_gate_mask = gate_mask
            while expanded_gate_mask.dim() < value.dim():
                expanded_gate_mask = expanded_gate_mask.unsqueeze(-1)
            return value.masked_fill(~expanded_gate_mask, 0.0)

        return replace(
            prepared,
            gate_input=mask_rows(prepared.gate_input),
            gate_logits=mask_rows(prepared.gate_logits),
        )

    def __complete_step(
        self,
        prepared: _SoftPreparedStep,
        previous_state: SoftHaltingState | None,
        candidate: Tensor,
        update_mask: Tensor,
    ) -> SoftHaltingState:
        if previous_state is None:
            output_hidden = candidate
        else:
            blended_hidden = (
                prepared.accumulated_hidden
                + prepared.continuation_probability.unsqueeze(-1) * candidate
            ).type_as(previous_state.output_hidden)
            frozen_mask = prepared.continuation_probability < (1.0 - self.threshold)
            output_hidden = torch.where(
                frozen_mask.unsqueeze(-1),
                previous_state.output_hidden,
                blended_hidden,
            )
            output_hidden = torch.where(
                update_mask.unsqueeze(-1),
                output_hidden,
                previous_state.output_hidden,
            )
        return SoftHaltingState(
            raw_hidden=candidate,
            output_hidden=output_hidden,
            accumulated_hidden=prepared.accumulated_hidden,
            continuation_probability=prepared.continuation_probability,
            halt_mask=prepared.halt_mask,
            valid_mask=prepared.valid_mask,
            step_count=prepared.step_count,
            log_continuation=prepared.log_continuation,
            accumulated_ponder_cost=prepared.accumulated_ponder_cost,
            halt_probability=prepared.halt_probability,
            gate_input=prepared.gate_input,
            gate_logits=prepared.gate_logits,
            advanced_mask=prepared.advanced_mask,
        )

    def finalize_weighted_accumulation(
        self,
        state: SoftHaltingState,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_hidden_tensor(
            current_hidden,
            self.input_dim,
            "current_hidden",
        )
        self.VALIDATOR.validate_tensor_shape(
            current_hidden,
            state.raw_hidden.shape,
            "current_hidden",
        )
        valid_weight = state.valid_mask & state.advanced_mask
        loss_by_position = (
            state.accumulated_ponder_cost
            + state.continuation_probability * state.step_count
        ) * valid_weight
        loss = loss_by_position.sum() / valid_weight.sum().clamp_min(1)
        return state.output_hidden, loss

    @staticmethod
    def __preserve_uncomputed_hidden(
        candidate: Tensor,
        previous_hidden: Tensor,
        computation_mask: Tensor,
    ) -> Tensor:
        return torch.where(
            computation_mask.unsqueeze(-1),
            candidate,
            previous_hidden,
        )
