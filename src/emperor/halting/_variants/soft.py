from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from emperor.halting._base import (
    ComputeStep,
    HaltingBase,
    HaltingComputation,
    HaltingStateBase,
)
from emperor.halting._config import HaltingHiddenStateModeOptions
from emperor.halting._validation import SoftHaltingValidator
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
    stop_requested: bool = False
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


@dataclass(frozen=True, kw_only=True)
class _SoftOwnerStep:
    previous_state: SoftHaltingState | None
    raw_hidden: Tensor
    valid_mask: Tensor
    update_mask: Tensor
    prepared: _SoftPreparedStep
    computation: HaltingComputation


class SoftHalting(HaltingBase[SoftHaltingState]):
    """SUT-compatible soft adaptive computation.

    In RAW mode with the canonical gate, the recurrence and ordering match the
    pinned SUT ``ACTWrapper``: step zero skips the gate, later steps gate the
    previous raw hidden, and computation receives the prior soft context.
    """

    VALIDATOR = SoftHaltingValidator
    DEFAULT_THRESHOLD = 0.999

    @classmethod
    def validate_resolved_config(cls, cfg) -> None:
        cls.VALIDATOR.validate_config(cfg)

    def __init__(
        self,
        cfg: "HaltingConfig | ModelConfig",
        overrides: "HaltingConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "halting_config", cfg)
        self.cfg: HaltingConfig = self._override_config(config, overrides)

        self.input_dim: int = self.cfg.input_dim
        self.threshold: float = (
            self.DEFAULT_THRESHOLD if self.cfg.threshold is None else self.cfg.threshold
        )
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
        self._initialize_gate_with_equal_logits(self._gate[-1].model)

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

    def run_step(
        self,
        previous_state: SoftHaltingState | None,
        raw_hidden: Tensor,
        compute_step: ComputeStep,
        *,
        valid_mask: Tensor | None = None,
        update_mask: Tensor | None = None,
    ) -> SoftHaltingState:
        owner_step = self.prepare_owner_step(
            previous_state,
            raw_hidden,
            valid_mask=valid_mask,
            update_mask=update_mask,
        )
        candidate = compute_step(owner_step.computation)
        return self.complete_owner_step(owner_step, candidate)

    def prepare_owner_step(
        self,
        previous_state: SoftHaltingState | None,
        raw_hidden: Tensor,
        *,
        valid_mask: Tensor | None = None,
        update_mask: Tensor | None = None,
    ) -> _SoftOwnerStep:
        self._validate_hidden(raw_hidden, "raw_hidden")
        valid_mask, update_mask = self._resolve_masks(
            previous_state,
            raw_hidden,
            valid_mask,
            update_mask,
        )
        if previous_state is None:
            prepared = self.__prepare_initial_step(
                raw_hidden,
                valid_mask,
                update_mask,
            )
        else:
            prepared = self.__prepare_later_step(
                previous_state,
                raw_hidden,
                update_mask,
            )

        computation_mask = update_mask & (
            prepared.continuation_probability >= (1.0 - self.threshold)
        )
        computation = HaltingComputation(
            raw_hidden=raw_hidden,
            context_hidden=self.__context_hidden(previous_state),
            continuation_probability=prepared.continuation_probability.masked_fill(
                ~update_mask,
                0.0,
            ),
            computation_mask=computation_mask,
        )
        return _SoftOwnerStep(
            previous_state=previous_state,
            raw_hidden=raw_hidden,
            valid_mask=valid_mask,
            update_mask=update_mask,
            prepared=prepared,
            computation=computation,
        )

    def complete_owner_step(
        self,
        owner_step: _SoftOwnerStep,
        candidate: Tensor,
    ) -> SoftHaltingState:
        self._validate_computed_hidden(candidate, owner_step.raw_hidden)
        candidate = self._preserve_uncomputed_hidden(
            candidate,
            owner_step.raw_hidden,
            owner_step.computation.computation_mask,
        )
        return self.__complete_step(
            owner_step.prepared,
            owner_step.previous_state,
            candidate,
            owner_step.update_mask,
        )

    def gather_owner_step_rows(
        self,
        owner_step: _SoftOwnerStep,
        row_indices: Tensor,
        previous_state: SoftHaltingState | None,
    ) -> _SoftOwnerStep:
        source_row_count = owner_step.raw_hidden.shape[0]

        def gather_tensor(value: Tensor | None) -> Tensor | None:
            if value is None or value.dim() == 0 or value.shape[0] != source_row_count:
                return value
            return value.index_select(0, row_indices)

        prepared_values = {
            field.name: gather_tensor(getattr(owner_step.prepared, field.name))
            for field in fields(owner_step.prepared)
        }
        computation = replace(
            owner_step.computation,
            raw_hidden=gather_tensor(owner_step.computation.raw_hidden),
            context_hidden=gather_tensor(owner_step.computation.context_hidden),
            continuation_probability=gather_tensor(
                owner_step.computation.continuation_probability
            ),
            computation_mask=gather_tensor(owner_step.computation.computation_mask),
        )
        return _SoftOwnerStep(
            previous_state=previous_state,
            raw_hidden=gather_tensor(owner_step.raw_hidden),
            valid_mask=gather_tensor(owner_step.valid_mask),
            update_mask=gather_tensor(owner_step.update_mask),
            prepared=_SoftPreparedStep(**prepared_values),
            computation=computation,
        )

    def restrict_owner_step_updates(
        self,
        owner_step: _SoftOwnerStep,
        retained_update_mask: Tensor,
    ) -> _SoftOwnerStep:
        retained_update_mask = owner_step.update_mask & retained_update_mask.bool()
        previous_state = owner_step.previous_state
        if previous_state is None:
            return owner_step

        def retain(updated: Tensor, previous: Tensor) -> Tensor:
            expanded_mask = retained_update_mask
            while expanded_mask.dim() < updated.dim():
                expanded_mask = expanded_mask.unsqueeze(-1)
            return torch.where(expanded_mask, updated, previous)

        prepared = replace(
            owner_step.prepared,
            accumulated_hidden=retain(
                owner_step.prepared.accumulated_hidden,
                previous_state.accumulated_hidden,
            ),
            continuation_probability=retain(
                owner_step.prepared.continuation_probability,
                previous_state.continuation_probability,
            ),
            halt_mask=retain(
                owner_step.prepared.halt_mask,
                previous_state.halt_mask,
            ),
            step_count=retain(
                owner_step.prepared.step_count,
                previous_state.step_count,
            ),
            log_continuation=retain(
                owner_step.prepared.log_continuation,
                previous_state.log_continuation,
            ),
            accumulated_ponder_cost=retain(
                owner_step.prepared.accumulated_ponder_cost,
                previous_state.accumulated_ponder_cost,
            ),
            halt_probability=torch.where(
                retained_update_mask,
                owner_step.prepared.halt_probability,
                torch.zeros_like(owner_step.prepared.halt_probability),
            ),
            advanced_mask=previous_state.advanced_mask | retained_update_mask,
        )
        computation = replace(
            owner_step.computation,
            continuation_probability=owner_step.computation.continuation_probability
            * retained_update_mask.to(
                owner_step.computation.continuation_probability.dtype
            ),
            computation_mask=owner_step.computation.computation_mask
            & retained_update_mask,
        )
        return replace(
            owner_step,
            update_mask=retained_update_mask,
            prepared=prepared,
            computation=computation,
        )

    def __context_hidden(
        self,
        previous_state: SoftHaltingState | None,
    ) -> Tensor | None:
        if previous_state is None:
            return None
        if self.hidden_state_mode == HaltingHiddenStateModeOptions.ACCUMULATED:
            return previous_state.accumulated_hidden
        return previous_state.output_hidden

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

    def finalize(
        self,
        state: SoftHaltingState,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._validate_hidden(current_hidden, "current_hidden")
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
        state.finalized = True
        return state.output_hidden, loss

    def owner_stop_mask(self, state: SoftHaltingState) -> Tensor:
        # Soft halting freezes its own computation rows but retains the
        # owner's configured fixed-depth schedule.
        return torch.zeros_like(state.halt_mask, dtype=torch.bool)
