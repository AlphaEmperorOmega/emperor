"""Frozen, network-free oracle for SUT's soft ACT wrapper.

This is a direct transcription of ``ACTWrapper`` from SUT commit
``192cbd3ed567e45e6ec6d33eb1ea9c3b331bc2b0``.  The source of truth is:

https://github.com/shawntan/SUT/blob/192cbd3ed567e45e6ec6d33eb1ea9c3b331bc2b0/halting.py#L74-L131

The oracle intentionally has no dependency on Emperor.  It keeps SUT's call
ordering visible so differential tests can detect a locally self-consistent,
but upstream-incompatible, recurrence.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class SutActState:
    step_count: Tensor
    log_continuation: Tensor
    accumulated_hidden: Tensor
    accumulated_expected_depth: Tensor
    output_hidden: Tensor


@dataclass
class SutActStep:
    state: SutActState
    raw_hidden: Tensor
    gate_input: Tensor | None
    gate_logits: Tensor | None
    halt_mass: Tensor
    continuation: Tensor
    computation_mask: Tensor
    loss: Tensor


SutComputeStep = Callable[[Tensor, Tensor | None, Tensor, Tensor], Tensor]


class SutActOracle:
    """Execute the pinned SUT ``ACTWrapper`` recurrence one step at a time."""

    def __init__(self, gate: nn.Module, threshold: float) -> None:
        self.gate = gate
        self.threshold = threshold

    def run_step(
        self,
        previous_state: SutActState | None,
        previous_raw_hidden: Tensor,
        valid_mask: Tensor,
        compute_step: SutComputeStep,
    ) -> SutActStep:
        valid_weight = valid_mask.to(previous_raw_hidden)
        leading_shape = previous_raw_hidden.shape[:-1]
        gate_input = None
        gate_logits = None
        halt_mass = previous_raw_hidden.new_zeros(leading_shape)

        if previous_state is None:
            step_count = previous_raw_hidden.new_zeros(leading_shape)
            log_continuation = previous_raw_hidden.new_zeros(leading_shape)
            accumulated_hidden = torch.zeros_like(previous_raw_hidden)
            accumulated_expected_depth = previous_raw_hidden.new_zeros(leading_shape)
            continuation = valid_weight
            context_hidden = None
        else:
            gate_input = previous_raw_hidden
            gate_logits = torch.log_softmax(self.gate(gate_input), dim=-1)
            log_halt = previous_state.log_continuation.unsqueeze(-1) + gate_logits
            candidate_log_continuation, log_halt_mass = torch.unbind(
                log_halt,
                dim=-1,
            )
            halt_mass = log_halt_mass.exp().masked_fill(~valid_mask, 0.0)
            log_continuation = torch.where(
                valid_mask,
                candidate_log_continuation,
                previous_state.log_continuation,
            )
            accumulated_hidden = (
                previous_state.accumulated_hidden
                + halt_mass.unsqueeze(-1) * previous_raw_hidden
            )
            accumulated_expected_depth = (
                previous_state.accumulated_expected_depth
                + previous_state.step_count * halt_mass
            )
            continuation = log_continuation.exp()
            continuation = continuation.masked_fill(
                continuation < (1.0 - self.threshold),
                0.0,
            )
            continuation = (continuation * valid_weight).contiguous()
            step_count = previous_state.step_count + valid_weight
            context_hidden = previous_state.output_hidden

        computation_mask = valid_mask & (continuation >= (1.0 - self.threshold))
        current_raw_hidden = compute_step(
            previous_raw_hidden,
            context_hidden,
            continuation,
            computation_mask,
        )
        current_raw_hidden = torch.where(
            computation_mask.unsqueeze(-1),
            current_raw_hidden,
            previous_raw_hidden,
        )

        if previous_state is None:
            output_hidden = current_raw_hidden
            loss = current_raw_hidden.new_zeros(())
        else:
            blended_hidden = (
                accumulated_hidden + continuation.unsqueeze(-1) * current_raw_hidden
            ).type_as(previous_state.output_hidden)
            output_hidden = torch.where(
                continuation.unsqueeze(-1) < (1.0 - self.threshold),
                previous_state.output_hidden,
                blended_hidden,
            )
            output_hidden = torch.where(
                valid_mask.unsqueeze(-1),
                output_hidden,
                previous_state.output_hidden,
            )
            loss_by_position = (
                accumulated_expected_depth + continuation * step_count
            ) * valid_weight
            loss = loss_by_position.sum() / valid_weight.sum()

        state = SutActState(
            step_count=step_count,
            log_continuation=log_continuation,
            accumulated_hidden=accumulated_hidden,
            accumulated_expected_depth=accumulated_expected_depth,
            output_hidden=output_hidden,
        )
        return SutActStep(
            state=state,
            raw_hidden=current_raw_hidden,
            gate_input=gate_input,
            gate_logits=gate_logits,
            halt_mass=halt_mass,
            continuation=continuation,
            computation_mask=computation_mask,
            loss=loss,
        )
