from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from emperor.halting._interface import HaltingInterface
from emperor.halting._validation import StickBreakingValidator
from emperor.nn import Module

StateT = TypeVar("StateT")


@dataclass(frozen=True, kw_only=True)
class HaltingComputation:
    """Strategy-neutral inputs for one owner computation step."""

    raw_hidden: Tensor
    context_hidden: Tensor | None
    continuation_probability: Tensor
    computation_mask: Tensor


ComputeStep = Callable[[HaltingComputation], Tensor]


@dataclass
class HaltingStateBase:
    halt_mask: Tensor | None = field(default=None, init=False)
    output_hidden: Tensor = field(init=False, repr=False)
    accumulated_hidden: Tensor = field(init=False, repr=False)
    continuation_probability: Tensor = field(init=False, repr=False)
    valid_mask: Tensor = field(init=False, repr=False)
    stop_requested: bool = field(default=False, init=False)
    finalized: bool = field(default=False, init=False)
    advanced_mask: Tensor = field(init=False, repr=False)
    step_indices: Tensor = field(init=False, repr=False)

    def __init__(
        self,
        *,
        output_hidden: Tensor,
        accumulated_hidden: Tensor,
        continuation_probability: Tensor,
        halt_mask: Tensor,
        valid_mask: Tensor,
        stop_requested: bool,
    ) -> None:
        self.output_hidden = output_hidden
        self.accumulated_hidden = accumulated_hidden
        self.continuation_probability = continuation_probability
        self.halt_mask = halt_mask
        self.valid_mask = valid_mask
        self.stop_requested = stop_requested
        self.finalized = False
        self.advanced_mask = valid_mask.clone()
        self.step_indices = continuation_probability.new_zeros(
            continuation_probability.shape
        )


class HaltingBase(Module, HaltingInterface[StateT], Generic[StateT], ABC):
    VALIDATOR = StickBreakingValidator

    @classmethod
    def implements_halting_interface(cls) -> bool:
        """Whether the strategy concretely implements the halting interface."""

        return (
            issubclass(cls, HaltingInterface)
            and cls.update_halting_state is not HaltingBase.update_halting_state
            and cls.finalize_weighted_accumulation
            is not HaltingBase.finalize_weighted_accumulation
        )

    @classmethod
    def validate_resolved_config(cls, cfg) -> None:
        cls.VALIDATOR.validate_config(cfg)

    def run_step(
        self,
        previous_state: StateT | None,
        raw_hidden: Tensor,
        compute_step: ComputeStep,
        *,
        valid_mask: Tensor | None = None,
        update_mask: Tensor | None = None,
    ) -> StateT:
        """Transitional adapter retained for deferred strategy work.

        This is not the supported owner-facing interface. The default delegates
        to the StickBreaking contract without changing StickBreaking itself.
        """

        self._validate_hidden(raw_hidden, "raw_hidden")
        resolved_valid_mask, resolved_update_mask = self._resolve_masks(
            previous_state,
            raw_hidden,
            valid_mask,
            update_mask,
        )
        continuation_probability = self._legacy_continuation_probability(
            previous_state,
            raw_hidden,
            resolved_valid_mask,
        )
        computation = HaltingComputation(
            raw_hidden=raw_hidden,
            context_hidden=None,
            continuation_probability=(
                continuation_probability
                * resolved_update_mask.to(raw_hidden.dtype)
            ),
            computation_mask=resolved_update_mask,
        )
        candidate = compute_step(computation)
        self._validate_computed_hidden(candidate, raw_hidden)
        candidate = self._preserve_uncomputed_hidden(
            candidate,
            raw_hidden,
            resolved_update_mask,
        )
        state, _output_hidden = self.update_halting_state(
            previous_state,
            candidate,
        )
        self._normalize_legacy_state(
            state,
            previous_state,
            raw_hidden,
            candidate,
            resolved_valid_mask,
            resolved_update_mask,
        )
        return state

    def finalize(
        self,
        state: StateT,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Transitional adapter around the supported finalization method."""

        self._validate_hidden(current_hidden, "current_hidden")
        self.VALIDATOR.validate_tensor_shape(
            current_hidden,
            state.accumulated_hidden.shape,
            "current_hidden",
        )
        finalized_hidden, loss = self.finalize_weighted_accumulation(
            state,
            current_hidden,
        )
        active_domain = state.valid_mask & state.advanced_mask
        finalized_hidden = torch.where(
            active_domain.unsqueeze(-1),
            finalized_hidden,
            state.output_hidden,
        )
        loss = loss * active_domain.to(loss.dtype)
        state.finalized = True
        return finalized_hidden, loss

    def owner_stop_mask(self, state: StateT) -> Tensor:
        """Transitional scheduling helper for deferred strategy work."""
        return state.halt_mask.bool()

    def update_halting_state(
        self,
        previous_state: StateT | None,
        model_hidden_state: Tensor,
    ) -> tuple[StateT, Tensor]:
        """Update the strategy state from one completed model step."""
        raise NotImplementedError

    def finalize_weighted_accumulation(
        self,
        state: StateT,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Finalize the hidden representation and auxiliary loss."""
        raise NotImplementedError

    @staticmethod
    def _legacy_continuation_probability(
        previous_state: StateT | None,
        hidden: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if previous_state is None:
            return valid_mask.to(hidden.dtype)
        return previous_state.log_continuation.exp()

    def _normalize_legacy_state(
        self,
        state: StateT,
        previous_state: StateT | None,
        raw_hidden: Tensor,
        candidate: Tensor,
        valid_mask: Tensor,
        update_mask: Tensor,
    ) -> None:
        if previous_state is None:
            state.halt_mask = state.halt_mask & update_mask
            state.log_continuation = torch.where(
                update_mask,
                state.log_continuation,
                torch.zeros_like(state.log_continuation),
            )
            state.accumulated_hidden = torch.where(
                update_mask.unsqueeze(-1),
                state.accumulated_hidden,
                torch.zeros_like(state.accumulated_hidden),
            )
            state.output_hidden = torch.where(
                update_mask.unsqueeze(-1),
                state.output_hidden,
                raw_hidden,
            )
            state.accumulated_halt_probabilities = torch.where(
                update_mask,
                state.accumulated_halt_probabilities,
                torch.zeros_like(state.accumulated_halt_probabilities),
            )
            step_indices = candidate.new_zeros(candidate.shape[:-1])
            advanced_mask = update_mask
        else:
            state.halt_mask = torch.where(
                update_mask,
                state.halt_mask,
                previous_state.halt_mask,
            )
            state.log_continuation = torch.where(
                update_mask,
                state.log_continuation,
                previous_state.log_continuation,
            )
            state.accumulated_hidden = torch.where(
                update_mask.unsqueeze(-1),
                state.accumulated_hidden,
                previous_state.accumulated_hidden,
            )
            state.output_hidden = torch.where(
                update_mask.unsqueeze(-1),
                state.output_hidden,
                previous_state.output_hidden,
            )
            state.accumulated_halt_probabilities = torch.where(
                update_mask,
                state.accumulated_halt_probabilities,
                previous_state.accumulated_halt_probabilities,
            )
            state.accumulated_ponder_cost = torch.where(
                update_mask,
                state.accumulated_ponder_cost,
                previous_state.accumulated_ponder_cost,
            )
            step_indices = previous_state.step_indices + update_mask.to(
                candidate.dtype
            )
            advanced_mask = previous_state.advanced_mask | update_mask

        state.raw_hidden = torch.where(
            update_mask.unsqueeze(-1),
            candidate,
            raw_hidden,
        )
        state.continuation_probability = torch.where(
            update_mask,
            state.log_continuation.exp(),
            self._legacy_continuation_probability(
                previous_state,
                raw_hidden,
                valid_mask,
            ),
        )
        state.valid_mask = valid_mask
        state.stop_requested = bool((state.halt_mask | ~valid_mask).all().item())
        state.finalized = False
        state.advanced_mask = advanced_mask
        state.step_indices = step_indices

    def _resolve_masks(
        self,
        previous_state: HaltingStateBase | None,
        hidden: Tensor,
        valid_mask: Tensor | None,
        update_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        resolved_valid_mask = self._normalize_mask(
            valid_mask,
            hidden,
            "valid_mask",
            default=(
                previous_state.valid_mask
                if previous_state is not None
                else torch.ones(
                    hidden.shape[:-1],
                    dtype=torch.bool,
                    device=hidden.device,
                )
            ),
        )
        if previous_state is not None:
            previous_valid_mask = previous_state.valid_mask.to(hidden.device)
            if not torch.equal(resolved_valid_mask, previous_valid_mask):
                raise ValueError(
                    "valid_mask cannot change after the first halting step"
                )
        resolved_update_mask = self._normalize_mask(
            update_mask,
            hidden,
            "update_mask",
            default=resolved_valid_mask,
        )
        return resolved_valid_mask, resolved_update_mask & resolved_valid_mask

    @staticmethod
    def _normalize_mask(
        mask: Tensor | None,
        hidden: Tensor,
        field_name: str,
        *,
        default: Tensor,
    ) -> Tensor:
        if mask is None:
            return default.to(device=hidden.device, dtype=torch.bool)
        if not isinstance(mask, Tensor):
            raise TypeError(
                f"{field_name} must be a Tensor or None, "
                f"received {type(mask).__name__}"
            )
        expected_shape = hidden.shape[:-1]
        if mask.shape != expected_shape:
            raise ValueError(
                f"{field_name} must have shape {tuple(expected_shape)}, "
                f"received {tuple(mask.shape)}"
            )
        if mask.dtype == torch.bool:
            return mask.to(device=hidden.device)
        if torch.is_complex(mask):
            raise TypeError(
                f"{field_name} must use bool, integer, or floating dtype, "
                f"received {mask.dtype}"
            )
        if torch.is_floating_point(mask) and not torch.isfinite(mask).all().item():
            raise ValueError(f"{field_name} values must be finite and binary")
        binary_values = (mask == 0) | (mask == 1)
        if not binary_values.all().item():
            raise ValueError(f"{field_name} values must be binary (0 or 1)")
        return mask.to(device=hidden.device, dtype=torch.bool)

    def _validate_hidden(self, hidden: Tensor, field_name: str) -> None:
        self.VALIDATOR.validate_hidden_tensor(hidden, self.input_dim, field_name)

    def _validate_computed_hidden(
        self,
        candidate: Tensor,
        raw_hidden: Tensor,
    ) -> None:
        self._validate_hidden(candidate, "compute_step result")
        self.VALIDATOR.validate_tensor_shape(
            candidate,
            raw_hidden.shape,
            "compute_step result",
        )

    @staticmethod
    def _preserve_uncomputed_hidden(
        candidate: Tensor,
        raw_hidden: Tensor,
        computation_mask: Tensor,
    ) -> Tensor:
        return torch.where(computation_mask.unsqueeze(-1), candidate, raw_hidden)

    @staticmethod
    def _initialize_gate_with_equal_logits(gate: nn.Module) -> None:
        """Initialize a configurable gate to equal logits when it has parameters."""

        for parameter in gate.parameters():
            nn.init.zeros_(parameter)
