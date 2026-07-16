from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.halting._base import HaltingBase, HaltingStateBase


class HaltingUsageTracker(Module):
    """Records adaptive-compute statistics for a single halting module.

    Buffers hold the most recent forward's survival curve (fraction of tokens
    still computing after each recurrence step) and scalar summaries (ponder
    depth, halted fraction, ponder loss). Survival is staged per step and
    committed on ``record_final``; its length equals the steps actually run, so
    it varies between forwards when halting exits early.
    """

    _DYNAMIC_BUFFER_NAMES = ("last_survival", "last_ponder_cost")

    def __init__(self):
        super().__init__()
        self.register_buffer("last_survival", torch.zeros(0))
        self.register_buffer("last_ponder_cost_mean", torch.zeros(()))
        self.register_buffer("last_ponder_cost_std", torch.zeros(()))
        self.register_buffer("last_ponder_cost", torch.zeros(0))
        self.register_buffer("last_step_count", torch.zeros(()))
        self.register_buffer("last_halted_fraction", torch.zeros(()))
        self.register_buffer("last_accumulated_halt_prob_mean", torch.zeros(()))
        self.register_buffer("last_remaining_mass_mean", torch.zeros(()))
        self.register_buffer("last_ponder_loss", torch.zeros(()))
        self._survival_stage: list[Tensor] = []

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for buffer_name in self._DYNAMIC_BUFFER_NAMES:
            saved_buffer = state_dict.get(f"{prefix}{buffer_name}")
            current_buffer = getattr(self, buffer_name)
            if saved_buffer is not None and saved_buffer.shape != current_buffer.shape:
                setattr(
                    self,
                    buffer_name,
                    current_buffer.new_empty(saved_buffer.shape),
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def begin_forward(self) -> None:
        self._survival_stage = []

    def record_step(self, halting_state: "HaltingStateBase") -> None:
        self._survival_stage.append(self.__compute_alive_fraction(halting_state))

    def record_final(
        self,
        ponder_loss: Tensor | None,
        halting_state: "HaltingStateBase",
    ) -> None:
        self.__commit_survival()
        self.__record_final_scalars(ponder_loss, halting_state)

    def __compute_alive_fraction(self, halting_state: "HaltingStateBase") -> Tensor:
        halt_mask = halting_state.halt_mask
        if halt_mask is not None:
            return (~halt_mask).float().mean().detach()
        continuation = getattr(halting_state, "continuation_probability", None)
        if continuation is not None:
            return continuation.detach().float().clamp(0.0, 1.0).mean()
        return self.last_survival.new_ones(())

    def __commit_survival(self) -> None:
        if self._survival_stage:
            self.last_survival = torch.stack(
                [
                    value.to(self.last_survival).reshape(())
                    for value in self._survival_stage
                ]
            )
        else:
            self.last_survival = self.last_survival.new_zeros(0)

    def __record_final_scalars(
        self,
        ponder_loss: Tensor | None,
        halting_state: "HaltingStateBase",
    ) -> None:
        self.last_step_count.fill_(float(len(self._survival_stage)))

        ponder_cost = getattr(halting_state, "accumulated_ponder_cost", None)
        if ponder_cost is not None:
            cost = ponder_cost.detach().float()
            self.last_ponder_cost = cost.reshape(-1)
            self.last_ponder_cost_mean.copy_(cost.mean())
            self.last_ponder_cost_std.copy_(cost.std(unbiased=False))

        halt_mask = halting_state.halt_mask
        if halt_mask is not None:
            self.last_halted_fraction.copy_(halt_mask.float().mean())

        accumulated = getattr(halting_state, "accumulated_halt_probabilities", None)
        if accumulated is not None:
            acc = accumulated.detach().float()
            self.last_accumulated_halt_prob_mean.copy_(acc.mean())
            remaining = 1.0 - acc
            if halt_mask is not None:
                remaining = remaining.masked_fill(halt_mask, 0.0)
            self.last_remaining_mass_mean.copy_(remaining.mean())

        if ponder_loss is not None:
            self.last_ponder_loss.copy_(ponder_loss.detach().float().mean())

    def reset(self) -> None:
        self.last_survival = self.last_survival.new_zeros(0)
        self.last_ponder_cost = self.last_ponder_cost.new_zeros(0)
        self.last_ponder_cost_mean.zero_()
        self.last_ponder_cost_std.zero_()
        self.last_step_count.zero_()
        self.last_halted_fraction.zero_()
        self.last_accumulated_halt_prob_mean.zero_()
        self.last_remaining_mass_mean.zero_()
        self.last_ponder_loss.zero_()
        self._survival_stage = []


@dataclass
class _HaltingAttachment:
    halting_model: "HaltingBase"


class HaltingUsageTrackerManager:
    """Attaches halting trackers without modifying any traced code.

    Capture is installed at runtime and removed on detach, mirroring how the
    linear monitor installs and removes forward hooks. Strategy methods are
    plain calls rather than ``__call__``, so the manager wraps
    ``update_halting_state`` and the strategy's completion method on the
    halting module instance. A fresh forward is detected from
    ``previous_state is None`` (the first halting step of any owner), so no
    owner-specific hook or ``max_steps`` is required.
    """

    TRACKER_MODULE_NAME = "_usage_tracker"
    COMPLETION_METHOD_NAMES = (
        "finalize_weighted_accumulation",
        "compute_output",
    )
    WRAPPED_METHOD_NAMES = ("update_halting_state", *COMPLETION_METHOD_NAMES)

    def __init__(self):
        self._attachments: list[_HaltingAttachment] = []

    def attach(self, halting_model: "HaltingBase") -> HaltingUsageTracker:
        existing_tracker = getattr(halting_model, self.TRACKER_MODULE_NAME, None)
        if existing_tracker is not None:
            return existing_tracker

        completion_method_name = self.__completion_method_name(halting_model)
        tracker = HaltingUsageTracker()
        try:
            self.__wrap_halting_methods(
                halting_model,
                tracker,
                completion_method_name,
            )
            halting_model.add_module(self.TRACKER_MODULE_NAME, tracker)
        except Exception:
            self.__restore_halting_methods(halting_model)
            raise
        self._attachments.append(_HaltingAttachment(halting_model))
        return tracker

    def detach(self, halting_model: "HaltingBase") -> None:
        attachment = next(
            (a for a in self._attachments if a.halting_model is halting_model),
            None,
        )
        if attachment is None:
            return
        self.__restore_halting_methods(halting_model)
        if self.TRACKER_MODULE_NAME in halting_model._modules:
            del halting_model._modules[self.TRACKER_MODULE_NAME]
        self._attachments.remove(attachment)

    def __wrap_halting_methods(
        self,
        halting_model: "HaltingBase",
        tracker: HaltingUsageTracker,
        completion_method_name: str,
    ) -> None:
        original_update = halting_model.update_halting_state
        original_completion = getattr(halting_model, completion_method_name)

        @wraps(original_update)
        def update_halting_state(previous_state, *args, **kwargs):
            if previous_state is None:
                tracker.begin_forward()
            state, hidden = original_update(previous_state, *args, **kwargs)
            tracker.record_step(state)
            return state, hidden

        @wraps(original_completion)
        def complete_halting(state, *args, **kwargs):
            hidden, ponder_loss = original_completion(state, *args, **kwargs)
            tracker.record_final(ponder_loss, state)
            return hidden, ponder_loss

        halting_model.update_halting_state = update_halting_state
        setattr(halting_model, completion_method_name, complete_halting)

    def __completion_method_name(self, halting_model: "HaltingBase") -> str:
        for method_name in self.COMPLETION_METHOD_NAMES:
            if callable(getattr(halting_model, method_name, None)):
                return method_name
        methods = ", ".join(self.COMPLETION_METHOD_NAMES)
        raise TypeError(f"{type(halting_model).__name__} must define one of: {methods}")

    def __restore_halting_methods(self, halting_model: "HaltingBase") -> None:
        for method_name in self.WRAPPED_METHOD_NAMES:
            if method_name in halting_model.__dict__:
                del halting_model.__dict__[method_name]
