"""Capture halting usage without monitoring-framework concerns."""

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.halting._base import HaltingBase, HaltingStateBase


class HaltingUsageTracker(Module):

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
        valid_mask = getattr(halting_state, "valid_mask", None)
        if valid_mask is not None:
            valid_count = valid_mask.sum()
            if not bool(valid_count.item()):
                reference = getattr(
                    halting_state,
                    "continuation_probability",
                    self.last_survival,
                )
                return reference.detach().new_zeros(())
            if halt_mask is not None:
                alive = valid_mask & ~halt_mask
                return (alive.float().sum() / valid_count).detach()
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

        valid_mask = getattr(halting_state, "valid_mask", None)
        ponder_cost = getattr(halting_state, "accumulated_ponder_cost", None)
        if ponder_cost is not None:
            cost = ponder_cost.detach().float()
            if valid_mask is not None and cost.dim() > 0:
                cost = cost[valid_mask]
            else:
                cost = cost.reshape(-1)
            self.last_ponder_cost = cost.reshape(-1)
            if cost.numel() > 0:
                self.last_ponder_cost_mean.copy_(cost.mean())
                self.last_ponder_cost_std.copy_(cost.std(unbiased=False))
            else:
                self.last_ponder_cost_mean.zero_()
                self.last_ponder_cost_std.zero_()

        halt_mask = halting_state.halt_mask
        if halt_mask is not None:
            if valid_mask is None:
                self.last_halted_fraction.copy_(halt_mask.float().mean())
            else:
                valid_count = valid_mask.sum()
                if bool(valid_count.item()):
                    halted_count = (halt_mask & valid_mask).sum()
                    self.last_halted_fraction.copy_(halted_count / valid_count)
                else:
                    self.last_halted_fraction.zero_()

        accumulated = getattr(halting_state, "accumulated_halt_probabilities", None)
        if accumulated is not None:
            acc = accumulated.detach().float()
            selected_halt_mask = halt_mask
            if valid_mask is not None:
                acc = acc[valid_mask]
                if halt_mask is not None:
                    selected_halt_mask = halt_mask[valid_mask]
            if acc.numel() == 0:
                self.last_accumulated_halt_prob_mean.zero_()
                self.last_remaining_mass_mean.zero_()
            else:
                self.last_accumulated_halt_prob_mean.copy_(acc.mean())
                remaining = 1.0 - acc
                if selected_halt_mask is not None:
                    remaining = remaining.masked_fill(selected_halt_mask, 0.0)
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
    plain calls rather than ``__call__``, so the manager wraps the two methods
    in the supported StickBreaking lifecycle on the halting module instance.
    """

    TRACKER_MODULE_NAME = "_usage_tracker"
    WRAPPED_METHOD_NAMES = (
        "update_halting_state",
        "finalize_weighted_accumulation",
    )

    def __init__(self):
        self._attachments: list[_HaltingAttachment] = []

    def attach(self, halting_model: "HaltingBase") -> HaltingUsageTracker:
        existing_tracker = getattr(halting_model, self.TRACKER_MODULE_NAME, None)
        if existing_tracker is not None:
            return existing_tracker

        self.__validate_supported_interface(halting_model)
        tracker = HaltingUsageTracker()
        try:
            self.__wrap_halting_methods(
                halting_model,
                tracker,
            )
            halting_model.add_module(self.TRACKER_MODULE_NAME, tracker)
        except Exception:
            self.__restore_halting_methods(halting_model)
            raise
        self._attachments.append(_HaltingAttachment(halting_model))
        return tracker

    @staticmethod
    def supports(halting_model: "HaltingBase") -> bool:
        return type(halting_model).implements_halting_interface()

    @classmethod
    def __validate_supported_interface(cls, halting_model: "HaltingBase") -> None:
        if cls.supports(halting_model):
            return
        strategy_type = type(halting_model)
        raise TypeError(
            f"{strategy_type.__name__} does not implement the supported halting "
            "interface: update_halting_state and "
            "finalize_weighted_accumulation"
        )

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
    ) -> None:
        original_update = halting_model.update_halting_state
        original_finalize = halting_model.finalize_weighted_accumulation

        @wraps(original_update)
        def update_halting_state(previous_state, *args, **kwargs):
            if previous_state is None:
                tracker.begin_forward()
            state, hidden = original_update(previous_state, *args, **kwargs)
            tracker.record_step(state)
            return state, hidden

        @wraps(original_finalize)
        def finalize_weighted_accumulation(state, *args, **kwargs):
            hidden, ponder_loss = original_finalize(state, *args, **kwargs)
            tracker.record_final(ponder_loss, state)
            return hidden, ponder_loss

        halting_model.update_halting_state = update_halting_state
        halting_model.finalize_weighted_accumulation = finalize_weighted_accumulation

    def __restore_halting_methods(self, halting_model: "HaltingBase") -> None:
        for method_name in self.WRAPPED_METHOD_NAMES:
            halting_model.__dict__.pop(method_name, None)
