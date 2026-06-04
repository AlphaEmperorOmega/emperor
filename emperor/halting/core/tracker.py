import torch

from dataclasses import dataclass
from torch import Tensor
from emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.halting.utils.options.base import HaltingBase, HaltingStateBase


class HaltingUsageTracker(Module):
    """Records adaptive-compute statistics for a single halting module.

    Buffers hold the most recent forward's survival curve (fraction of tokens
    still computing after each recurrence step) and scalar summaries (ponder
    depth, halted fraction, ponder loss). Survival is staged per step and
    committed on ``record_final``; its length equals the steps actually run, so
    it varies between forwards when halting exits early.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("last_survival", torch.zeros(0))
        self.register_buffer("last_ponder_cost_mean", torch.zeros(()))
        self.register_buffer("last_ponder_cost_std", torch.zeros(()))
        self.register_buffer("last_step_count", torch.zeros(()))
        self.register_buffer("last_halted_fraction", torch.zeros(()))
        self.register_buffer("last_accumulated_halt_prob_mean", torch.zeros(()))
        self.register_buffer("last_remaining_mass_mean", torch.zeros(()))
        self.register_buffer("last_ponder_loss", torch.zeros(()))
        self._survival_stage: list[Tensor] = []

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
        halt_mask = getattr(halting_state, "halt_mask", None)
        if halt_mask is not None:
            return (~halt_mask).float().mean().detach()
        continuation = getattr(halting_state, "continuation_probability", None)
        if continuation is not None:
            return continuation.detach().float().clamp(0.0, 1.0).mean()
        return torch.ones((), device=self.last_survival.device)

    def __commit_survival(self) -> None:
        device = self.last_survival.device
        if self._survival_stage:
            self.last_survival = torch.stack(
                [value.to(device).reshape(()) for value in self._survival_stage]
            )
        else:
            self.last_survival = torch.zeros(0, device=device)

    def __record_final_scalars(
        self,
        ponder_loss: Tensor | None,
        halting_state: "HaltingStateBase",
    ) -> None:
        self.last_step_count.copy_(
            torch.tensor(
                float(len(self._survival_stage)),
                device=self.last_step_count.device,
            )
        )

        ponder_cost = getattr(halting_state, "accumulated_ponder_cost", None)
        if ponder_cost is not None:
            cost = ponder_cost.detach().float()
            self.last_ponder_cost_mean.copy_(cost.mean())
            self.last_ponder_cost_std.copy_(cost.std(unbiased=False))

        halt_mask = getattr(halting_state, "halt_mask", None)
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
        self.last_survival = torch.zeros(0, device=self.last_survival.device)
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
    tracker: HaltingUsageTracker


class HaltingUsageTrackerManager:
    """Attaches halting trackers without modifying any traced code.

    Capture is installed at runtime and removed on detach, mirroring how the
    linear monitor installs and removes forward hooks. Because
    ``update_halting_state`` / ``finalize_weighted_accumulation`` are plain method
    calls (not ``__call__``), they are wrapped on the halting module instance
    rather than hooked. A fresh forward is detected from ``previous_state is
    None`` (the first halting step of any owner: recurrent layer or neuron
    cluster), so no owner-specific hook or ``max_steps`` is required.
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

        tracker = HaltingUsageTracker()
        halting_model.add_module(self.TRACKER_MODULE_NAME, tracker)
        self.__wrap_halting_methods(halting_model, tracker)
        self._attachments.append(_HaltingAttachment(halting_model, tracker))
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
    ) -> None:
        original_update = halting_model.update_halting_state
        original_finalize = halting_model.finalize_weighted_accumulation

        def update_halting_state(previous_state, model_hidden_state):
            if previous_state is None:
                tracker.begin_forward()
            state, hidden = original_update(previous_state, model_hidden_state)
            tracker.record_step(state)
            return state, hidden

        def finalize_weighted_accumulation(state, current_hidden):
            hidden, ponder_loss = original_finalize(state, current_hidden)
            tracker.record_final(ponder_loss, state)
            return hidden, ponder_loss

        halting_model.update_halting_state = update_halting_state
        halting_model.finalize_weighted_accumulation = finalize_weighted_accumulation

    def __restore_halting_methods(self, halting_model: "HaltingBase") -> None:
        for method_name in self.WRAPPED_METHOD_NAMES:
            if method_name in halting_model.__dict__:
                del halting_model.__dict__[method_name]
