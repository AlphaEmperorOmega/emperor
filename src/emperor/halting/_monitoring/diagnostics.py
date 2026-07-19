"""Calculate halting diagnostics without Lightning or emission concerns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting._monitoring.tracking import HaltingUsageTracker


@dataclass(frozen=True)
class _HaltingDiagnosticMetrics:
    ponder_cost_mean: Tensor
    ponder_cost_std: Tensor
    ponder_cost: Tensor
    step_count: Tensor
    halted_fraction: Tensor
    accumulated_halt_probability_mean: Tensor
    remaining_mass_mean: Tensor
    final_survival_fraction: Tensor
    ponder_loss: Tensor
    survival: Tensor


class _HaltingDiagnostics:
    @staticmethod
    def calculate(tracker: HaltingUsageTracker) -> _HaltingDiagnosticMetrics:
        survival = tracker.last_survival.detach().float()
        final_survival_fraction = (
            survival[-1] if survival.numel() else survival.new_zeros(())
        )
        return _HaltingDiagnosticMetrics(
            ponder_cost_mean=tracker.last_ponder_cost_mean.detach().float(),
            ponder_cost_std=tracker.last_ponder_cost_std.detach().float(),
            ponder_cost=tracker.last_ponder_cost.detach().float(),
            step_count=tracker.last_step_count.detach().float(),
            halted_fraction=tracker.last_halted_fraction.detach().float(),
            accumulated_halt_probability_mean=(
                tracker.last_accumulated_halt_prob_mean.detach().float()
            ),
            remaining_mass_mean=tracker.last_remaining_mass_mean.detach().float(),
            final_survival_fraction=final_survival_fraction,
            ponder_loss=tracker.last_ponder_loss.detach().float(),
            survival=survival,
        )
