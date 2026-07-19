import unittest
from types import SimpleNamespace

import torch
from emperor.halting._monitoring.tracking import HaltingUsageTracker


def state(
    *,
    halt_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    ponder_cost: torch.Tensor,
    accumulated_probability: torch.Tensor,
) -> SimpleNamespace:
    return SimpleNamespace(
        halt_mask=halt_mask,
        valid_mask=valid_mask,
        continuation_probability=1.0 - accumulated_probability,
        accumulated_ponder_cost=ponder_cost,
        accumulated_halt_probabilities=accumulated_probability,
    )


class HaltingUsageValidMaskTests(unittest.TestCase):
    def test_metrics_reduce_over_valid_positions_only(self) -> None:
        tracker = HaltingUsageTracker()
        halting_state = state(
            halt_mask=torch.tensor([False, False, True]),
            valid_mask=torch.tensor([True, False, True]),
            ponder_cost=torch.tensor([2.0, 100.0, 4.0]),
            accumulated_probability=torch.tensor([0.25, 50.0, 0.75]),
        )

        tracker.begin_forward()
        tracker.record_step(halting_state)
        tracker.record_final(torch.tensor(0.7), halting_state)

        torch.testing.assert_close(tracker.last_survival, torch.tensor([0.5]))
        torch.testing.assert_close(tracker.last_ponder_cost, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(tracker.last_ponder_cost_mean, torch.tensor(3.0))
        torch.testing.assert_close(tracker.last_ponder_cost_std, torch.tensor(1.0))
        torch.testing.assert_close(tracker.last_halted_fraction, torch.tensor(0.5))
        torch.testing.assert_close(
            tracker.last_accumulated_halt_prob_mean,
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            tracker.last_remaining_mass_mean,
            torch.tensor(0.375),
        )

    def test_empty_valid_domain_produces_finite_zero_metrics(self) -> None:
        tracker = HaltingUsageTracker()
        halting_state = state(
            halt_mask=torch.tensor([True, False]),
            valid_mask=torch.tensor([False, False]),
            ponder_cost=torch.tensor([2.0, 4.0]),
            accumulated_probability=torch.tensor([0.25, 0.75]),
        )

        tracker.begin_forward()
        tracker.record_step(halting_state)
        tracker.record_final(torch.tensor(0.0), halting_state)

        self.assertEqual(tracker.last_ponder_cost.numel(), 0)
        for metric in (
            tracker.last_survival,
            tracker.last_ponder_cost_mean,
            tracker.last_ponder_cost_std,
            tracker.last_halted_fraction,
            tracker.last_accumulated_halt_prob_mean,
            tracker.last_remaining_mass_mean,
        ):
            self.assertTrue(torch.isfinite(metric).all().item())
            torch.testing.assert_close(metric, torch.zeros_like(metric))


if __name__ == "__main__":
    unittest.main()
