import unittest
from types import SimpleNamespace

import torch

from emperor.halting import HaltingUsageTracker


def state(halt_mask: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        halt_mask=halt_mask,
        valid_mask=torch.ones_like(halt_mask, dtype=torch.bool),
        continuation_probability=(~halt_mask).float(),
        accumulated_ponder_cost=torch.zeros_like(halt_mask, dtype=torch.float32),
        accumulated_halt_probabilities=halt_mask.float(),
    )


class HaltingUsageReconciliationTests(unittest.TestCase):
    def test_injected_step_observation_sees_post_return_reconciliation(self) -> None:
        tracker = HaltingUsageTracker()
        tracker.begin_forward()
        reconciled_state = state(torch.tensor([False, False]))
        tracker.record_step(reconciled_state)

        reconciled_state.halt_mask.fill_(True)
        reconciled_state.continuation_probability.zero_()
        tracker.record_final(torch.tensor(0.0), reconciled_state)

        torch.testing.assert_close(tracker.last_survival, torch.tensor([0.0]))
        torch.testing.assert_close(tracker.last_step_count, torch.tensor(1.0))
        self.assertEqual(tracker._survival_stage, [])


if __name__ == "__main__":
    unittest.main()
