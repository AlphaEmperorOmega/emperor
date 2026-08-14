import unittest

import torch
from torch import nn

from emperor.layers._composition.recurrent.runtime.execution.runtime_state import (
    RecurrentRuntimeStateGuard,
)


class TestRecurrentRuntimeStateGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.starting_rng_state = torch.get_rng_state()
        self.guard = RecurrentRuntimeStateGuard()
        self.module = nn.Module()
        self.module.register_buffer("progress", torch.tensor(2.0))
        self.module.register_buffer("optional", None)
        self.progress_buffer = self.module.progress

    def tearDown(self) -> None:
        torch.set_rng_state(self.starting_rng_state)

    def test_provisional_branch_always_restores_buffers_and_rng(self) -> None:
        torch.manual_seed(17)
        expected_next_random_value = torch.rand(())
        torch.manual_seed(17)

        with self.guard.isolate_provisional_branch(self.module):
            self.module.progress.add_(5.0)
            self.module.register_buffer(
                "progress",
                torch.tensor(99.0),
                persistent=False,
            )
            self.module.register_buffer(
                "temporary",
                torch.tensor(1.0),
                persistent=False,
            )
            self.module.register_buffer("optional", torch.tensor(4.0))
            torch.rand(())

        self.assertIs(self.module.progress, self.progress_buffer)
        torch.testing.assert_close(self.module.progress, torch.tensor(2.0))
        self.assertIsNone(self.module.optional)
        self.assertFalse(hasattr(self.module, "temporary"))
        self.assertEqual(set(self.module.state_dict()), {"progress"})
        torch.testing.assert_close(torch.rand(()), expected_next_random_value)

    def test_failed_handoff_restores_buffers_and_rng(self) -> None:
        torch.manual_seed(23)
        expected_next_random_value = torch.rand(())
        torch.manual_seed(23)

        with self.assertRaisesRegex(RuntimeError, "handoff failed"):
            with self.guard.rollback_handoff_on_failure(self.module):
                self.module.progress.add_(4.0)
                self.module.register_buffer("progress", torch.tensor(77.0))
                torch.rand(())
                raise RuntimeError("handoff failed")

        self.assertIs(self.module.progress, self.progress_buffer)
        torch.testing.assert_close(self.module.progress, torch.tensor(2.0))
        torch.testing.assert_close(torch.rand(()), expected_next_random_value)

    def test_successful_handoff_commits_buffers_and_rng(self) -> None:
        torch.manual_seed(31)
        expected_values = torch.rand(2)
        torch.manual_seed(31)

        with self.guard.rollback_handoff_on_failure(self.module):
            self.module.progress.add_(3.0)
            actual_in_scope_random_value = torch.rand(())

        self.assertIs(self.module.progress, self.progress_buffer)
        torch.testing.assert_close(self.module.progress, torch.tensor(5.0))
        torch.testing.assert_close(actual_in_scope_random_value, expected_values[0])
        torch.testing.assert_close(torch.rand(()), expected_values[1])
        self.assertEqual(vars(self.guard), {})


if __name__ == "__main__":
    unittest.main()
