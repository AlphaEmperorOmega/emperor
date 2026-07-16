from __future__ import annotations

import asyncio
import threading
import unittest

from emperor_workbench.failures import FailureKind
from emperor_workbench.log_experiments import (
    LogExperimentFailure,
    LogExperimentMutationCoordinator,
)


class LogExperimentMutationCoordinatorTests(unittest.TestCase):
    def test_reversed_multi_experiment_requests_use_one_stable_lock_order(
        self,
    ) -> None:
        coordinator = LogExperimentMutationCoordinator()
        ready = threading.Barrier(2)
        entered: list[str] = []

        def coordinate(label: str, experiments: list[str]) -> None:
            ready.wait(timeout=5)
            with coordinator.coordinate(experiments):
                entered.append(label)

        first = threading.Thread(
            target=coordinate,
            args=("first", ["alpha", "beta"]),
        )
        second = threading.Thread(
            target=coordinate,
            args=("second", ["beta", "alpha"]),
        )
        first.start()
        second.start()
        first.join(timeout=5)
        second.join(timeout=5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertCountEqual(entered, ["first", "second"])

    def test_different_experiments_do_not_block_each_other(self) -> None:
        coordinator = LogExperimentMutationCoordinator()
        first_held = threading.Event()
        release_first = threading.Event()

        def hold_first_experiment() -> None:
            with coordinator.coordinate(["first"]):
                first_held.set()
                release_first.wait(timeout=5)

        holder = threading.Thread(target=hold_first_experiment)
        holder.start()
        self.assertTrue(first_held.wait(timeout=5))
        try:
            with coordinator.coordinate(["second"]):
                pass
        finally:
            release_first.set()
            holder.join(timeout=5)

        self.assertFalse(holder.is_alive())

    def test_scope_releases_after_exception_and_cancellation(self) -> None:
        coordinator = LogExperimentMutationCoordinator()

        with self.assertRaisesRegex(RuntimeError, "operation failed"):
            with coordinator.coordinate(["experiment"]):
                raise RuntimeError("operation failed")
        with coordinator.coordinate(["experiment"]):
            pass

        with self.assertRaises(asyncio.CancelledError):
            with coordinator.coordinate(["experiment"]):
                raise asyncio.CancelledError
        with coordinator.coordinate(["experiment"]):
            pass

    def test_timeout_releases_partial_scope_and_later_callers_can_continue(
        self,
    ) -> None:
        coordinator = LogExperimentMutationCoordinator(acquire_timeout_seconds=0.05)
        held = threading.Event()
        release = threading.Event()

        def hold_second_experiment() -> None:
            with coordinator.coordinate(["second"]):
                held.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=hold_second_experiment)
        holder.start()
        self.assertTrue(held.wait(timeout=5))
        try:
            with self.assertRaises(LogExperimentFailure) as context:
                with coordinator.coordinate(["first", "second"]):
                    pass
            self.assertEqual(context.exception.kind, FailureKind.UNAVAILABLE)
            with coordinator.coordinate(["first"]):
                pass
        finally:
            release.set()
            holder.join(timeout=5)

        self.assertFalse(holder.is_alive())
        with coordinator.coordinate(["second"]):
            pass


if __name__ == "__main__":
    unittest.main()
