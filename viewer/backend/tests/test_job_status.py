from __future__ import annotations

import unittest

from viewer.backend.runtime.job_status import (
    ACTIVE_JOB_STATUSES,
    LIVE_PROCESS_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    is_active_job_status,
    is_live_process_job_status,
    is_terminal_job_status,
)


class TrainingJobStatusTests(unittest.TestCase):
    def test_active_statuses_match_current_training_job_blockers(self) -> None:
        self.assertEqual(
            ACTIVE_JOB_STATUSES,
            frozenset({"queued", "running", "unknown"}),
        )
        for status in ACTIVE_JOB_STATUSES:
            with self.subTest(status=status):
                self.assertTrue(is_active_job_status(status))

    def test_terminal_statuses_are_not_active(self) -> None:
        self.assertEqual(
            TERMINAL_JOB_STATUSES,
            frozenset({"completed", "failed", "cancelled"}),
        )
        for status in TERMINAL_JOB_STATUSES:
            with self.subTest(status=status):
                self.assertTrue(is_terminal_job_status(status))
                self.assertFalse(is_active_job_status(status))

    def test_only_queued_and_running_have_live_process_handles(self) -> None:
        self.assertEqual(
            LIVE_PROCESS_JOB_STATUSES,
            frozenset({"queued", "running"}),
        )
        self.assertTrue(is_live_process_job_status("queued"))
        self.assertTrue(is_live_process_job_status("running"))
        self.assertFalse(is_live_process_job_status("unknown"))

    def test_unrecognized_status_is_not_classified(self) -> None:
        self.assertFalse(is_active_job_status("paused"))
        self.assertFalse(is_terminal_job_status("paused"))
        self.assertFalse(is_live_process_job_status("paused"))


if __name__ == "__main__":
    unittest.main()
