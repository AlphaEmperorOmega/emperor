from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.training_jobs._monitoring import TrainingMonitorLocator
from tests.unit.training_jobs._support import make_job


class TrainingMonitorLocatorTests(unittest.TestCase):
    def test_locator_matches_exact_presets_and_latest_log_dir(self) -> None:
        locator = TrainingMonitorLocator()
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            job.presets = ["expert_gate"]

            events = [
                {
                    "dataset": "Mnist",
                    "preset": "expert_gate",
                    "logDir": "old",
                },
                {
                    "dataset": "Mnist",
                    "preset": "expert_gate",
                    "logDir": "new",
                },
            ]

            self.assertTrue(locator.preset_in_job(job, "expert_gate"))
            self.assertFalse(locator.preset_in_job(job, "expert-gate"))
            self.assertEqual(
                locator.log_dir_for_monitor_data(
                    events=events,
                    dataset="Mnist",
                    preset="expert_gate",
                ),
                "new",
            )


if __name__ == "__main__":
    unittest.main()
