from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.log_migration import (
    LogMigrationError,
    apply_log_migration,
    plan_log_migration,
)


def write_flat_run(root: Path, *parts: str) -> Path:
    run_dir = root.joinpath(*parts, "BASELINE", "Mnist", "run", "version_0")
    run_dir.mkdir(parents=True)
    return run_dir


class LogMigrationTests(unittest.TestCase):
    def test_dry_run_plans_flat_default_model_without_moving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            source_run = write_flat_run(logs_root, "linear")

            actions = plan_log_migration(logs_root)

            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0].model_id, "linears/linear")
            self.assertEqual(actions[0].source, logs_root / "linear")
            self.assertEqual(actions[0].destination, logs_root / "linears" / "linear")
            self.assertTrue(source_run.exists())
            self.assertFalse((logs_root / "linears" / "linear").exists())

    def test_apply_moves_custom_flat_model_and_rerun_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            source_run = write_flat_run(logs_root, "comparison", "linear")

            actions = apply_log_migration(logs_root, apply=True)
            rerun_actions = apply_log_migration(logs_root, apply=True)

            destination = (
                logs_root
                / "comparison"
                / "linears"
                / "linear"
                / "BASELINE"
                / "Mnist"
                / "run"
                / "version_0"
            )
            self.assertEqual(len(actions), 1)
            self.assertFalse(source_run.exists())
            self.assertTrue(destination.exists())
            self.assertEqual(rerun_actions, [])

    def test_destination_collision_raises_without_merging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_flat_run(logs_root, "linear")
            (logs_root / "linears" / "linear").mkdir(parents=True)

            with self.assertRaisesRegex(LogMigrationError, "Destination already exists"):
                plan_log_migration(logs_root)

    def test_symlinked_source_raises_without_moving_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            logs_root.mkdir()
            outside = root / "outside_linear"
            write_flat_run(outside)
            symlink_source = logs_root / "linear"
            symlink_source.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(LogMigrationError, "symlinked"):
                plan_log_migration(logs_root)

            self.assertTrue(symlink_source.is_symlink())
            self.assertTrue(outside.exists())


if __name__ == "__main__":
    unittest.main()
