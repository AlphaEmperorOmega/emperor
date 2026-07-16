from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from emperor_workbench.run_history import (
    RunHistoryFailure,
)
from tests.support.training_jobs import (
    write_tensorboard_run,
)
from tests.unit.run_history._support import log_run_scanner
from tests.unit.run_history._support import (
    run_history as _run_history,
)


class RunHistoryExperimentDeletionTests(unittest.TestCase):
    def test_run_history_deletes_experiment_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            deleted_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            second_deleted_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )
            remaining_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model_2",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "ccc_20260601_030405",
                    "version_0",
                ],
            )
            outside_target = root / "outside.txt"
            outside_target.write_text("outside", encoding="utf-8")
            logs_root.joinpath("test_model", "outside-link").symlink_to(outside_target)

            scanner = log_run_scanner(logs_root=logs_root)
            run_ids_by_path = {run.relative_path: run.id for run in scanner.list_runs()}
            service = _run_history(logs_root)
            result = service.delete_experiment("test_model")
            remaining_paths = {
                run.relative_path
                for run in log_run_scanner(logs_root=logs_root).list_runs()
            }

            self.assertEqual(result.experiment, "test_model")
            self.assertEqual(result.deleted_run_count, 2)
            self.assertEqual(result.deleted_relative_path, "test_model")
            self.assertEqual(
                set(result.deleted_run_ids),
                {
                    run_ids_by_path[deleted_run.relative_to(logs_root).as_posix()],
                    run_ids_by_path[
                        second_deleted_run.relative_to(logs_root).as_posix()
                    ],
                },
            )
            self.assertFalse(logs_root.joinpath("test_model").exists())
            self.assertTrue(remaining_run.exists())
            self.assertTrue(outside_target.exists())
            self.assertEqual(
                remaining_paths,
                {remaining_run.relative_to(logs_root).as_posix()},
            )

    def test_run_history_refuses_symlink_experiment_delete_and_preserves_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            logs_root.mkdir()
            outside_experiment = root / "outside_experiment"
            outside_experiment.mkdir()
            outside_marker = outside_experiment / "keep.txt"
            outside_marker.write_text("outside", encoding="utf-8")
            symlink_experiment = logs_root / "linked"
            symlink_experiment.symlink_to(
                outside_experiment,
                target_is_directory=True,
            )

            with self.assertRaisesRegex(RunHistoryFailure, "symlink"):
                _run_history(logs_root).delete_experiment("linked")

            self.assertTrue(symlink_experiment.is_symlink())
            self.assertTrue(outside_experiment.exists())
            self.assertEqual(outside_marker.read_text(encoding="utf-8"), "outside")

    def test_run_history_rejects_invalid_delete_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
            )
            outside_experiment = root / "outside_experiment"
            write_tensorboard_run(
                outside_experiment,
                ["linear", "BASELINE", "Mnist", "bbb_20260601_020304", "version_0"],
            )
            logs_root.joinpath("linked").symlink_to(
                outside_experiment,
                target_is_directory=True,
            )

            service = _run_history(logs_root)
            for experiment in (
                "",
                "../outside",
                "linear/BASELINE",
                ".",
                "..",
                "missing",
            ):
                with self.subTest(experiment=experiment):
                    with self.assertRaises(RunHistoryFailure):
                        service.delete_experiment(experiment)

            with self.assertRaisesRegex(RunHistoryFailure, "symlink"):
                service.delete_experiment("linked")

            self.assertTrue(logs_root.joinpath("linear").exists())
            self.assertTrue(outside_experiment.exists())


if __name__ == "__main__":
    unittest.main()
