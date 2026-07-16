from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support.training_jobs import (
    delete_filters_for_runs,
    write_tensorboard_run,
)
from tests.unit.run_history._support import (
    delete_plan as _delete_plan,
)
from tests.unit.run_history._support import (
    delete_runs as _delete_runs,
)
from tests.unit.run_history._support import log_run_scanner
from tests.unit.run_history._support import (
    run_history as _run_history,
)


class RunHistoryFilteredDeletionExecutionTests(unittest.TestCase):
    def test_run_history_deletes_filtered_version_dirs_and_prunes_empty_parents(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            mnist_run = write_tensorboard_run(
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
            cifar_run = write_tensorboard_run(
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
            gating_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "GATING",
                    "Mnist",
                    "ccc_20260601_030405",
                    "version_0",
                ],
            )
            other_experiment_run = write_tensorboard_run(
                logs_root,
                [
                    "other_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "ddd_20260601_040506",
                    "version_0",
                ],
            )

            service = _run_history(logs_root)
            runs = log_run_scanner(logs_root=logs_root).list_runs()
            filters = delete_filters_for_runs(
                runs,
                experiments=["test_model"],
                datasets=["Mnist"],
                presets=["BASELINE"],
            )
            plan = _delete_plan(service, filters)
            result = _delete_runs(service, filters)
            remaining_paths = {
                run.relative_path
                for run in log_run_scanner(logs_root=logs_root).list_runs()
            }

            self.assertTrue(plan.can_delete)
            self.assertEqual(len(plan.candidates), 1)
            self.assertEqual(len(result.deleted_run_ids), 1)
            self.assertEqual(
                result.deleted_relative_paths,
                (mnist_run.relative_to(logs_root).as_posix(),),
            )
            self.assertFalse(mnist_run.exists())
            self.assertFalse(mnist_run.parent.exists())
            self.assertTrue(cifar_run.exists())
            self.assertTrue(gating_run.exists())
            self.assertTrue(other_experiment_run.exists())
            self.assertEqual(
                remaining_paths,
                {
                    cifar_run.relative_to(logs_root).as_posix(),
                    gating_run.relative_to(logs_root).as_posix(),
                    other_experiment_run.relative_to(logs_root).as_posix(),
                },
            )

            second_filters = delete_filters_for_runs(
                log_run_scanner(logs_root=logs_root).list_runs(),
                experiments=["test_model"],
                datasets=["Cifar10"],
                presets=["BASELINE"],
            )
            _delete_runs(service, second_filters)
            self.assertFalse(cifar_run.exists())
            self.assertFalse(
                logs_root.joinpath(
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                ).exists()
            )
            self.assertTrue(logs_root.joinpath("test_model").exists())

    def test_run_history_deletes_exact_run_id_filter_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            first_run = write_tensorboard_run(
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
            second_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )

            runs = log_run_scanner(logs_root=logs_root).list_runs()
            first_run_id = next(
                run.id
                for run in runs
                if run.relative_path == first_run.relative_to(logs_root).as_posix()
            )
            filters = delete_filters_for_runs(runs, run_ids=[first_run_id])
            result = _delete_runs(_run_history(logs_root), filters)

            self.assertEqual(result.deleted_run_ids, (first_run_id,))
            self.assertFalse(first_run.exists())
            self.assertTrue(second_run.exists())

    def test_run_history_deletes_runs_across_multiple_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dirs = [
                write_tensorboard_run(
                    logs_root,
                    [
                        experiment,
                        "linear",
                        "BASELINE",
                        "Mnist",
                        f"run_20260711_0{index}0101",
                        "version_0",
                    ],
                )
                for index, experiment in enumerate(("exp_a", "exp_b"), start=1)
            ]
            runs = log_run_scanner(logs_root=logs_root).list_runs()
            filters = delete_filters_for_runs(runs)

            result = _delete_runs(_run_history(logs_root), filters)

            self.assertEqual(len(result.deleted_run_ids), 2)
            self.assertEqual(set(result.deleted_run_ids), {run.id for run in runs})
            self.assertTrue(all(not run_dir.exists() for run_dir in run_dirs))

    def test_partial_filtered_delete_failure_invalidates_public_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for name in (
                "first_20260711_010101",
                "second_20260711_020202",
            ):
                write_tensorboard_run(
                    logs_root,
                    [
                        "test_model",
                        "linear",
                        "BASELINE",
                        "Mnist",
                        name,
                        "version_0",
                    ],
                )
            service = _run_history(logs_root)
            runs = log_run_scanner(logs_root=logs_root).list_runs()
            filters = delete_filters_for_runs(runs)
            self.assertEqual(service.list_runs(limit=10, offset=0).total, 2)
            original_rmtree = shutil.rmtree
            delete_count = 0

            def fail_second_delete(path: Path):
                nonlocal delete_count
                delete_count += 1
                if delete_count == 2:
                    raise OSError("forced second Run delete failure")
                return original_rmtree(path)

            with (
                patch(
                    "emperor_workbench.run_history._deletion.shutil.rmtree",
                    side_effect=fail_second_delete,
                ),
                self.assertRaises(OSError),
            ):
                _delete_runs(service, filters)

            self.assertEqual(service.list_runs(limit=10, offset=0).total, 1)

    def test_run_history_prunes_only_empty_parents_under_logs_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            run_dir = write_tensorboard_run(
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
            sibling_dir = logs_root.joinpath(
                "test_model",
                "linear",
                "BASELINE",
                "Cifar10",
            )
            sibling_dir.mkdir(parents=True)
            sibling_marker = sibling_dir / "keep.txt"
            sibling_marker.write_text("keep", encoding="utf-8")
            scanner = log_run_scanner(logs_root=logs_root)

            _delete_runs(
                _run_history(logs_root),
                delete_filters_for_runs(scanner.list_runs()),
            )

            self.assertFalse(run_dir.exists())
            self.assertFalse(run_dir.parent.exists())
            self.assertFalse(
                logs_root.joinpath(
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                ).exists()
            )
            self.assertTrue(logs_root.exists())
            self.assertTrue(sibling_dir.exists())
            self.assertEqual(sibling_marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
