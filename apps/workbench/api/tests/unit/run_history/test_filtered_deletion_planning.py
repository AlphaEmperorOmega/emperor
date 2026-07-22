from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from emperor_workbench.run_history import (
    LogRunDeleteCandidate,
    LogRunDeleteFilters,
    LogRunDeletePlan,
    RunHistoryFailure,
)
from emperor_workbench.run_history._deletion import LogRunDeletionExecutor
from tests.support.training_jobs import (
    delete_filters_for_runs,
    write_tensorboard_run,
)
from tests.unit.run_history._support import (
    ActiveWriter as _ActiveWriter,
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


class RunHistoryFilteredDeletionPlanningTests(unittest.TestCase):
    @staticmethod
    def _delete_candidate(
        relative_path: str,
        *,
        candidate_id: str = "candidate-id",
        experiment: str = "test_model",
        model: str = "linears/linear",
        preset: str = "BASELINE",
        dataset: str = "Mnist",
        run_name: str = "aaa_20260601_010203",
    ) -> LogRunDeleteCandidate:
        return LogRunDeleteCandidate(
            id=candidate_id,
            experiment=experiment,
            model=model,
            preset=preset,
            dataset=dataset,
            run_name=run_name,
            version=Path(relative_path).name,
            relative_path=relative_path,
        )

    def test_run_history_filtered_delete_candidate_safety_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            non_version_dir = run_dir.parent / "notes"
            non_version_dir.mkdir()
            non_version_marker = non_version_dir / "keep.txt"
            non_version_marker.write_text("keep", encoding="utf-8")
            outside_version = root / "outside_version"
            outside_version.mkdir(parents=True)
            outside_marker = outside_version / "keep.txt"
            outside_marker.write_text("outside", encoding="utf-8")
            run_parent = logs_root.joinpath(
                "test_model",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "bbb_20260601_020304",
            )
            run_parent.mkdir(parents=True)
            symlink_version = run_parent / "version_0"
            symlink_version.symlink_to(outside_version, target_is_directory=True)
            escaped_version = root.joinpath(
                "outside",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "aaa_20260601_010203",
                "version_0",
            )
            escaped_version.mkdir(parents=True)
            escaped_marker = escaped_version / "keep.txt"
            escaped_marker.write_text("escaped", encoding="utf-8")
            scanner = log_run_scanner(logs_root=logs_root)
            executor = LogRunDeletionExecutor(scanner=scanner)
            filters = delete_filters_for_runs(scanner.list_runs())
            result = _delete_runs(_run_history(logs_root), filters)

            self.assertEqual(
                result.deleted_relative_paths,
                (
                    "test_model/linears/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
                ),
            )
            self.assertFalse(run_dir.exists())

            cases = (
                (
                    "symlink_version_candidate",
                    symlink_version.relative_to(logs_root).as_posix(),
                    "symlink log run",
                    (symlink_version, outside_marker),
                ),
                (
                    "escaped_relative_path",
                    "../outside/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
                    "Invalid log run path",
                    (escaped_version, escaped_marker),
                ),
                (
                    "non_version_directory",
                    non_version_dir.relative_to(logs_root).as_posix(),
                    "non-version log folder",
                    (non_version_dir, non_version_marker),
                ),
            )

            for label, relative_path, error_pattern, preserved_paths in cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(
                        RunHistoryFailure,
                        error_pattern,
                    ):
                        executor.delete_runs(
                            LogRunDeletePlan(
                                candidates=[self._delete_candidate(relative_path)]
                            )
                        )
                    for preserved_path in preserved_paths:
                        self.assertTrue(preserved_path.exists())
            self.assertTrue(symlink_version.is_symlink())
            self.assertEqual(outside_marker.read_text(encoding="utf-8"), "outside")
            self.assertEqual(escaped_marker.read_text(encoding="utf-8"), "escaped")
            self.assertEqual(non_version_marker.read_text(encoding="utf-8"), "keep")

    def test_preset_delete_is_backend_authoritative_beyond_filter_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for index in range(55):
                write_tensorboard_run(
                    logs_root,
                    [
                        "test_model",
                        "linears",
                        "linear",
                        "BASELINE",
                        "Mnist",
                        f"run_{index:03d}_20260711_010101",
                        "version_0",
                    ],
                    checkpoint=False,
                )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "GATING",
                    "Mnist",
                    "keep_20260711_020202",
                    "version_0",
                ],
                checkpoint=False,
            )
            service = _run_history(logs_root)

            preview = service.create_preset_delete_plan(
                experiment="test_model",
                preset="BASELINE",
            )
            result = service.delete_preset(
                experiment="test_model",
                preset="BASELINE",
            )

            self.assertEqual(len(preview.candidates), 55)
            self.assertEqual(len(result.deleted_run_ids), 55)
            remaining = service.list_runs(limit=100, offset=0).runs
            self.assertEqual([run.preset for run in remaining], ["GATING"])

    def test_preset_delete_recomputes_stale_preview_and_blocks_active_writer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            base_parts = [
                "test_model",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
            ]
            write_tensorboard_run(
                logs_root,
                [*base_parts, "first_20260711_010101", "version_0"],
            )
            service = _run_history(logs_root)
            self.assertEqual(
                len(
                    service.create_preset_delete_plan(
                        experiment="test_model",
                        preset="BASELINE",
                    ).candidates
                ),
                1,
            )
            write_tensorboard_run(
                logs_root,
                [*base_parts, "second_20260711_020202", "version_0"],
            )
            self.assertEqual(
                len(
                    service.delete_preset(
                        experiment="test_model",
                        preset="BASELINE",
                    ).deleted_run_ids
                ),
                2,
            )

            write_tensorboard_run(
                logs_root,
                [*base_parts, "third_20260711_030303", "version_0"],
            )
            blocked = _run_history(
                logs_root,
                active_writers=[
                    _ActiveWriter(
                        id="job-1",
                        status="running",
                        log_folder="test_model",
                    )
                ],
            )
            blocked_plan = blocked.create_preset_delete_plan(
                experiment="test_model",
                preset="BASELINE",
            )
            self.assertFalse(blocked_plan.can_delete)
            self.assertEqual(blocked_plan.blocked_by_active_jobs[0].id, "job-1")
            with self.assertRaisesRegex(RunHistoryFailure, "still writing"):
                blocked.delete_preset(
                    experiment="test_model",
                    preset="BASELINE",
                )

    def test_delete_recomputes_stale_preview_without_a_plan_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            relative_parts = [
                "test_model",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "run_20260711_080910",
                "version_0",
            ]
            write_tensorboard_run(logs_root, relative_parts)
            service = _run_history(logs_root)
            run = log_run_scanner(logs_root=logs_root).list_runs()[0]
            filters = delete_filters_for_runs([run])
            preview = _delete_plan(service, filters)
            self.assertEqual(len(preview.candidates), 1)

            service.delete_experiment("test_model")
            with self.assertRaisesRegex(
                RunHistoryFailure,
                "No log runs match",
            ):
                _delete_runs(service, filters)

            recreated = write_tensorboard_run(logs_root, relative_parts)
            recreated_preview = _delete_plan(service, filters)
            self.assertEqual(len(recreated_preview.candidates), 1)
            result = _delete_runs(service, filters)
            self.assertEqual(result.deleted_run_ids, (run.id,))
            self.assertFalse(recreated.exists())

    def test_log_run_delete_partial_filters_match_nothing_and_preserve_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            service = _run_history(logs_root)
            run = log_run_scanner(logs_root=logs_root).list_runs()[0]
            full_filter_fields = {
                "experiments": [run.experiment],
                "datasets": [run.dataset],
                "models": [run.model],
                "presets": [run.preset],
                "run_ids": [run.id],
            }
            cases = (
                ("missing_experiments", {**full_filter_fields, "experiments": []}),
                ("missing_datasets", {**full_filter_fields, "datasets": []}),
                ("missing_models", {**full_filter_fields, "models": []}),
                ("missing_presets", {**full_filter_fields, "presets": []}),
                ("missing_run_ids", {**full_filter_fields, "run_ids": []}),
                (
                    "mismatched_dataset",
                    {**full_filter_fields, "datasets": ["Cifar10"]},
                ),
                (
                    "mismatched_experiment",
                    {**full_filter_fields, "experiments": ["other_model"]},
                ),
                ("mismatched_model", {**full_filter_fields, "models": ["convnet"]}),
                ("mismatched_preset", {**full_filter_fields, "presets": ["GATING"]}),
                ("mismatched_run_id", {**full_filter_fields, "run_ids": ["missing"]}),
            )

            for label, fields in cases:
                with self.subTest(label=label):
                    filters = LogRunDeleteFilters(**fields)
                    plan = _delete_plan(service, filters)

                    self.assertFalse(plan.can_delete)
                    self.assertEqual(plan.candidates, ())
                    with self.assertRaisesRegex(
                        RunHistoryFailure,
                        "No log runs match",
                    ):
                        _delete_runs(service, filters)
                    self.assertTrue(run_dir.exists())

    def test_log_run_delete_empty_filters_match_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )

            service = _run_history(logs_root)
            filters = LogRunDeleteFilters(
                experiments=("test_model",),
                datasets=(),
                models=("linears/linear",),
                presets=("BASELINE",),
                run_ids=(log_run_scanner(logs_root=logs_root).list_runs()[0].id,),
            )
            plan = _delete_plan(service, filters)

            self.assertFalse(plan.can_delete)
            self.assertEqual(plan.candidates, ())
            with self.assertRaisesRegex(
                RunHistoryFailure,
                "No log runs match",
            ):
                _delete_runs(service, filters)

    def test_run_history_active_job_blocks_filtered_destructive_delete(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            filters = delete_filters_for_runs(
                log_run_scanner(logs_root=logs_root).list_runs()
            )
            service = _run_history(
                logs_root,
                active_writers=[
                    _ActiveWriter(
                        id="job-1",
                        log_folder="test_model",
                        status="running",
                    )
                ],
            )

            plan = _delete_plan(service, filters)

            self.assertFalse(plan.can_delete)
            self.assertEqual(len(plan.blocked_by_active_jobs), 1)
            self.assertEqual(
                plan.blocked_by_active_jobs[0].log_folder,
                "test_model",
            )
            with self.assertRaisesRegex(
                RunHistoryFailure,
                "training job is still writing",
            ):
                _delete_runs(service, filters)
            self.assertTrue(run_dir.exists())


if __name__ == "__main__":
    unittest.main()
