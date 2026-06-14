from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from viewer.backend.inspector.errors import InspectorError
from viewer.backend.log_runs import (
    LOG_EXPERIMENT_NAME_RE,
    ActiveLogRunDeleteBlocker,
    LogRunDeleteCandidate,
    LogRunDeleteFilters,
    LogRunDeletePlan,
    LogRunDeleteResult,
    LogRunIndex,
    is_valid_log_experiment_name,
    validate_log_experiment_name,
)
from viewer.backend.tests.helpers import (
    FakeRunner,
    delete_filters_for_runs,
    write_tensorboard_run,
)
from viewer.backend.training_jobs import TrainingJobManager


class LogExperimentNameTests(unittest.TestCase):
    def test_log_experiment_name_regex_pattern_is_stable(self) -> None:
        self.assertEqual(
            LOG_EXPERIMENT_NAME_RE.pattern,
            r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$",
        )

    def test_is_valid_log_experiment_name_accepts_allowed_names(self) -> None:
        for name in ("abc", "abc_123", "A1_B2"):
            with self.subTest(name=name):
                self.assertTrue(is_valid_log_experiment_name(name))

    def test_is_valid_log_experiment_name_rejects_disallowed_names(self) -> None:
        for name in (
            "",
            ".",
            "..",
            "_abc",
            "abc_",
            "abc__def",
            "abc-def",
            "abc.def",
            "abc/def",
            "abcé",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_valid_log_experiment_name(name))

    def test_validate_log_experiment_name_returns_allowed_names(self) -> None:
        for name in ("abc", "abc_123", "A1_B2"):
            with self.subTest(name=name):
                self.assertEqual(validate_log_experiment_name(name), name)

    def test_validate_log_experiment_name_rejects_disallowed_names(self) -> None:
        for name in (
            "",
            ".",
            "..",
            "_abc",
            "abc_",
            "abc__def",
            "abc-def",
            "abc.def",
            "abc/def",
            "abcé",
        ):
            with self.subTest(name=name):
                with self.assertRaises(InspectorError):
                    validate_log_experiment_name(name)


class LogRunDeleteResponseTests(unittest.TestCase):
    def test_delete_plan_and_result_response_payloads_are_stable(self) -> None:
        candidate = LogRunDeleteCandidate(
            id="run-1",
            experiment="test_model",
            model="linears/linear",
            preset="BASELINE",
            dataset="Mnist",
            runName="aaa_20260601_010203",
            version="version_0",
            relativePath=(
                "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
            ),
        )
        blocker = ActiveLogRunDeleteBlocker(
            id="job-1",
            logFolder="test_model",
            status="running",
        )
        expected_common = {
            "candidateCount": 1,
            "counts": {
                "runs": 1,
                "experiments": 1,
                "datasets": 1,
                "models": 1,
                "presets": 1,
            },
            "affected": {
                "experiments": ["test_model"],
                "datasets": ["Mnist"],
                "models": ["linears/linear"],
                "presets": ["BASELINE"],
                "runIds": ["run-1"],
            },
            "candidates": [
                {
                    "id": "run-1",
                    "experiment": "test_model",
                    "model": "linears/linear",
                    "preset": "BASELINE",
                    "dataset": "Mnist",
                    "runName": "aaa_20260601_010203",
                    "version": "version_0",
                    "relativePath": (
                        "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
                    ),
                }
            ],
        }

        plan_payload = LogRunDeletePlan(
            candidates=[candidate],
            blockedByActiveJobs=[blocker],
        ).to_response()
        result_payload = LogRunDeleteResult(
            candidates=[candidate],
            deletedRunIds=["run-1"],
            deletedRelativePaths=[candidate.relativePath],
        ).to_response()

        self.assertEqual(
            plan_payload,
            {
                **expected_common,
                "blockedByActiveJobs": [
                    {
                        "id": "job-1",
                        "logFolder": "test_model",
                        "status": "running",
                    }
                ],
                "canDelete": False,
            },
        )
        self.assertEqual(
            result_payload,
            {
                "deletedRunIds": ["run-1"],
                "deletedRunCount": 1,
                "deletedRelativePaths": [candidate.relativePath],
                **expected_common,
                "blockedByActiveJobs": [],
                "canDelete": True,
            },
        )


class LogRunIndexAndApiTests(unittest.TestCase):
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
            runName=run_name,
            version=Path(relative_path).name,
            relativePath=relative_path,
        )

    def test_log_run_index_parses_supported_log_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            default_run = write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
                metrics={"test/accuracy": 0.9},
            )
            categorized_run = write_tensorboard_run(
                logs_root,
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "categorized_20260601_010204",
                    "version_0",
                ],
                metrics={"test/accuracy": 0.91},
            )
            custom_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear_adaptive",
                    "DUAL_MODEL_WEIGHT",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_1",
                ],
                metrics=None,
                hparams=False,
                checkpoint=False,
            )
            viewer_run = write_tensorboard_run(
                logs_root,
                [
                    "viewer-training",
                    "job-123",
                    "linear",
                    "BASELINE",
                    "FashionMNIST",
                    "ccc_20260601_030405",
                    "version_0",
                ],
                metrics={"validation/accuracy": 0.8},
            )
            no_event_run = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Mnist",
                "ddd_20260601_040506",
                "version_2",
            )
            no_event_run.mkdir(parents=True)
            malformed_result_run = write_tensorboard_run(
                logs_root,
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "eee_20260601_050607",
                    "version_0",
                ],
                metrics=None,
            )
            (malformed_result_run / "result.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )
            outside_run = write_tensorboard_run(
                Path(tmp) / "outside",
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "fff_20260601_060708",
                    "version_0",
                ],
                metrics={"test/accuracy": 1.0},
            )
            escaped_run_parent = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Mnist",
                "escaped_20260601_070809",
            )
            escaped_run_parent.mkdir(parents=True)
            escaped_run = escaped_run_parent / "version_99"
            escaped_run.symlink_to(outside_run, target_is_directory=True)

            runs = LogRunIndex(logs_root=logs_root).list_runs()

        by_path = {run.relativePath: run for run in runs}
        default_summary = by_path[default_run.relative_to(logs_root).as_posix()]
        categorized_summary = by_path[categorized_run.relative_to(logs_root).as_posix()]
        custom_summary = by_path[custom_run.relative_to(logs_root).as_posix()]
        viewer_summary = by_path[viewer_run.relative_to(logs_root).as_posix()]
        no_event_summary = by_path[no_event_run.relative_to(logs_root).as_posix()]
        malformed_result_summary = by_path[
            malformed_result_run.relative_to(logs_root).as_posix()
        ]

        self.assertIsNone(default_summary.group)
        self.assertEqual(default_summary.experiment, "linear")
        self.assertEqual(default_summary.model, "linears/linear")
        self.assertEqual(default_summary.preset, "BASELINE")
        self.assertEqual(default_summary.dataset, "Mnist")
        self.assertEqual(default_summary.timestamp, "2026-06-01 01:02:03")
        self.assertTrue(default_summary.hasResult)
        self.assertGreater(default_summary.eventFileCount, 0)
        self.assertEqual(default_summary.checkpointCount, 1)
        self.assertTrue(default_summary.hasHparams)
        self.assertEqual(default_summary.metrics["test/accuracy"], 0.9)

        self.assertIsNone(categorized_summary.group)
        self.assertEqual(categorized_summary.experiment, "linears")
        self.assertEqual(categorized_summary.model, "linears/linear")
        self.assertEqual(categorized_summary.preset, "BASELINE")
        self.assertEqual(categorized_summary.dataset, "Mnist")
        self.assertEqual(categorized_summary.metrics["test/accuracy"], 0.91)

        self.assertEqual(custom_summary.group, "test_model")
        self.assertEqual(custom_summary.experiment, "test_model")
        self.assertEqual(custom_summary.model, "linears/linear_adaptive")
        self.assertFalse(custom_summary.hasResult)
        self.assertFalse(custom_summary.hasHparams)
        self.assertEqual(custom_summary.checkpointCount, 0)

        self.assertEqual(viewer_summary.group, "viewer-training/job-123")
        self.assertEqual(viewer_summary.experiment, "viewer-training")
        self.assertEqual(viewer_summary.model, "linears/linear")
        self.assertEqual(viewer_summary.dataset, "FashionMNIST")

        self.assertFalse(no_event_summary.hasResult)
        self.assertEqual(no_event_summary.eventFileCount, 0)
        self.assertEqual(no_event_summary.checkpointCount, 0)
        self.assertFalse(no_event_summary.hasHparams)
        self.assertEqual(no_event_summary.metrics, {})

        self.assertTrue(malformed_result_summary.hasResult)
        self.assertGreater(malformed_result_summary.eventFileCount, 0)
        self.assertEqual(malformed_result_summary.metrics, {})
        self.assertNotIn(
            "linear/BASELINE/Mnist/escaped_20260601_070809/version_99",
            by_path,
        )

    def test_log_run_index_reads_checkpoints_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
                metrics={"test/accuracy": 0.9},
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "params": {"learning_rate": 0.01, "optimizer": "adam"},
                        "metrics": {"test/accuracy": 0.9},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "hparams.yaml").write_text(
                "\n".join(
                    [
                        "batch_size: 4",
                        "use_bias: true",
                        "description: 'baseline run'",
                        "nested:",
                        "ignored_list: [1, 2]",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "last.ckpt").write_text(
                "checkpoint",
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "epoch=2-step=300.ckpt").write_text(
                "checkpoint",
                encoding="utf-8",
            )
            malformed_run = write_tensorboard_run(
                logs_root,
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "malformed_20260601_050607",
                    "version_0",
                ],
                metrics=None,
            )
            (malformed_run / "result.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )
            (malformed_run / "hparams.yaml").write_text(
                "nested:\nignored_list: [1, 2]\n",
                encoding="utf-8",
            )

            index = LogRunIndex(logs_root=logs_root)
            runs_by_path = {run.relativePath: run for run in index.list_runs()}
            run = runs_by_path[
                "linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
            ]
            malformed = runs_by_path[
                "linear/BASELINE/Mnist/malformed_20260601_050607/version_0"
            ]

            checkpoints = index.checkpoints_for_runs([run.id])
            artifacts = index.artifacts_for_run(run.id)
            malformed_artifacts = index.artifacts_for_run(malformed.id)
            with self.assertRaises(InspectorError):
                index.checkpoints_for_runs(["not-a-run"])
            with self.assertRaises(InspectorError):
                index.artifacts_for_run("not-a-run")

        self.assertEqual(
            [
                (
                    checkpoint["filename"],
                    checkpoint["epoch"],
                    checkpoint["step"],
                )
                for checkpoint in checkpoints
            ],
            [
                ("epoch=0-step=1.ckpt", 0, 1),
                ("epoch=2-step=300.ckpt", 2, 300),
                ("last.ckpt", None, None),
            ],
        )
        self.assertTrue(
            checkpoints[0]["relativePath"].endswith(
                "linear/BASELINE/Mnist/aaa_20260601_010203/version_0/"
                "checkpoints/epoch=0-step=1.ckpt"
            )
        )
        self.assertGreater(checkpoints[0]["sizeBytes"], 0)
        self.assertTrue(checkpoints[0]["modifiedAt"].endswith("Z"))
        self.assertEqual(artifacts["runId"], run.id)
        self.assertEqual(
            artifacts["params"],
            {
                "batch_size": 4,
                "use_bias": True,
                "description": "baseline run",
                "learning_rate": 0.01,
                "optimizer": "adam",
            },
        )
        self.assertEqual(artifacts["metrics"], {"test/accuracy": 0.9})
        self.assertEqual(
            sorted({artifact["kind"] for artifact in artifacts["artifacts"]}),
            ["checkpoint", "event_file", "hparams", "result"],
        )
        self.assertEqual(
            len(
                [
                    artifact
                    for artifact in artifacts["artifacts"]
                    if artifact["kind"] == "checkpoint"
                ]
            ),
            3,
        )
        self.assertEqual(malformed_artifacts["params"], {})
        self.assertEqual(malformed_artifacts["metrics"], {})

    def test_log_run_index_deletes_experiment_tree(self) -> None:
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

            index = LogRunIndex(logs_root=logs_root)
            run_ids_by_path = {run.relativePath: run.id for run in index.list_runs()}
            result = index.delete_experiment("test_model")
            remaining_paths = {run.relativePath for run in index.list_runs()}

            self.assertEqual(result.experiment, "test_model")
            self.assertEqual(result.deletedRunCount, 2)
            self.assertEqual(result.deletedRelativePath, "test_model")
            self.assertEqual(
                set(result.deletedRunIds),
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

    def test_log_run_index_refuses_symlink_experiment_delete_and_preserves_target(
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

            with self.assertRaisesRegex(InspectorError, "symlink"):
                LogRunIndex(logs_root=logs_root).delete_experiment("linked")

            self.assertTrue(symlink_experiment.is_symlink())
            self.assertTrue(outside_experiment.exists())
            self.assertEqual(outside_marker.read_text(encoding="utf-8"), "outside")

    def test_log_run_index_filtered_delete_candidate_safety_matrix(self) -> None:
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
                "linear",
                "BASELINE",
                "Mnist",
                "aaa_20260601_010203",
                "version_0",
            )
            escaped_version.mkdir(parents=True)
            escaped_marker = escaped_version / "keep.txt"
            escaped_marker.write_text("escaped", encoding="utf-8")
            index = LogRunIndex(logs_root=logs_root)
            filters = delete_filters_for_runs(index.list_runs())
            result = index.delete_runs(filters, active_jobs=[])

            self.assertEqual(
                result.deletedRelativePaths,
                ["test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0"],
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
                    with self.assertRaisesRegex(InspectorError, error_pattern):
                        index._validated_delete_candidate_path(
                            self._delete_candidate(relative_path),
                            index._resolved_root(),
                        )
                    for preserved_path in preserved_paths:
                        self.assertTrue(preserved_path.exists())
            self.assertTrue(symlink_version.is_symlink())
            self.assertEqual(outside_marker.read_text(encoding="utf-8"), "outside")
            self.assertEqual(escaped_marker.read_text(encoding="utf-8"), "escaped")
            self.assertEqual(non_version_marker.read_text(encoding="utf-8"), "keep")

    def test_log_run_index_deletes_filtered_version_dirs_and_prunes_empty_parents(
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

            index = LogRunIndex(logs_root=logs_root)
            runs = index.list_runs()
            filters = delete_filters_for_runs(
                runs,
                experiments=["test_model"],
                datasets=["Mnist"],
                presets=["BASELINE"],
            )
            plan = index.create_delete_plan(filters, active_jobs=[])
            result = index.delete_runs(filters, active_jobs=[])
            remaining_paths = {run.relativePath for run in index.list_runs()}

            self.assertTrue(plan.canDelete)
            self.assertEqual(plan.to_response()["candidateCount"], 1)
            self.assertEqual(len(result.deletedRunIds), 1)
            self.assertEqual(
                result.deletedRelativePaths,
                [mnist_run.relative_to(logs_root).as_posix()],
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
                index.list_runs(),
                experiments=["test_model"],
                datasets=["Cifar10"],
                presets=["BASELINE"],
            )
            index.delete_runs(second_filters, active_jobs=[])
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

    def test_log_run_index_deletes_exact_run_id_filter_only(self) -> None:
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

            index = LogRunIndex(logs_root=logs_root)
            runs = index.list_runs()
            first_run_id = next(
                run.id
                for run in runs
                if run.relativePath == first_run.relative_to(logs_root).as_posix()
            )
            filters = delete_filters_for_runs(runs, run_ids=[first_run_id])
            result = index.delete_runs(filters, active_jobs=[])

            self.assertEqual(result.deletedRunIds, [first_run_id])
            self.assertFalse(first_run.exists())
            self.assertTrue(second_run.exists())

    def test_log_run_delete_partial_filters_match_nothing_and_preserve_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
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
            index = LogRunIndex(logs_root=logs_root)
            run = index.list_runs()[0]
            full_filter_fields = {
                "experiments": [run.experiment],
                "datasets": [run.dataset],
                "models": [run.model],
                "presets": [run.preset],
                "runIds": [run.id],
            }
            cases = (
                ("missing_experiments", {**full_filter_fields, "experiments": []}),
                ("missing_datasets", {**full_filter_fields, "datasets": []}),
                ("missing_models", {**full_filter_fields, "models": []}),
                ("missing_presets", {**full_filter_fields, "presets": []}),
                ("missing_run_ids", {**full_filter_fields, "runIds": []}),
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
                ("mismatched_run_id", {**full_filter_fields, "runIds": ["missing"]}),
            )

            for label, fields in cases:
                with self.subTest(label=label):
                    filters = LogRunDeleteFilters(**fields)
                    plan = index.create_delete_plan(filters, active_jobs=[])

                    self.assertFalse(plan.canDelete)
                    self.assertEqual(plan.candidates, [])
                    with self.assertRaisesRegex(InspectorError, "No log runs match"):
                        index.delete_runs(filters, active_jobs=[])
                    self.assertTrue(run_dir.exists())

    def test_log_run_delete_empty_filters_match_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
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

            index = LogRunIndex(logs_root=logs_root)
            filters = LogRunDeleteFilters(
                experiments=["test_model"],
                datasets=[],
                models=["linears/linear"],
                presets=["BASELINE"],
                runIds=[index.list_runs()[0].id],
            )
            plan = index.create_delete_plan(filters, active_jobs=[])

            self.assertFalse(plan.canDelete)
            self.assertEqual(plan.candidates, [])
            with self.assertRaisesRegex(InspectorError, "No log runs match"):
                index.delete_runs(filters, active_jobs=[])

    def test_log_run_index_prunes_only_empty_parents_under_logs_root(self) -> None:
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
            outside_parent = root / "outside_parent"
            outside_child = outside_parent / "empty_child"
            outside_child.mkdir(parents=True)
            index = LogRunIndex(logs_root=logs_root)

            index.delete_runs(
                delete_filters_for_runs(index.list_runs()),
                active_jobs=[],
            )
            index._prune_empty_run_parents(
                start=outside_child,
                experiment_dir=outside_parent,
                root=index._resolved_root(),
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
            self.assertTrue(outside_child.exists())

    def test_log_run_index_active_job_blocks_filtered_destructive_delete(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
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
            index = LogRunIndex(logs_root=logs_root)
            filters = delete_filters_for_runs(index.list_runs())
            active_jobs = [
                {
                    "id": "job-1",
                    "logFolder": "test_model",
                    "status": "running",
                }
            ]

            plan = index.create_delete_plan(filters, active_jobs=active_jobs)

            self.assertFalse(plan.canDelete)
            self.assertEqual(len(plan.blockedByActiveJobs), 1)
            self.assertEqual(plan.blockedByActiveJobs[0].logFolder, "test_model")
            with self.assertRaisesRegex(
                InspectorError,
                "training job is still writing",
            ):
                index.delete_runs(filters, active_jobs=active_jobs)
            self.assertTrue(run_dir.exists())

    def test_log_run_index_lists_safe_top_level_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            logs_root.mkdir()
            logs_root.joinpath("empty_experiment").mkdir()
            logs_root.joinpath("bad-name").mkdir()
            logs_root.joinpath("_bad_name").mkdir()
            outside_experiment = root / "outside_experiment"
            outside_experiment.mkdir()
            logs_root.joinpath("linked_experiment").symlink_to(
                outside_experiment,
                target_is_directory=True,
            )
            write_tensorboard_run(
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
            write_tensorboard_run(
                logs_root,
                [
                    "viewer-training",
                    "job-123",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )

            experiments = LogRunIndex(logs_root=logs_root).list_experiments()

        self.assertEqual(
            [experiment.experiment for experiment in experiments],
            ["empty_experiment", "test_model"],
        )
        by_name = {experiment.experiment: experiment for experiment in experiments}
        self.assertEqual(by_name["empty_experiment"].runCount, 0)
        self.assertEqual(by_name["test_model"].runCount, 1)
        self.assertEqual(by_name["test_model"].relativePath, "test_model")

    def test_log_run_index_rejects_invalid_delete_experiments(self) -> None:
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

            index = LogRunIndex(logs_root=logs_root)
            for experiment in (
                "",
                "../outside",
                "linear/BASELINE",
                ".",
                "..",
                "missing",
            ):
                with self.subTest(experiment=experiment):
                    with self.assertRaises(InspectorError):
                        index.delete_experiment(experiment)

            with self.assertRaisesRegex(InspectorError, "symlink"):
                index.delete_experiment("linked")

            self.assertTrue(logs_root.joinpath("linear").exists())
            self.assertTrue(outside_experiment.exists())

    def test_log_api_deletes_experiment_and_refreshes_runs(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
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
            write_tensorboard_run(
                logs_root,
                [
                    "test_model_2",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )

            async def call_api() -> tuple[
                httpx.Response, httpx.Response, httpx.Response
            ]:
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    before_response = await client.get("/logs/runs")
                    delete_response = await client.delete(
                        "/logs/experiments/test_model",
                    )
                    after_response = await client.get("/logs/runs")
                    return before_response, delete_response, after_response

            before_response, delete_response, after_response = asyncio.run(call_api())

            self.assertEqual(before_response.status_code, 200)
            self.assertEqual(delete_response.status_code, 200)
            self.assertFalse(logs_root.joinpath("test_model").exists())
            self.assertTrue(logs_root.joinpath("test_model_2").exists())

        delete_payload = delete_response.json()
        self.assertEqual(delete_payload["experiment"], "test_model")
        self.assertEqual(delete_payload["deletedRunCount"], 1)
        self.assertEqual(delete_payload["deletedRelativePath"], "test_model")
        self.assertEqual(len(delete_payload["deletedRunIds"]), 1)
        self.assertEqual(
            [run["experiment"] for run in after_response.json()["runs"]],
            ["test_model_2"],
        )

    def test_log_api_deletes_valid_empty_experiment_folder(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            logs_root.mkdir()
            logs_root.joinpath("new_empty").mkdir()

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.delete("/logs/experiments/new_empty")

            response = asyncio.run(call_api())

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "experiment": "new_empty",
                    "deletedRunIds": [],
                    "deletedRunCount": 0,
                    "deletedRelativePath": "new_empty",
                },
            )
            self.assertFalse(logs_root.joinpath("new_empty").exists())

    def test_log_api_blocks_experiment_delete_with_matching_active_job(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        cases = (
            (
                "matching_running_job",
                "test_model",
                "running",
                True,
            ),
            (
                "matching_queued_job",
                "test_model",
                "queued",
                True,
            ),
            (
                "matching_completed_job",
                "test_model",
                "completed",
                False,
            ),
            (
                "matching_failed_job",
                "test_model",
                "failed",
                False,
            ),
            (
                "matching_cancelled_job",
                "test_model",
                "cancelled",
                False,
            ),
            (
                "non_matching_running_job",
                "other_model",
                "running",
                False,
            ),
        )

        for label, job_log_folder, job_status, should_block in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmp:
                    logs_root = Path(tmp) / "logs"
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
                    manager = TrainingJobManager(
                        root=Path(tmp) / "jobs",
                        logs_root=logs_root,
                        runner=FakeRunner(),
                    )
                    job = manager.create_job(
                        model="linears/linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        log_folder=job_log_folder,
                        monitors=[],
                    )
                    job_id = str(job["id"])
                    if job_status == "cancelled":
                        manager.cancel_job(job_id)
                    elif job_status != "running":
                        manager.jobs[job_id].status = job_status

                    async def call_api(
                        logs_root: Path = logs_root,
                        manager: TrainingJobManager = manager,
                    ) -> httpx.Response:
                        transport = httpx.ASGITransport(
                            app=create_app(
                                ViewerApiSettings(logs_root=str(logs_root)),
                                training_manager=manager,
                            )
                        )
                        async with httpx.AsyncClient(
                            transport=transport,
                            base_url="http://testserver",
                        ) as client:
                            return await client.delete("/logs/experiments/test_model")

                    response = asyncio.run(call_api())

                    if should_block:
                        self.assertEqual(response.status_code, 400)
                        self.assertEqual(
                            response.json(),
                            {
                                "detail": (
                                    "A training job is still writing to this "
                                    "log folder."
                                )
                            },
                        )
                        self.assertTrue(logs_root.joinpath("test_model").exists())
                        self.assertTrue(run_dir.exists())
                    else:
                        self.assertEqual(response.status_code, 200)
                        payload = response.json()
                        self.assertFalse(logs_root.joinpath("test_model").exists())
                        self.assertFalse(run_dir.exists())
                        if job_log_folder != "test_model":
                            self.assertTrue(logs_root.joinpath(job_log_folder).exists())
                        self.assertEqual(
                            payload,
                            {
                                "experiment": "test_model",
                                "deletedRunIds": payload["deletedRunIds"],
                                "deletedRunCount": 1,
                                "deletedRelativePath": "test_model",
                            },
                        )
                        self.assertEqual(len(payload["deletedRunIds"]), 1)

                    self.assertEqual(
                        manager.jobs[job_id].status,
                        job_status,
                    )

    def test_log_api_restart_behavior_fresh_manager_preserves_active_delete_blocker(
        self,
    ) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

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
            original_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            job = original_manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )
            job_id = str(job["id"])
            self.assertEqual(
                original_manager.active_jobs(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )

            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            self.assertEqual(
                fresh_manager.active_jobs(),
                [
                    {
                        "id": job_id,
                        "status": "unknown",
                        "logFolder": "test_model",
                    }
                ],
            )

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(logs_root=str(logs_root)),
                        training_manager=fresh_manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.delete("/logs/experiments/test_model")

            response = asyncio.run(call_api())

            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.json(),
                {"detail": ("A training job is still writing to this log folder.")},
            )
            self.assertTrue(logs_root.joinpath("test_model").exists())
            self.assertTrue(run_dir.exists())
            self.assertEqual(
                original_manager.active_jobs(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )

    def test_restart_fresh_manager_preserves_unknown_run_delete_blocker(
        self,
    ) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

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
            original_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            job = original_manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )
            job_id = str(job["id"])
            self.assertEqual(
                original_manager.active_jobs(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )
            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            filters = {
                "experiments": [run.experiment],
                "datasets": [run.dataset],
                "models": [run.model],
                "presets": [run.preset],
                "runIds": [run.id],
            }

            async def create_plan() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(logs_root=str(logs_root)),
                        training_manager=fresh_manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/logs/runs/delete-plan",
                        json=filters,
                    )

            plan_response = asyncio.run(create_plan())

            self.assertEqual(plan_response.status_code, 200, plan_response.text)
            plan_payload = plan_response.json()
            self.assertFalse(plan_payload["canDelete"])
            self.assertEqual(
                plan_payload["blockedByActiveJobs"],
                [
                    {
                        "id": job_id,
                        "logFolder": "test_model",
                        "status": "unknown",
                    }
                ],
            )

            async def delete_runs() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(logs_root=str(logs_root)),
                        training_manager=fresh_manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/logs/runs/delete",
                        json=filters,
                    )

            delete_response = asyncio.run(delete_runs())

            self.assertEqual(delete_response.status_code, 400)
            self.assertIn(
                "A training job is still writing to this log folder.",
                delete_response.text,
            )
            self.assertTrue(run_dir.exists())

    def test_restart_fresh_manager_blocks_experiment_delete_for_unknown_job(
        self,
    ) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

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
            original_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            job = original_manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )
            job_id = str(job["id"])
            self.assertEqual(
                original_manager.active_jobs(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )
            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(logs_root=str(logs_root)),
                        training_manager=fresh_manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.delete("/logs/experiments/test_model")

            response = asyncio.run(call_api())

            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.json(),
                {"detail": ("A training job is still writing to this log folder.")},
            )
            self.assertTrue(logs_root.joinpath("test_model").exists())
            self.assertTrue(run_dir.exists())

    def test_log_api_plans_and_deletes_filtered_runs_with_active_job_guard(
        self,
    ) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
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
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(logs_root=str(logs_root)),
                        training_manager=manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    runs_response = await client.get("/logs/runs")
                    run = runs_response.json()["runs"][0]
                    filters = {
                        "experiments": [run["experiment"]],
                        "datasets": [run["dataset"]],
                        "models": [run["model"]],
                        "presets": [run["preset"]],
                        "runIds": [run["id"]],
                    }
                    plan_response = await client.post(
                        "/logs/runs/delete-plan",
                        json=filters,
                    )
                    delete_response = await client.post(
                        "/logs/runs/delete",
                        json=filters,
                    )
                    return plan_response, delete_response

            plan_response, delete_response = asyncio.run(call_api())

            self.assertEqual(plan_response.status_code, 200)
            plan_payload = plan_response.json()
            self.assertEqual(plan_payload["candidateCount"], 1)
            self.assertFalse(plan_payload["canDelete"])
            self.assertEqual(
                plan_payload["blockedByActiveJobs"][0]["logFolder"],
                "test_model",
            )
            self.assertEqual(delete_response.status_code, 400)
            self.assertIn(
                "A training job is still writing to this log folder.",
                delete_response.text,
            )
            self.assertTrue(run_dir.exists())

    def test_log_api_lists_experiments_including_empty_folders(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            logs_root.mkdir()
            logs_root.joinpath("new_empty").mkdir()
            logs_root.joinpath("bad-name").mkdir()
            write_tensorboard_run(
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

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.get("/logs/experiments")

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["experiments"],
            [
                {
                    "experiment": "new_empty",
                    "runCount": 0,
                    "relativePath": "new_empty",
                },
                {
                    "experiment": "test_model",
                    "runCount": 1,
                    "relativePath": "test_model",
                },
            ],
        )

    def test_log_api_paginates_unbounded_list_endpoints(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for index, experiment in enumerate(("exp_a", "exp_b", "exp_c"), start=1):
                write_tensorboard_run(
                    logs_root,
                    [
                        experiment,
                        "linear",
                        "BASELINE",
                        "Mnist",
                        f"run_{index}_2026060{index}_010203",
                        "version_0",
                    ],
                )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    runs_response = await client.get(
                        "/logs/runs",
                        params={"limit": 2, "offset": 1},
                    )
                    experiments_response = await client.get(
                        "/logs/experiments",
                        params={"limit": 1, "offset": 1},
                    )
                    return runs_response, experiments_response

            runs_response, experiments_response = asyncio.run(call_api())

        self.assertEqual(runs_response.status_code, 200)
        runs_payload = runs_response.json()
        self.assertEqual(runs_payload["total"], 3)
        self.assertEqual(runs_payload["limit"], 2)
        self.assertEqual(runs_payload["offset"], 1)
        self.assertFalse(runs_payload["hasMore"])
        self.assertEqual(len(runs_payload["runs"]), 2)
        self.assertEqual(
            [run["experiment"] for run in runs_payload["runs"]],
            ["exp_b", "exp_a"],
        )

        self.assertEqual(experiments_response.status_code, 200)
        experiments_payload = experiments_response.json()
        self.assertEqual(experiments_payload["total"], 3)
        self.assertEqual(experiments_payload["limit"], 1)
        self.assertEqual(experiments_payload["offset"], 1)
        self.assertTrue(experiments_payload["hasMore"])
        self.assertEqual(len(experiments_payload["experiments"]), 1)
        self.assertEqual(
            [
                experiment["experiment"]
                for experiment in experiments_payload["experiments"]
            ],
            ["exp_b"],
        )

    def test_log_api_reads_tags_scalars_and_rejects_unknown_run_ids(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
                scalars={
                    "train/loss": [(1, 0.5), (2, 0.25)],
                    "validation/accuracy": [(2, 0.75)],
                },
                metrics={"test/accuracy": 0.9},
            )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model_2",
                    "linear_adaptive",
                    "DUAL_MODEL_WEIGHT",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
                metrics=None,
                hparams=False,
                checkpoint=False,
            )
            no_event_run = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Mnist",
                "no_events_20260601_040506",
                "version_0",
            )
            no_event_run.mkdir(parents=True)
            malformed_result_run = write_tensorboard_run(
                logs_root,
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "malformed_20260601_050607",
                    "version_0",
                ],
                metrics=None,
            )
            (malformed_result_run / "result.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )

            async def call_api() -> tuple[
                httpx.Response,
                httpx.Response,
                httpx.Response,
                httpx.Response,
                httpx.Response,
            ]:
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    runs_response = await client.get("/logs/runs")
                    run_id = next(
                        run["id"]
                        for run in runs_response.json()["runs"]
                        if run["relativePath"]
                        == "linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
                    )
                    tags_response = await client.post(
                        "/logs/tags",
                        json={"runIds": [run_id]},
                    )
                    scalars_response = await client.post(
                        "/logs/scalars",
                        json={"runIds": [run_id], "tags": ["train/loss"]},
                    )
                    unknown_response = await client.post(
                        "/logs/tags",
                        json={"runIds": ["not-a-run"]},
                    )
                    raw_path_response = await client.post(
                        "/logs/scalars",
                        json={
                            "runIds": [
                                "linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
                            ],
                            "tags": ["train/loss"],
                        },
                    )
                    return (
                        runs_response,
                        tags_response,
                        scalars_response,
                        unknown_response,
                        raw_path_response,
                    )

            (
                runs_response,
                tags_response,
                scalars_response,
                unknown_response,
                raw_path_response,
            ) = asyncio.run(call_api())

        self.assertEqual(runs_response.status_code, 200)
        runs_payload = runs_response.json()["runs"]
        by_path = {run["relativePath"]: run for run in runs_payload}
        run_payload = by_path["linear/BASELINE/Mnist/aaa_20260601_010203/version_0"]
        incomplete_payload = by_path[
            "test_model_2/linear_adaptive/DUAL_MODEL_WEIGHT/Cifar10/"
            "bbb_20260601_020304/version_0"
        ]
        no_event_payload = by_path[
            "linear/BASELINE/Mnist/no_events_20260601_040506/version_0"
        ]
        malformed_result_payload = by_path[
            "linear/BASELINE/Mnist/malformed_20260601_050607/version_0"
        ]
        self.assertEqual(run_payload["experiment"], "linear")
        self.assertEqual(run_payload["dataset"], "Mnist")
        self.assertTrue(run_payload["hasResult"])
        self.assertGreater(run_payload["eventFileCount"], 0)
        self.assertEqual(run_payload["metrics"]["test/accuracy"], 0.9)
        self.assertEqual(incomplete_payload["experiment"], "test_model_2")
        self.assertFalse(incomplete_payload["hasResult"])
        self.assertFalse(no_event_payload["hasResult"])
        self.assertEqual(no_event_payload["eventFileCount"], 0)
        self.assertEqual(no_event_payload["metrics"], {})
        self.assertTrue(malformed_result_payload["hasResult"])
        self.assertEqual(malformed_result_payload["metrics"], {})

        self.assertEqual(tags_response.status_code, 200)
        self.assertEqual(
            set(tags_response.json()["runs"][0]["scalarTags"]),
            {"train/loss", "validation/accuracy"},
        )

        self.assertEqual(scalars_response.status_code, 200)
        series = scalars_response.json()["series"][0]
        self.assertEqual(series["tag"], "train/loss")
        self.assertEqual([point["step"] for point in series["points"]], [1, 2])
        self.assertEqual(series["points"][1]["value"], 0.25)

        self.assertEqual(unknown_response.status_code, 400)
        self.assertIn("Unknown log run id", unknown_response.json()["detail"])
        self.assertEqual(raw_path_response.status_code, 400)
        self.assertIn("Unknown log run id", raw_path_response.json()["detail"])

    def test_log_run_index_reads_tensorboard_media_summaries(self) -> None:
        import torch
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Cifar10",
                "diagnostics_20260613_120000",
                "version_0",
            )
            writer = SummaryWriter(log_dir=str(run_dir))
            writer.add_scalar("gap/accuracy", 0.12, 1)
            writer.add_image(
                "validation/examples/predictions",
                torch.zeros(3, 8, 8),
                1,
            )
            writer.add_text(
                "validation/examples/predictions",
                "true=cat predicted=dog confidence=0.91",
                1,
            )
            writer.flush()
            writer.close()

            index = LogRunIndex(logs_root=logs_root)
            run = index.list_runs()[0]
            tags = index.tags_for_runs([run.id])[0]
            media = index.media_for_runs(
                run_ids=[run.id],
                image_tags=["validation/examples/predictions"],
                text_tags=["validation/examples/predictions/text_summary"],
            )

        self.assertEqual(tags["scalarTags"], ["gap/accuracy"])
        self.assertEqual(tags["imageTags"], ["validation/examples/predictions"])
        self.assertEqual(
            tags["textTags"],
            ["validation/examples/predictions/text_summary"],
        )
        self.assertEqual(media["images"][0]["runId"], run.id)
        self.assertTrue(
            media["images"][0]["dataUrl"].startswith("data:image/png;base64,")
        )
        self.assertEqual(media["texts"][0]["runId"], run.id)
        self.assertIn("true=cat", media["texts"][0]["text"])

    def test_log_run_query_service_reads_latest_summary_across_event_dirs(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            early_writer = SummaryWriter(log_dir=str(run_dir / "early"))
            early_writer.add_text("validation/examples/predictions", "early", 1)
            early_writer.flush()
            early_writer.close()
            late_writer = SummaryWriter(log_dir=str(run_dir / "late"))
            late_writer.add_text("validation/examples/predictions", "late", 2)
            late_writer.flush()
            late_writer.close()

            index = LogRunIndex(logs_root=Path(tmp))
            summary = index.query_service.read_text_summary(
                run_dir,
                "validation/examples/predictions/text_summary",
            )

        self.assertIsNotNone(summary)
        self.assertEqual(summary["step"], 2)
        self.assertEqual(summary["text"], "late")


if __name__ == "__main__":
    unittest.main()
