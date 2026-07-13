from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
import uuid
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from emperor.model_packages import model_identity_payload_from_id

from workbench.backend.api.v1.logs_mapping import (
    LOG_METADATA_RESPONSE_LIMIT,
    log_run_artifacts_to_payload,
    log_run_delete_plan_to_payload,
    log_run_delete_result_to_payload,
)
from workbench.backend.failures import FailureKind
from workbench.backend.log_experiments import (
    LOG_EXPERIMENT_NAME_RE,
    LogExperimentFailure,
    LogExperimentMutationCoordinator,
    is_valid_log_experiment_name,
    validate_log_experiment_name,
)
from workbench.backend.run_history import RunHistoryService
from workbench.backend.run_history import artifacts as run_artifacts
from workbench.backend.run_history.artifacts import (
    RunArtifactBudgets,
    observe_run_artifacts,
)
from workbench.backend.run_history.deletion import LogRunDeletionExecutor
from workbench.backend.run_history.errors import RunHistoryFailure
from workbench.backend.run_history.query import LogRunQueryService
from workbench.backend.run_history.records import (
    ActiveLogRunDeleteBlocker,
    LogRunDeleteCandidate,
    LogRunDeleteFilters,
    LogRunDeletePlan,
    LogRunDeleteResult,
)
from workbench.backend.run_history.scanner import LogRunScanner
from workbench.backend.tensorboard import events as tensorboard_events
from workbench.backend.tests.helpers import (
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
    delete_filters_for_runs,
    write_tensorboard_run,
)


@dataclass(frozen=True, slots=True)
class _ActiveWriter:
    id: str
    status: str
    log_folder: str


def _run_history(
    logs_root: Path,
    *,
    active_writers: list[_ActiveWriter] | None = None,
) -> RunHistoryService:
    writers = active_writers or []
    return RunHistoryService(
        logs_root=logs_root,
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: list(writers),
    )


def _delete_plan(
    service: RunHistoryService,
    filters: LogRunDeleteFilters,
) -> LogRunDeletePlan:
    return service.create_delete_plan(
        experiments=list(filters.experiments),
        datasets=list(filters.datasets),
        models=list(filters.models),
        presets=list(filters.presets),
        run_ids=list(filters.run_ids),
    )


def _delete_runs(
    service: RunHistoryService,
    filters: LogRunDeleteFilters,
) -> LogRunDeleteResult:
    return service.delete_runs(
        experiments=list(filters.experiments),
        datasets=list(filters.datasets),
        models=list(filters.models),
        presets=list(filters.presets),
        run_ids=list(filters.run_ids),
    )


class FakeScalarEvent:
    def __init__(self, step: int, value: float, wall_time: float | None = None) -> None:
        self.step = step
        self.value = value
        self.wall_time = float(step if wall_time is None else wall_time)


class FakeTensorBoardAccumulator:
    def Tags(self) -> dict[str, list[str]]:
        return {
            "scalars": ["train/loss"],
            "histograms": ["weights"],
            "images": ["validation/examples/predictions"],
            "tensors": ["validation/examples/predictions/text_summary"],
        }

    def Scalars(self, tag: str) -> list[FakeScalarEvent]:
        if tag != "train/loss":
            raise KeyError(tag)
        return [
            FakeScalarEvent(step=1, value=0.5),
            FakeScalarEvent(step=2, value=0.25),
            FakeScalarEvent(step=3, value=0.125),
        ]


def write_fake_event_run(
    logs_root: Path,
    run_name: str,
    payload: bytes,
) -> Path:
    run_dir = logs_root.joinpath(
        "linear",
        "BASELINE",
        "Mnist",
        run_name,
        "version_0",
    )
    run_dir.mkdir(parents=True)
    (run_dir / "events.out.tfevents.fake").write_bytes(payload)
    return run_dir


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
                with self.assertRaises(LogExperimentFailure):
                    validate_log_experiment_name(name)


class LogRunDeleteHttpMappingTests(unittest.TestCase):
    def test_delete_plan_and_result_response_payloads_are_stable(self) -> None:
        candidate = LogRunDeleteCandidate(
            id="run-1",
            experiment="test_model",
            model="linears/linear",
            preset="BASELINE",
            dataset="Mnist",
            run_name="aaa_20260601_010203",
            version="version_0",
            relative_path=(
                "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
            ),
        )
        blocker = ActiveLogRunDeleteBlocker(
            id="job-1",
            log_folder="test_model",
            status="running",
        )
        expected_common = {
            "candidateCount": 1,
            "sourceItemCount": 1,
            "returnedItemCount": 1,
            "truncated": False,
            "truncationReason": None,
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
                "models": [{"modelType": "linears", "model": "linear"}],
                "presets": ["BASELINE"],
                "runIds": ["run-1"],
            },
            "candidates": [
                {
                    "id": "run-1",
                    "experiment": "test_model",
                    "modelType": "linears",
                    "model": "linear",
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

        plan_payload = log_run_delete_plan_to_payload(
            LogRunDeletePlan(
                candidates=(candidate,),
                blocked_by_active_jobs=(blocker,),
            )
        )
        result_payload = log_run_delete_result_to_payload(
            LogRunDeleteResult(
                candidates=(candidate,),
                deleted_run_ids=("run-1",),
                deleted_relative_paths=(candidate.relative_path,),
            )
        )

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
                "deletedRelativePaths": [candidate.relative_path],
                **expected_common,
                "blockedByActiveJobs": [],
                "canDelete": True,
            },
        )

    def test_delete_plan_response_caps_candidate_preview(self) -> None:
        candidates = [
            LogRunDeleteCandidate(
                id=f"run-{index}",
                experiment="test_model",
                model="linears/linear",
                preset="BASELINE",
                dataset="Mnist",
                run_name=f"run_{index:06d}_20260601_010203",
                version="version_0",
                relative_path=(
                    "test_model/linear/BASELINE/Mnist/"
                    f"run_{index:06d}_20260601_010203/version_0"
                ),
            )
            for index in range(LOG_METADATA_RESPONSE_LIMIT + 3)
        ]

        payload = log_run_delete_plan_to_payload(
            LogRunDeletePlan(candidates=tuple(candidates))
        )

        self.assertEqual(payload["candidateCount"], LOG_METADATA_RESPONSE_LIMIT + 3)
        self.assertEqual(payload["sourceItemCount"], LOG_METADATA_RESPONSE_LIMIT + 3)
        self.assertEqual(payload["returnedItemCount"], LOG_METADATA_RESPONSE_LIMIT)
        self.assertEqual(len(payload["candidates"]), LOG_METADATA_RESPONSE_LIMIT)
        self.assertTrue(payload["truncated"])
        self.assertIn("capped", payload["truncationReason"])


class RunHistoryAndApiTests(unittest.TestCase):
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

    def test_catalog_generation_exposes_external_run_before_cache_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            service = _run_history(logs_root)
            self.assertEqual(service.list_runs(limit=10, offset=0).total, 0)

            write_tensorboard_run(
                logs_root,
                [
                    "experiment",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "completed_20260601_010203",
                    "version_0",
                ],
            )

            refreshed = service.list_runs(limit=10, offset=0)

        self.assertEqual(refreshed.total, 1)
        self.assertEqual(refreshed.runs[0].experiment, "experiment")

    def test_run_history_parses_supported_log_shapes(self) -> None:
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
            workbench_run = write_tensorboard_run(
                logs_root,
                [
                    "workbench-training",
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

            runs = LogRunScanner(logs_root=logs_root).list_runs()

        by_path = {run.relative_path: run for run in runs}
        default_summary = by_path[default_run.relative_to(logs_root).as_posix()]
        categorized_summary = by_path[categorized_run.relative_to(logs_root).as_posix()]
        custom_summary = by_path[custom_run.relative_to(logs_root).as_posix()]
        workbench_summary = by_path[workbench_run.relative_to(logs_root).as_posix()]
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
        self.assertTrue(default_summary.has_result)
        self.assertGreater(default_summary.event_file_count, 0)
        self.assertEqual(default_summary.checkpoint_count, 1)
        self.assertTrue(default_summary.has_hparams)
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
        self.assertFalse(custom_summary.has_result)
        self.assertFalse(custom_summary.has_hparams)
        self.assertEqual(custom_summary.checkpoint_count, 0)

        self.assertEqual(workbench_summary.group, "workbench-training/job-123")
        self.assertEqual(workbench_summary.experiment, "workbench-training")
        self.assertEqual(workbench_summary.model, "linears/linear")
        self.assertEqual(workbench_summary.dataset, "FashionMNIST")

        self.assertFalse(no_event_summary.has_result)
        self.assertEqual(no_event_summary.event_file_count, 0)
        self.assertEqual(no_event_summary.checkpoint_count, 0)
        self.assertFalse(no_event_summary.has_hparams)
        self.assertEqual(no_event_summary.metrics, {})

        self.assertTrue(malformed_result_summary.has_result)
        self.assertGreater(malformed_result_summary.event_file_count, 0)
        self.assertEqual(malformed_result_summary.metrics, {})
        self.assertNotIn(
            "linear/BASELINE/Mnist/escaped_20260601_070809/version_99",
            by_path,
        )

    def test_log_run_scanner_reuses_recent_run_parse_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
            )
            write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "bbb_20260601_020304", "version_0"],
            )
            scanner = LogRunScanner(logs_root=logs_root, cache_ttl_seconds=60)

            with patch.object(scanner, "parse_run", wraps=scanner.parse_run) as parse:
                first = scanner.list_runs()
                second = scanner.list_runs()
                scanner.clear_cache()
                third = scanner.list_runs()

        self.assertEqual([run.id for run in first], [run.id for run in second])
        self.assertEqual([run.id for run in first], [run.id for run in third])
        self.assertEqual(parse.call_count, 4)

    def test_log_run_scanner_reuses_expired_cache_when_catalog_is_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
            )
            scanner = LogRunScanner(logs_root=logs_root, cache_ttl_seconds=0)

            with patch.object(scanner, "parse_run", wraps=scanner.parse_run) as parse:
                first = scanner.list_runs()
                second = scanner.list_runs()
                (run_dir / "result.json").write_text(
                    json.dumps({"metrics": {"accuracy": 0.95}}),
                    encoding="utf-8",
                )
                third = scanner.list_runs()

        self.assertEqual([run.id for run in first], [run.id for run in second])
        self.assertEqual([run.id for run in first], [run.id for run in third])
        self.assertEqual(parse.call_count, 2)

    def test_expired_catalog_detects_new_nested_event_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Mnist",
                "nested_20260601_010203",
                "version_0",
            )
            nested = run_dir / "already-present"
            nested.mkdir(parents=True)
            scanner = LogRunScanner(logs_root=logs_root, cache_ttl_seconds=0)

            initial = scanner.list_runs()[0]
            nested.joinpath("events.out.tfevents.new").write_bytes(b"event")
            nested.joinpath("epoch=1-step=2.ckpt").write_bytes(b"checkpoint")
            changed = scanner.list_runs()[0]

        self.assertEqual(initial.event_file_count, 0)
        self.assertEqual(initial.checkpoint_count, 0)
        self.assertEqual(changed.event_file_count, 1)
        self.assertEqual(changed.checkpoint_count, 1)

    def test_summary_listing_parses_result_once_without_projecting_metrics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
            )
            run_dir.joinpath("result.json").write_text(
                json.dumps(
                    {
                        "experimentTask": "image_classification",
                        "metrics": {"test/accuracy": 0.9},
                    }
                ),
                encoding="utf-8",
            )
            service = _run_history(logs_root)

            with (
                patch.object(
                    run_artifacts,
                    "_read_result_payload",
                    wraps=run_artifacts._read_result_payload,
                ) as read_result,
                patch.object(
                    run_artifacts.RunArtifactObservation,
                    "metrics",
                    side_effect=AssertionError(
                        "summary projection must not materialize metrics"
                    ),
                ) as project_metrics,
            ):
                run = service.list_runs(
                    limit=10,
                    offset=0,
                    projection="summary",
                ).runs[0]

        self.assertEqual(run.experiment_task, "image_classification")
        self.assertEqual(run.metrics, {})
        self.assertEqual(read_result.call_count, 1)
        project_metrics.assert_not_called()

    def test_run_artifact_observation_reuses_one_bounded_metadata_parse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "version_0"
            run_dir.mkdir()
            run_dir.joinpath("result.json").write_text(
                json.dumps(
                    {
                        "experimentTask": "image_classification",
                        "metrics": {"accuracy": 0.9},
                        "params": {"batch_size": 8},
                    }
                ),
                encoding="utf-8",
            )
            observation = observe_run_artifacts(run_dir, root)

            with patch.object(
                run_artifacts,
                "_read_result_payload",
                wraps=run_artifacts._read_result_payload,
            ) as read_result:
                experiment_task = observation.experiment_task()
                metrics = observation.metrics()
                params = observation.params()

        self.assertEqual(experiment_task, "image_classification")
        self.assertEqual(metrics, {"accuracy": 0.9})
        self.assertEqual(params, {"batch_size": 8})
        read_result.assert_called_once()

    def test_run_artifacts_cannot_resolve_into_a_sibling_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            first_run = root / "experiment" / "first"
            second_run = root / "experiment" / "second"
            first_run.mkdir(parents=True)
            second_run.mkdir(parents=True)
            second_run.joinpath("result.json").write_text(
                json.dumps({"metrics": {"secret": 1.0}}),
                encoding="utf-8",
            )
            second_run.joinpath("secret.ckpt").write_bytes(b"secret")
            first_run.joinpath("result.json").symlink_to(second_run / "result.json")
            first_run.joinpath("stolen.ckpt").symlink_to(second_run / "secret.ckpt")

            observation = observe_run_artifacts(first_run, root)

        self.assertEqual(observation.metrics(), {})
        self.assertIsNone(observation.result)
        self.assertEqual(observation.checkpoints, ())

    def test_run_artifact_in_run_symlink_is_allowed_but_replacement_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            run_dir = root / "experiment" / "run"
            run_dir.mkdir(parents=True)
            target = run_dir / "contained-result.json"
            target.write_text(
                json.dumps({"metrics": {"accuracy": 0.9}}),
                encoding="utf-8",
            )
            result = run_dir / "result.json"
            result.symlink_to(target)

            allowed = observe_run_artifacts(run_dir, root)
            self.assertEqual(allowed.metrics(), {"accuracy": 0.9})

            result.unlink()
            result.write_text(
                json.dumps({"metrics": {"beforeSwap": 1.0}}),
                encoding="utf-8",
            )
            replaced = observe_run_artifacts(run_dir, root)
            outside = Path(tmp) / "outside-result.json"
            outside.write_text(
                json.dumps({"metrics": {"secret": 1.0}}),
                encoding="utf-8",
            )
            result.unlink()
            result.symlink_to(outside)

            self.assertEqual(replaced.metrics(), {})

    def test_run_artifact_observation_enforces_file_depth_and_byte_budgets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "version_0"
            deep = run_dir / "level-1" / "level-2"
            deep.mkdir(parents=True)
            run_dir.joinpath("result.json").write_text(
                json.dumps({"metrics": {"accuracy": 0.9}}),
                encoding="utf-8",
            )
            for name in ("a.ckpt", "b.ckpt", "c.ckpt"):
                run_dir.joinpath(name).write_bytes(b"checkpoint")
            deep.joinpath("deep.ckpt").write_bytes(b"checkpoint")

            observation = observe_run_artifacts(
                run_dir,
                root,
                budgets=RunArtifactBudgets(
                    max_files=5,
                    max_depth=1,
                    max_metadata_file_bytes=8,
                ),
            )
            depth_limited = observe_run_artifacts(
                run_dir,
                root,
                budgets=RunArtifactBudgets(
                    max_files=100,
                    max_depth=1,
                    max_metadata_file_bytes=1024,
                ),
            )

        self.assertLessEqual(observation.observed_entry_count, 5)
        self.assertLessEqual(len(observation.checkpoints), 3)
        self.assertEqual(observation.metrics(), {})
        self.assertTrue(observation.truncated)
        self.assertTrue(
            any("item cap" in reason for reason in observation.truncation_reasons)
        )
        self.assertTrue(
            any("byte cap" in reason for reason in observation.truncation_reasons)
        )
        self.assertNotIn(
            "deep.ckpt",
            {artifact.path.name for artifact in depth_limited.checkpoints},
        )
        self.assertTrue(
            any(
                "recursion cap" in reason for reason in depth_limited.truncation_reasons
            )
        )

    def test_incomplete_event_observation_refuses_tensorboard_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            run_dir = root / "experiment" / "run"
            run_dir.mkdir(parents=True)
            run_dir.joinpath("events.out.tfevents.safe").write_bytes(b"safe")
            omitted = run_dir / "z-omitted"
            omitted.mkdir()
            omitted.joinpath("events.out.tfevents.hidden").write_bytes(b"hidden")

            observation = observe_run_artifacts(
                run_dir,
                root,
                budgets=RunArtifactBudgets(
                    max_files=1,
                    max_depth=16,
                    max_metadata_file_bytes=1024,
                ),
            )
            with patch.object(
                tensorboard_events,
                "load_event_accumulator",
                return_value=FakeTensorBoardAccumulator(),
            ) as load:
                accumulator = observation.event_files.load_accumulator(run_dir)

        self.assertTrue(observation.truncated)
        self.assertFalse(observation.event_files.complete)
        self.assertIsNone(accumulator)
        load.assert_not_called()

    def test_event_file_replacement_after_observation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            run_dir = root / "experiment" / "run"
            run_dir.mkdir(parents=True)
            event_path = run_dir / "events.out.tfevents.safe"
            event_path.write_bytes(b"safe")
            observation = observe_run_artifacts(run_dir, root)
            outside = Path(tmp) / "events.out.tfevents.outside"
            outside.write_bytes(b"secret")
            event_path.unlink()
            event_path.symlink_to(outside)

            with patch.object(
                tensorboard_events,
                "load_event_accumulator",
                return_value=FakeTensorBoardAccumulator(),
            ) as load:
                accumulator = observation.event_files.load_accumulator(run_dir)

        self.assertIsNone(accumulator)
        load.assert_not_called()

    def test_run_artifact_observation_composes_shared_event_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "version_0"
            nested = run_dir / "nested"
            nested.mkdir(parents=True)
            event_file = nested / "events.out.tfevents.test"
            event_file.write_bytes(b"event")

            with patch.object(
                tensorboard_events,
                "event_file_index",
                wraps=tensorboard_events.event_file_index,
            ) as observe_events:
                observation = observe_run_artifacts(run_dir, root)

        observe_events.assert_called_once_with(
            run_dir,
            candidates=(event_file,),
            complete=True,
        )
        self.assertEqual(observation.event_files.files, (event_file,))
        self.assertEqual(
            tuple(artifact.path for artifact in observation.event_artifacts),
            (event_file,),
        )

    def test_run_history_tensorboard_query_reuses_catalog_event_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Mnist",
                "aaa_20260601_010203",
                "version_0",
            )
            run_dir.mkdir(parents=True)
            run_dir.joinpath("events.out.tfevents.test").write_bytes(b"event")
            scanner = LogRunScanner(logs_root=logs_root)
            query = LogRunQueryService(scanner=scanner)

            with (
                patch.object(
                    tensorboard_events,
                    "event_file_index",
                    wraps=tensorboard_events.event_file_index,
                ) as observe_events,
                patch.object(
                    tensorboard_events,
                    "load_event_accumulator",
                    return_value=FakeTensorBoardAccumulator(),
                ),
            ):
                run_id = scanner.list_runs(result_projection="none")[0].id
                tags = query.tags_for_runs([run_id])

        self.assertEqual(observe_events.call_count, 1)
        self.assertEqual(tags[0]["scalarTags"], ["train/loss"])

    def test_run_history_reads_checkpoints_and_artifacts(self) -> None:
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

            scanner = LogRunScanner(logs_root=logs_root)
            query = LogRunQueryService(scanner=scanner)
            runs_by_path = {run.relative_path: run for run in scanner.list_runs()}
            run = runs_by_path["linear/BASELINE/Mnist/aaa_20260601_010203/version_0"]
            malformed = runs_by_path[
                "linear/BASELINE/Mnist/malformed_20260601_050607/version_0"
            ]

            checkpoints = query.checkpoints_for_runs([run.id])
            artifacts = query.artifacts_for_run(run.id)
            malformed_artifacts = query.artifacts_for_run(malformed.id)
            with self.assertRaises(RunHistoryFailure):
                query.checkpoints_for_runs(["not-a-run"])
            with self.assertRaises(RunHistoryFailure):
                query.artifacts_for_run("not-a-run")

        self.assertEqual(
            [
                (
                    checkpoint.filename,
                    checkpoint.epoch,
                    checkpoint.step,
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
            checkpoints[0].relative_path.endswith(
                "linear/BASELINE/Mnist/aaa_20260601_010203/version_0/"
                "checkpoints/epoch=0-step=1.ckpt"
            )
        )
        self.assertGreater(checkpoints[0].size_bytes, 0)
        self.assertTrue(checkpoints[0].modified_at.endswith("Z"))
        self.assertEqual(artifacts.run_id, run.id)
        self.assertEqual(
            artifacts.params,
            {
                "batch_size": 4,
                "use_bias": True,
                "description": "baseline run",
                "learning_rate": 0.01,
                "optimizer": "adam",
            },
        )
        self.assertEqual(artifacts.metrics, {"test/accuracy": 0.9})
        self.assertEqual(
            sorted({artifact.kind for artifact in artifacts.artifacts}),
            ["checkpoint", "event_file", "hparams", "result"],
        )
        self.assertEqual(
            len(
                [
                    artifact
                    for artifact in artifacts.artifacts
                    if artifact.kind == "checkpoint"
                ]
            ),
            3,
        )
        self.assertEqual(malformed_artifacts.params, {})
        self.assertEqual(malformed_artifacts.metrics, {})

    def test_run_history_caps_artifact_metadata_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            run_dir = write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
            )
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            for index in range(LOG_METADATA_RESPONSE_LIMIT + 10):
                (checkpoint_dir / f"epoch=0-step={index}.ckpt").write_text(
                    "checkpoint",
                    encoding="utf-8",
                )

            scanner = LogRunScanner(logs_root=logs_root)
            query = LogRunQueryService(scanner=scanner)
            run = scanner.list_runs()[0]
            payload = log_run_artifacts_to_payload(query.artifacts_for_run(run.id))

        returned_count = len(payload["artifacts"]) + len(payload["checkpoints"])
        self.assertEqual(returned_count, LOG_METADATA_RESPONSE_LIMIT)
        self.assertGreater(payload["sourceItemCount"], payload["returnedItemCount"])
        self.assertEqual(payload["returnedItemCount"], LOG_METADATA_RESPONSE_LIMIT)
        self.assertTrue(payload["truncated"])
        self.assertIn("capped", payload["truncationReason"])

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

            scanner = LogRunScanner(logs_root=logs_root)
            run_ids_by_path = {run.relative_path: run.id for run in scanner.list_runs()}
            service = _run_history(logs_root)
            result = service.delete_experiment("test_model")
            remaining_paths = {
                run.relative_path
                for run in LogRunScanner(logs_root=logs_root).list_runs()
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

    def test_run_history_filtered_delete_candidate_safety_matrix(self) -> None:
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
            scanner = LogRunScanner(logs_root=logs_root)
            executor = LogRunDeletionExecutor(scanner=scanner)
            filters = delete_filters_for_runs(scanner.list_runs())
            result = _delete_runs(_run_history(logs_root), filters)

            self.assertEqual(
                result.deleted_relative_paths,
                ("test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",),
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
            runs = LogRunScanner(logs_root=logs_root).list_runs()
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
                for run in LogRunScanner(logs_root=logs_root).list_runs()
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
                LogRunScanner(logs_root=logs_root).list_runs(),
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

            runs = LogRunScanner(logs_root=logs_root).list_runs()
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

    def test_preset_delete_is_backend_authoritative_beyond_filter_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for index in range(55):
                write_tensorboard_run(
                    logs_root,
                    [
                        "test_model",
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

    def test_log_run_listing_filters_facets_by_experiment_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for task, experiment in (
                ("image-classification", "image_exp"),
                ("language-modeling", "language_exp"),
            ):
                run_dir = write_tensorboard_run(
                    logs_root,
                    [
                        experiment,
                        "linears",
                        "linear",
                        "BASELINE",
                        "Mnist",
                        f"run_20260711_01010{len(experiment)}",
                        "version_0",
                    ],
                )
                (run_dir / "result.json").write_text(
                    json.dumps({"experimentTask": task, "metrics": {}}),
                    encoding="utf-8",
                )

            payload = _run_history(logs_root).list_runs(
                limit=10,
                offset=0,
                model=["linears/linear"],
                experiment_task="image-classification",
                projection="summary",
            )

            self.assertEqual(payload.total, 1)
            self.assertEqual(payload.runs[0].experiment, "image_exp")
            self.assertEqual(
                [facet.experiment for facet in payload.facets.experiments],
                ["image_exp"],
            )

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
            runs = LogRunScanner(logs_root=logs_root).list_runs()
            filters = delete_filters_for_runs(runs)

            result = _delete_runs(_run_history(logs_root), filters)

            self.assertEqual(len(result.deleted_run_ids), 2)
            self.assertEqual(set(result.deleted_run_ids), {run.id for run in runs})
            self.assertTrue(all(not run_dir.exists() for run_dir in run_dirs))

    def test_delete_recomputes_stale_preview_without_a_plan_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            relative_parts = [
                "test_model",
                "linear",
                "BASELINE",
                "Mnist",
                "run_20260711_080910",
                "version_0",
            ]
            write_tensorboard_run(logs_root, relative_parts)
            service = _run_history(logs_root)
            run = LogRunScanner(logs_root=logs_root).list_runs()[0]
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
            runs = LogRunScanner(logs_root=logs_root).list_runs()
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
                    "workbench.backend.run_history.deletion.shutil.rmtree",
                    side_effect=fail_second_delete,
                ),
                self.assertRaises(OSError),
            ):
                _delete_runs(service, filters)

            self.assertEqual(service.list_runs(limit=10, offset=0).total, 1)

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
            service = _run_history(logs_root)
            run = LogRunScanner(logs_root=logs_root).list_runs()[0]
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
                run_ids=(LogRunScanner(logs_root=logs_root).list_runs()[0].id,),
            )
            plan = _delete_plan(service, filters)

            self.assertFalse(plan.can_delete)
            self.assertEqual(plan.candidates, ())
            with self.assertRaisesRegex(
                RunHistoryFailure,
                "No log runs match",
            ):
                _delete_runs(service, filters)

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
            scanner = LogRunScanner(logs_root=logs_root)

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

    def test_run_history_active_job_blocks_filtered_destructive_delete(
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
            filters = delete_filters_for_runs(
                LogRunScanner(logs_root=logs_root).list_runs()
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

    def test_run_history_lists_safe_top_level_experiments(self) -> None:
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
                    "workbench-training",
                    "job-123",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )

            experiments = LogRunScanner(logs_root=logs_root).list_experiments()

        self.assertEqual(
            [experiment.experiment for experiment in experiments],
            ["empty_experiment", "test_model"],
        )
        by_name = {experiment.experiment: experiment for experiment in experiments}
        self.assertEqual(by_name["empty_experiment"].run_count, 0)
        self.assertEqual(by_name["test_model"].run_count, 1)
        self.assertEqual(by_name["test_model"].relative_path, "test_model")

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

    def test_log_api_deletes_experiment_and_refreshes_runs(self) -> None:
        import httpx

        from workbench.backend.api import WorkbenchApiSettings, create_app

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
                    app=create_app(
                        WorkbenchApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        )
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
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

        from workbench.backend.api import WorkbenchApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            logs_root.mkdir()
            logs_root.joinpath("new_empty").mkdir()

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(
                        WorkbenchApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        )
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
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

        from workbench.backend.api import WorkbenchApiSettings

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
                    manager = TrainingJobServiceHarness(
                        root=Path(tmp) / "jobs",
                        logs_root=logs_root,
                        runner=FakeRunner(),
                    )
                    job = manager.create_job_payload(
                        model="linears/linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        log_folder=job_log_folder,
                        monitors=[],
                    )
                    job_id = str(job["id"])
                    if job_status == "cancelled":
                        manager.cancel_job_payload(job_id)
                    elif job_status == "completed":
                        manager.runner.process.exit_code = 0
                    elif job_status == "failed":
                        manager.runner.process.exit_code = 1
                    elif job_status != "running":
                        manager.jobs[job_id].status = job_status

                    async def call_api(
                        logs_root: Path = logs_root,
                        manager: TrainingJobServiceHarness = manager,
                    ) -> httpx.Response:
                        transport = httpx.ASGITransport(
                            app=create_app_with_training_service(
                                WorkbenchApiSettings(
                                    logs_root=str(logs_root),
                                    allow_unsafe_local_mutations=True,
                                ),
                                manager,
                            )
                        )
                        async with httpx.AsyncClient(
                            transport=transport,
                            base_url="http://localhost",
                            headers={
                                "X-Workbench-Mutation": "true",
                                "Idempotency-Key": uuid.uuid4().hex,
                            },
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
                        "running" if job_status == "queued" else job_status,
                    )

    def test_log_api_restart_behavior_fresh_manager_preserves_active_delete_blocker(
        self,
    ) -> None:
        import httpx

        from workbench.backend.api import WorkbenchApiSettings

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
            original_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            job = original_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )
            job_id = str(job["id"])
            self.assertEqual(
                original_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )

            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            self.assertEqual(
                fresh_manager.active_job_payloads(),
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
                    app=create_app_with_training_service(
                        WorkbenchApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        ),
                        fresh_manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
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
                original_manager.active_job_payloads(),
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

        from workbench.backend.api import WorkbenchApiSettings

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
            original_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            job = original_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )
            job_id = str(job["id"])
            self.assertEqual(
                original_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )
            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            run = LogRunScanner(logs_root=logs_root).list_runs()[0]
            filters = {
                "experiments": [run.experiment],
                "datasets": [run.dataset],
                "models": [model_identity_payload_from_id(run.model)],
                "presets": [run.preset],
                "runIds": [run.id],
            }

            async def create_plan() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app_with_training_service(
                        WorkbenchApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        ),
                        fresh_manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
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
                    app=create_app_with_training_service(
                        WorkbenchApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        ),
                        fresh_manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
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

        from workbench.backend.api import WorkbenchApiSettings

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
            original_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            job = original_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )
            job_id = str(job["id"])
            self.assertEqual(
                original_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )
            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app_with_training_service(
                        WorkbenchApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        ),
                        fresh_manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
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

        from workbench.backend.api import WorkbenchApiSettings

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
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(
                    app=create_app_with_training_service(
                        WorkbenchApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        ),
                        manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    runs_response = await client.get("/logs/runs")
                    run = runs_response.json()["runs"][0]
                    filters = {
                        "experiments": [run["experiment"]],
                        "datasets": [run["dataset"]],
                        "models": [
                            {
                                "modelType": run["modelType"],
                                "model": run["model"],
                            }
                        ],
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

        from workbench.backend.api import WorkbenchApiSettings, create_app

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
                    app=create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
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

        from workbench.backend.api import WorkbenchApiSettings, create_app

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
                    app=create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
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

        from workbench.backend.api import WorkbenchApiSettings, create_app

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
                    app=create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
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

    def test_run_history_reads_tensorboard_media_summaries(self) -> None:
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

            scanner = LogRunScanner(logs_root=logs_root)
            query = LogRunQueryService(scanner=scanner)
            run = scanner.list_runs()[0]
            tags = query.tags_for_runs([run.id])[0]
            media = query.media_for_runs(
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

            query = LogRunQueryService(scanner=LogRunScanner(logs_root=Path(tmp)))
            summary = query.read_text_summary(
                run_dir,
                "validation/examples/predictions/text_summary",
            )

        self.assertIsNotNone(summary)
        self.assertEqual(summary["step"], 2)
        self.assertEqual(summary["text"], "late")

    def test_log_run_query_service_streams_exact_bounded_scalar_tails(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for point_count in (501, 1_000, 2_000):
                with self.subTest(point_count=point_count):
                    run_dir = root / f"run-{point_count}"
                    writer = SummaryWriter(log_dir=str(run_dir))
                    for step in range(point_count):
                        writer.add_scalar("train/loss", float(step), step)
                    writer.close()
                    query = LogRunQueryService(scanner=LogRunScanner(logs_root=root))

                    series = query.read_scalar_series(
                        run_dir,
                        "train/loss",
                        max_points=500,
                    )

                    self.assertEqual(series["sourcePointCount"], point_count)
                    self.assertTrue(series["truncated"])
                    self.assertEqual(len(series["points"]), 500)
                    self.assertEqual(
                        [point["step"] for point in series["points"]],
                        list(range(point_count - 500, point_count)),
                    )

    def test_log_run_query_service_caches_tags_and_scalar_tails_until_event_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            event_file = run_dir / "events.out.tfevents.cache"
            event_file.write_text("first", encoding="utf-8")
            service = LogRunQueryService(
                scanner=LogRunScanner(logs_root=Path(tmp)),
            )
            load_calls: list[Path] = []

            def load_accumulator(
                event_dir: Path,
                **_kwargs,
            ) -> FakeTensorBoardAccumulator:
                load_calls.append(event_dir)
                return FakeTensorBoardAccumulator()

            scalar_tail = {
                "train/loss": {
                    "points": [
                        {"step": 2, "wallTime": 2.0, "value": 0.25},
                        {"step": 3, "wallTime": 3.0, "value": 0.125},
                    ],
                    "sourcePointCount": 3,
                    "truncated": True,
                }
            }

            with (
                patch(
                    "workbench.backend.tensorboard.events.load_event_accumulator",
                    load_accumulator,
                ),
                patch.object(
                    tensorboard_events,
                    "exact_scalar_tails",
                    return_value=scalar_tail,
                ) as stream_tails,
            ):
                first_tags = service.read_tags(run_dir)
                second_tags = service.read_tags(run_dir)
                first_scalars = service.read_scalar_series(
                    run_dir,
                    "train/loss",
                    max_points=2,
                )
                second_scalars = service.read_scalar_series(
                    run_dir,
                    "train/loss",
                    max_points=2,
                )
                event_file.write_text("first-second", encoding="utf-8")
                changed_tags = service.read_tags(run_dir)

        self.assertEqual(first_tags, second_tags)
        self.assertEqual(first_tags, changed_tags)
        self.assertEqual(first_scalars, second_scalars)
        self.assertEqual(first_scalars["sourcePointCount"], 3)
        self.assertTrue(first_scalars["truncated"])
        self.assertEqual(
            [point["step"] for point in first_scalars["points"]],
            [2, 3],
        )
        self.assertEqual(len(load_calls), 2)
        stream_tails.assert_called_once()

    def test_log_run_query_service_streams_each_scalar_tag_batch_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "events.out.tfevents.cache").write_text(
                "events",
                encoding="utf-8",
            )
            service = LogRunQueryService(
                scanner=LogRunScanner(logs_root=Path(tmp)),
            )

            streamed = {
                tag: {
                    "points": [
                        {
                            "step": 1,
                            "wallTime": 1.0,
                            "value": 0.5 if tag == "loss" else 0.8,
                        }
                    ],
                    "sourcePointCount": 1,
                    "truncated": False,
                }
                for tag in ("loss", "accuracy")
            }
            with patch.object(
                tensorboard_events,
                "exact_scalar_tails",
                return_value=streamed,
            ) as stream_tails:
                result = service.read_scalar_series_batch(
                    run_dir,
                    ["loss", "accuracy"],
                    max_points=2,
                )

        stream_tails.assert_called_once()
        self.assertEqual(result["loss"]["points"][0]["value"], 0.5)
        self.assertEqual(result["accuracy"]["points"][0]["value"], 0.8)

    def test_log_run_query_service_skips_tag_scan_for_oversized_event_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            event_file = run_dir / "events.out.tfevents.large"
            event_file.write_text("large-event-payload", encoding="utf-8")
            service = LogRunQueryService(
                scanner=LogRunScanner(logs_root=Path(tmp)),
                max_tag_event_bytes=4,
            )

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator"
            ) as load:
                tags = service.read_tags(run_dir)

        self.assertEqual(tags["scalars"], [])
        self.assertEqual(tags["histograms"], [])
        self.assertEqual(tags["images"], [])
        self.assertEqual(tags["texts"], [])
        self.assertEqual(tags["eventBytes"], len("large-event-payload"))
        self.assertEqual(tags["skippedEventFiles"], 1)
        self.assertEqual(tags["sourceItemCount"], 1)
        self.assertEqual(tags["returnedItemCount"], 0)
        self.assertTrue(tags["truncated"])
        self.assertIn("event files skipped", tags["truncationReason"])
        load.assert_not_called()

    def test_log_run_query_service_rejects_uncached_tags_past_batch_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            write_fake_event_run(
                logs_root,
                "first_20260601_010203",
                b"1111",
            )
            write_fake_event_run(
                logs_root,
                "second_20260601_020304",
                b"2222",
            )
            write_fake_event_run(
                logs_root,
                "third_20260601_030405",
                b"3333",
            )
            scanner = LogRunScanner(logs_root=logs_root)
            run_ids = {run.run_name: run.id for run in scanner.list_runs()}
            service = LogRunQueryService(
                scanner=scanner,
                max_tag_event_bytes=100,
                max_tag_batch_event_bytes=8,
            )
            loaded_event_dirs: list[Path] = []

            def load_accumulator(
                event_dir: Path,
                **_kwargs,
            ) -> FakeTensorBoardAccumulator:
                loaded_event_dirs.append(event_dir)
                return FakeTensorBoardAccumulator()

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator",
                load_accumulator,
            ):
                with self.assertRaises(RunHistoryFailure) as raised:
                    service.tags_for_runs(
                        [
                            run_ids["first_20260601_010203"],
                            run_ids["second_20260601_020304"],
                            run_ids["third_20260601_030405"],
                        ]
                    )

        self.assertEqual(len(loaded_event_dirs), 2)
        self.assertEqual(raised.exception.kind, FailureKind.TOO_LARGE)
        self.assertIn("8 byte read budget", raised.exception.detail)

    def test_log_run_query_service_does_not_cache_batch_budget_skips(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            write_fake_event_run(
                logs_root,
                "first_20260601_010203",
                b"1111",
            )
            write_fake_event_run(
                logs_root,
                "second_20260601_020304",
                b"2222",
            )
            write_fake_event_run(
                logs_root,
                "third_20260601_030405",
                b"3333",
            )
            scanner = LogRunScanner(logs_root=logs_root)
            run_ids = {run.run_name: run.id for run in scanner.list_runs()}
            service = LogRunQueryService(
                scanner=scanner,
                max_tag_event_bytes=100,
                max_tag_batch_event_bytes=8,
            )

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator",
                return_value=FakeTensorBoardAccumulator(),
            ) as load:
                with self.assertRaises(RunHistoryFailure):
                    service.tags_for_runs(
                        [
                            run_ids["first_20260601_010203"],
                            run_ids["second_20260601_020304"],
                            run_ids["third_20260601_030405"],
                        ]
                    )
                second_payloads = service.tags_for_runs(
                    [run_ids["third_20260601_030405"]]
                )

        self.assertEqual(load.call_count, 3)
        self.assertFalse(second_payloads[0]["truncated"])
        self.assertEqual(second_payloads[0]["scalarTags"], ["train/loss"])

    def test_log_run_query_service_cached_tags_bypass_batch_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            write_fake_event_run(
                logs_root,
                "first_20260601_010203",
                b"11111111",
            )
            write_fake_event_run(
                logs_root,
                "second_20260601_020304",
                b"2222",
            )
            scanner = LogRunScanner(logs_root=logs_root)
            run_ids = {run.run_name: run.id for run in scanner.list_runs()}
            service = LogRunQueryService(
                scanner=scanner,
                max_tag_event_bytes=100,
                max_tag_batch_event_bytes=8,
            )

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator",
                return_value=FakeTensorBoardAccumulator(),
            ) as load:
                service.tags_for_runs([run_ids["first_20260601_010203"]])
                payloads = service.tags_for_runs(
                    [
                        run_ids["first_20260601_010203"],
                        run_ids["second_20260601_020304"],
                    ]
                )

        self.assertEqual(load.call_count, 2)
        self.assertFalse(payloads[0]["truncated"])
        self.assertFalse(payloads[1]["truncated"])
        self.assertEqual(payloads[0]["scalarTags"], ["train/loss"])
        self.assertEqual(payloads[1]["scalarTags"], ["train/loss"])

    def test_log_api_filters_runs_before_pagination(self) -> None:
        import httpx

        from workbench.backend.api import WorkbenchApiSettings, create_app

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
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )
            write_tensorboard_run(
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
            no_event_run = logs_root.joinpath(
                "test_model",
                "linear",
                "BASELINE",
                "Mnist",
                "no_events_20260601_040506",
                "version_0",
            )
            no_event_run.mkdir(parents=True)

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(
                    app=create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    scoped_response = await client.get(
                        "/logs/runs",
                        params=[
                            ("model", "linear"),
                            ("preset", "BASELINE"),
                            ("dataset", "Mnist"),
                            ("hasEventFiles", "true"),
                            ("limit", "5"),
                        ],
                    )
                    no_event_response = await client.get(
                        "/logs/runs",
                        params={
                            "model": "linear",
                            "preset": "BASELINE",
                            "dataset": "Mnist",
                            "hasEventFiles": "false",
                        },
                    )
                    return scoped_response, no_event_response

            scoped_response, no_event_response = asyncio.run(call_api())

        self.assertEqual(scoped_response.status_code, 200)
        scoped_payload = scoped_response.json()
        self.assertEqual(scoped_payload["total"], 1)
        self.assertEqual(len(scoped_payload["runs"]), 1)
        self.assertEqual(scoped_payload["runs"][0]["dataset"], "Mnist")
        self.assertGreater(scoped_payload["runs"][0]["eventFileCount"], 0)

        self.assertEqual(no_event_response.status_code, 200)
        no_event_payload = no_event_response.json()
        self.assertEqual(no_event_payload["total"], 1)
        self.assertEqual(no_event_payload["runs"][0]["eventFileCount"], 0)

    def test_qualified_model_filter_does_not_match_a_shared_leaf_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for model_type in ("linears", "experts"):
                write_tensorboard_run(
                    logs_root,
                    [
                        "experiment",
                        model_type,
                        "linear",
                        "BASELINE",
                        "Mnist",
                        f"{model_type}_20260601_010203",
                        "version_0",
                    ],
                )
            service = _run_history(logs_root)

            qualified = service.list_runs(
                limit=10,
                offset=0,
                model=["linears/linear"],
            )
            legacy = service.list_runs(
                limit=10,
                offset=0,
                model=["linear"],
            )

        self.assertEqual(qualified.total, 1)
        self.assertEqual(qualified.runs[0].model, "linears/linear")
        self.assertEqual(legacy.total, 2)

    def test_log_api_reports_layer_monitor_eligibility_from_tag_cache(
        self,
    ) -> None:
        import httpx

        from workbench.backend.api import WorkbenchApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "layer_20260601_010203",
                    "version_0",
                ],
                scalars={"main_model.0.model/weights/mean": [(1, 0.5)]},
            )
            write_tensorboard_run(
                logs_root,
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "perf_20260601_020304",
                    "version_0",
                ],
                scalars={"train/loss": [(1, 0.5)]},
            )

            async def call_api() -> tuple[
                httpx.Response, httpx.Response, httpx.Response
            ]:
                transport = httpx.ASGITransport(
                    app=create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    with patch(
                        "workbench.backend.tensorboard.events.load_event_accumulator"
                    ) as load:
                        before_response = await client.get("/logs/runs")
                        load.assert_not_called()
                    run_ids_by_name = {
                        run["runName"]: run["id"]
                        for run in before_response.json()["runs"]
                    }
                    tags_response = await client.post(
                        "/logs/tags",
                        json={
                            "runIds": [
                                run_ids_by_name["layer_20260601_010203"],
                                run_ids_by_name["perf_20260601_020304"],
                            ]
                        },
                    )
                    after_response = await client.get("/logs/runs")
                    return before_response, tags_response, after_response

            before_response, tags_response, after_response = asyncio.run(call_api())

        self.assertEqual(before_response.status_code, 200)
        self.assertEqual(tags_response.status_code, 200)
        self.assertEqual(after_response.status_code, 200)

        before_by_name = {run["runName"]: run for run in before_response.json()["runs"]}
        self.assertIsNone(
            before_by_name["layer_20260601_010203"]["hasLayerMonitorData"]
        )
        self.assertIsNone(before_by_name["perf_20260601_020304"]["hasLayerMonitorData"])

        tags_by_run_id = {run["runId"]: run for run in tags_response.json()["runs"]}
        layer_run_id = before_by_name["layer_20260601_010203"]["id"]
        perf_run_id = before_by_name["perf_20260601_020304"]["id"]
        self.assertTrue(tags_by_run_id[layer_run_id]["hasLayerMonitorData"])
        self.assertFalse(tags_by_run_id[perf_run_id]["hasLayerMonitorData"])

        after_by_name = {run["runName"]: run for run in after_response.json()["runs"]}
        self.assertTrue(after_by_name["layer_20260601_010203"]["hasLayerMonitorData"])
        self.assertFalse(after_by_name["perf_20260601_020304"]["hasLayerMonitorData"])

    def test_log_api_scalar_request_limits_and_metadata(self) -> None:
        import httpx

        from workbench.backend.api import WorkbenchApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
                scalars={"train/loss": [(1, 0.5), (2, 0.25), (3, 0.125)]},
            )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(
                    app=create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    runs_response = await client.get("/logs/runs")
                    run_id = runs_response.json()["runs"][0]["id"]
                    scalars_response = await client.post(
                        "/logs/scalars",
                        json={
                            "runIds": [run_id],
                            "tags": ["train/loss"],
                            "maxPoints": 2,
                            "sampling": "tail",
                        },
                    )
                    invalid_response = await client.post(
                        "/logs/scalars",
                        json={
                            "runIds": [run_id],
                            "tags": ["train/loss"],
                            "maxPoints": 2001,
                        },
                    )
                    return scalars_response, invalid_response

            scalars_response, invalid_response = asyncio.run(call_api())

        self.assertEqual(scalars_response.status_code, 200)
        series = scalars_response.json()["series"][0]
        self.assertEqual([point["step"] for point in series["points"]], [2, 3])
        self.assertEqual(series["sourcePointCount"], 3)
        self.assertTrue(series["truncated"])
        self.assertEqual(invalid_response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
