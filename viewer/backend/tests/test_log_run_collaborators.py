from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from viewer.backend.inspector.errors import InspectorError
from viewer.backend.log_runs import (
    LogRun,
    LogRunDeleteCandidate,
    LogRunDeletionExecutor,
    LogRunQueryService,
    LogRunScanner,
)


def _log_run(
    run_id: str,
    *,
    path: Path,
    dataset: str = "Mnist",
    preset: str = "BASELINE",
) -> LogRun:
    return LogRun(
        id=run_id,
        group=None,
        experiment="test_model",
        model="linears/linear",
        preset=preset,
        dataset=dataset,
        runName="aaa_20260601_010203",
        timestamp="2026-06-01 01:02:03",
        version="version_0",
        relativePath=(
            f"test_model/linears/linear/{preset}/{dataset}/"
            "aaa_20260601_010203/version_0"
        ),
        hasResult=False,
        eventFileCount=0,
        checkpointCount=0,
        hasHparams=False,
        path=path,
    )


def _delete_candidate(relative_path: str) -> LogRunDeleteCandidate:
    return LogRunDeleteCandidate(
        id="run-1",
        experiment="test_model",
        model="linears/linear",
        preset="BASELINE",
        dataset="Mnist",
        runName="aaa_20260601_010203",
        version="version_0",
        relativePath=relative_path,
    )


class StaticLogRunScanner:
    def __init__(self, runs: list[LogRun]) -> None:
        self.runs_by_id = {run.id: run for run in runs}
        self.requested_run_ids: list[list[str]] = []

    def resolve_runs(self, run_ids: list[str]) -> list[LogRun]:
        self.requested_run_ids.append(run_ids)
        unknown = [run_id for run_id in run_ids if run_id not in self.runs_by_id]
        if unknown:
            raise InspectorError(f"Unknown log run id: {unknown[0]}")
        return [self.runs_by_id[run_id] for run_id in dict.fromkeys(run_ids)]


class RecordingMonitorReader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def read(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"source": "monitor", **kwargs}


class RecordingParameterStatusReader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def read(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"source": "parameter-status", **kwargs}


class StubLogRunQueryService(LogRunQueryService):
    def __init__(
        self,
        *,
        scanner: StaticLogRunScanner,
        monitor_reader: RecordingMonitorReader,
        parameter_status_reader: RecordingParameterStatusReader,
    ) -> None:
        super().__init__(
            scanner=scanner,
            monitor_reader=monitor_reader,
            parameter_status_reader=parameter_status_reader,
        )
        self.scalar_requests: list[tuple[Path, str]] = []

    def read_tags(self, run_dir: Path) -> dict[str, list[str]]:
        if run_dir.name == "run-1":
            return {
                "scalars": ["accuracy", "loss"],
                "histograms": ["weights"],
                "images": [],
                "texts": ["notes/text_summary"],
            }
        return {
            "scalars": ["loss"],
            "histograms": [],
            "images": ["sample"],
            "texts": [],
        }

    def read_scalar_points(self, run_dir: Path, tag: str) -> list[dict[str, Any]]:
        self.scalar_requests.append((run_dir, tag))
        return [{"step": 1, "wallTime": 10.0, "value": 0.5}]


class LogRunQueryServiceTests(unittest.TestCase):
    def test_read_only_queries_use_scanner_and_readers_without_delete_flow(
        self,
    ) -> None:
        first_path = Path("/logs/run-1")
        second_path = Path("/logs/run-2")
        scanner = StaticLogRunScanner(
            [
                _log_run("run-1", path=first_path),
                _log_run("run-2", path=second_path, dataset="Cifar10"),
            ]
        )
        monitor_reader = RecordingMonitorReader()
        parameter_status_reader = RecordingParameterStatusReader()
        service = StubLogRunQueryService(
            scanner=scanner,
            monitor_reader=monitor_reader,
            parameter_status_reader=parameter_status_reader,
        )

        tags = service.tags_for_runs(["run-1"])
        scalars = service.scalars_for_runs(
            run_ids=["run-1", "run-1", "run-2"],
            tags=["loss", "loss", "accuracy", "missing"],
        )
        monitor = service.monitor_data_for_run("run-1", node_path="main.layer")
        statuses = service.parameter_status_for_runs(["run-1", "run-2"])

        self.assertEqual(
            tags,
            [
                {
                    "runId": "run-1",
                    "scalarTags": ["accuracy", "loss"],
                    "histogramTags": ["weights"],
                    "imageTags": [],
                    "textTags": ["notes/text_summary"],
                }
            ],
        )
        self.assertEqual(
            scalars,
            [
                {
                    "runId": "run-1",
                    "tag": "loss",
                    "points": [{"step": 1, "wallTime": 10.0, "value": 0.5}],
                },
                {
                    "runId": "run-1",
                    "tag": "accuracy",
                    "points": [{"step": 1, "wallTime": 10.0, "value": 0.5}],
                },
                {
                    "runId": "run-2",
                    "tag": "loss",
                    "points": [{"step": 1, "wallTime": 10.0, "value": 0.5}],
                },
            ],
        )
        self.assertEqual(
            service.scalar_requests,
            [
                (first_path, "loss"),
                (first_path, "accuracy"),
                (second_path, "loss"),
            ],
        )
        self.assertEqual(monitor["job_id"], "run-1")
        self.assertEqual(monitor["node_path"], "main.layer")
        self.assertEqual(monitor["dataset"], "Mnist")
        self.assertEqual(monitor["log_dir"], str(first_path))
        self.assertEqual(
            statuses,
            [
                {
                    "source": "parameter-status",
                    "source_id": "run-1",
                    "preset": "BASELINE",
                    "dataset": "Mnist",
                    "log_dir": str(first_path),
                },
                {
                    "source": "parameter-status",
                    "source_id": "run-2",
                    "preset": "BASELINE",
                    "dataset": "Cifar10",
                    "log_dir": str(second_path),
                },
            ],
        )
        self.assertEqual(
            scanner.requested_run_ids,
            [
                ["run-1"],
                ["run-1", "run-1", "run-2"],
                ["run-1"],
                ["run-1", "run-2"],
            ],
        )


class LogRunDeletionExecutorTests(unittest.TestCase):
    def test_delete_path_validation_is_isolated_from_plan_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            valid_run = logs_root.joinpath(
                "test_model",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "aaa_20260601_010203",
                "version_0",
            )
            valid_run.mkdir(parents=True)
            invalid_run = valid_run.parent / "not_a_version"
            invalid_run.mkdir()
            outside_run = root.joinpath(
                "outside",
                "test_model",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "aaa_20260601_010203",
                "version_0",
            )
            outside_run.mkdir(parents=True)
            symlink_run = valid_run.parent / "version_link"
            symlink_run.symlink_to(outside_run, target_is_directory=True)
            scanner = LogRunScanner(logs_root=logs_root)
            executor = LogRunDeletionExecutor(scanner=scanner)
            resolved_root = scanner.resolved_root()

            self.assertEqual(
                executor.validated_delete_candidate_path(
                    _delete_candidate(valid_run.relative_to(logs_root).as_posix()),
                    resolved_root,
                ),
                valid_run,
            )

            cases = (
                (
                    "non_version_directory",
                    invalid_run.relative_to(logs_root).as_posix(),
                    "non-version log folder",
                ),
                (
                    "escaped_path",
                    "../outside/test_model/linears/linear/BASELINE/Mnist/"
                    "aaa_20260601_010203/version_0",
                    "Invalid log run path",
                ),
                (
                    "symlink_version",
                    symlink_run.relative_to(logs_root).as_posix(),
                    "symlink log run",
                ),
            )
            for label, relative_path, error_pattern in cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(InspectorError, error_pattern):
                        executor.validated_delete_candidate_path(
                            _delete_candidate(relative_path),
                            resolved_root,
                        )
            self.assertTrue(valid_run.exists())
            self.assertTrue(invalid_run.exists())
            self.assertTrue(outside_run.exists())
            self.assertTrue(symlink_run.is_symlink())


if __name__ == "__main__":
    unittest.main()
