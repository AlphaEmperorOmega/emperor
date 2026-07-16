from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from emperor_workbench.run_history import (
    LogRun,
    LogRunDeleteCandidate,
    LogRunDeletePlan,
    RunHistoryFailure,
)
from emperor_workbench.run_history._deletion import LogRunDeletionExecutor
from emperor_workbench.run_history._query import LogRunQueryService
from emperor_workbench.tensorboard import (
    EventFileIndex,
    MonitorData,
    ParameterStatus,
    ScalarPoint,
    ScalarTail,
    TagCatalog,
    event_file_index,
)
from tests.unit.run_history._support import log_run_scanner
from tests.unit.tensorboard._support import patch_event_accumulator_loader


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
        run_name="aaa_20260601_010203",
        timestamp="2026-06-01 01:02:03",
        version="version_0",
        relative_path=(
            f"test_model/linears/linear/{preset}/{dataset}/"
            "aaa_20260601_010203/version_0"
        ),
        has_result=False,
        event_file_count=0,
        checkpoint_count=0,
        has_hparams=False,
        path=path,
    )


def _delete_candidate(relative_path: str) -> LogRunDeleteCandidate:
    return LogRunDeleteCandidate(
        id="run-1",
        experiment="test_model",
        model="linears/linear",
        preset="BASELINE",
        dataset="Mnist",
        run_name="aaa_20260601_010203",
        version="version_0",
        relative_path=relative_path,
    )


@dataclass(frozen=True, slots=True)
class _StaticArtifactObservation:
    event_files: EventFileIndex


class StaticLogRunScanner:
    def __init__(self, runs: list[LogRun]) -> None:
        self.runs_by_id = {run.id: run for run in runs}
        self.requested_run_ids: list[list[str]] = []

    def resolve_runs(self, run_ids: list[str]) -> list[LogRun]:
        self.requested_run_ids.append(run_ids)
        unknown = [run_id for run_id in run_ids if run_id not in self.runs_by_id]
        if unknown:
            raise RunHistoryFailure(f"Unknown log run id: {unknown[0]}")
        return [self.runs_by_id[run_id] for run_id in dict.fromkeys(run_ids)]

    def artifact_observation(self, run: LogRun) -> _StaticArtifactObservation:
        return _StaticArtifactObservation(event_file_index(run.path))


class RecordingMonitorReader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def read(self, **kwargs: Any) -> MonitorData:
        kwargs.pop("event_files", None)
        self.calls.append(kwargs)
        return MonitorData(
            job_id=kwargs["job_id"],
            node_path=kwargs["node_path"],
            preset=kwargs.get("preset"),
            dataset=kwargs["dataset"],
            log_dir=kwargs["log_dir"],
            scalar_series=(),
            histograms=(),
            images=(),
        )


class RecordingParameterStatusReader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def read(self, **kwargs: Any) -> ParameterStatus:
        kwargs.pop("event_files", None)
        self.calls.append(kwargs)
        return ParameterStatus(
            source_id=kwargs["source_id"],
            preset=kwargs["preset"],
            dataset=kwargs["dataset"],
            log_dir=kwargs["log_dir"],
            nodes=(),
        )


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

    def read_tags(
        self,
        run_dir: Path,
        *,
        event_files: object | None = None,
        cache_generation: int | None = None,
    ) -> TagCatalog:
        del event_files, cache_generation
        if run_dir.name == "run-1":
            return TagCatalog(
                scalar_tags=("accuracy", "loss"),
                histogram_tags=("weights",),
                image_tags=(),
                text_tags=("notes/text_summary",),
            )
        return TagCatalog(
            scalar_tags=("loss",),
            histogram_tags=(),
            image_tags=("sample",),
            text_tags=(),
        )

    def read_scalar_series(
        self,
        run_dir: Path,
        tag: str,
        *,
        max_points: int | None = None,
        sampling: str = "tail",
    ) -> ScalarTail:
        del max_points, sampling
        self.scalar_requests.append((run_dir, tag))
        return ScalarTail(
            points=(ScalarPoint(step=1, wall_time=10.0, value=0.5),),
            source_point_count=1,
            truncated=False,
        )

    def read_scalar_series_batch(
        self,
        run_dir: Path,
        tags: list[str],
        *,
        max_points: int | None = None,
        sampling: str = "tail",
        event_files: object | None = None,
        cache_generation: int | None = None,
    ) -> dict[str, ScalarTail]:
        del event_files, cache_generation
        return {
            tag: self.read_scalar_series(
                run_dir,
                tag,
                max_points=max_points,
                sampling=sampling,
            )
            for tag in tags
        }


class LogRunQueryServiceTests(unittest.TestCase):
    def test_scalars_for_runs_batches_scalar_reads_per_event_dir(self) -> None:
        class ScalarEvent:
            def __init__(self, step: int, value: float) -> None:
                self.step = step
                self.value = value
                self.wall_time = float(step)

        class BatchAccumulator:
            def Tags(self) -> dict[str, list[str]]:
                return {
                    "scalars": ["accuracy", "loss"],
                    "histograms": [],
                    "images": [],
                    "tensors": [],
                }

            def Scalars(self, tag: str) -> list[ScalarEvent]:
                return [ScalarEvent(step=1, value=0.5 if tag == "loss" else 0.8)]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-1"
            run_dir.mkdir()
            (run_dir / "events.out.tfevents.batch").write_text(
                "event",
                encoding="utf-8",
            )
            scanner = StaticLogRunScanner([_log_run("run-1", path=run_dir)])
            service = LogRunQueryService(scanner=scanner)
            load_calls: list[Path] = []

            def load_accumulator(
                _event_dir: Path,
                **_kwargs: Any,
            ) -> BatchAccumulator:
                return BatchAccumulator()

            def exact_scalar_tails(
                event_files,  # type: ignore[no-untyped-def]
                tags: list[str],
                **_kwargs: Any,
            ) -> dict[str, ScalarTail]:
                load_calls.append(event_files.root)
                accumulator = BatchAccumulator()
                return {
                    tag: ScalarTail(
                        points=tuple(
                            ScalarPoint(
                                step=event.step,
                                wall_time=event.wall_time,
                                value=event.value,
                            )
                            for event in accumulator.Scalars(tag)
                        ),
                        source_point_count=1,
                        truncated=False,
                    )
                    for tag in tags
                }

            with (
                patch_event_accumulator_loader(load_accumulator),
                patch(
                    "emperor_workbench.tensorboard.exact_scalar_tails",
                    exact_scalar_tails,
                ),
            ):
                service.read_tags(run_dir)
                load_calls.clear()
                scalars = service.scalars_for_runs(
                    run_ids=["run-1"],
                    tags=["loss", "accuracy"],
                )

        self.assertEqual(load_calls, [run_dir])
        self.assertEqual(
            [(series.tag, series.points[0].value) for series in scalars],
            [("loss", 0.5), ("accuracy", 0.8)],
        )

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

        self.assertEqual(tags[0].run_id, "run-1")
        self.assertFalse(tags[0].has_layer_monitor_data)
        self.assertEqual(tags[0].scalar_tags, ("accuracy", "loss"))
        self.assertEqual(tags[0].histogram_tags, ("weights",))
        self.assertEqual(tags[0].image_tags, ())
        self.assertEqual(tags[0].text_tags, ("notes/text_summary",))
        self.assertEqual(
            [(series.run_id, series.tag, series.points[0].value) for series in scalars],
            [
                ("run-1", "loss", 0.5),
                ("run-1", "accuracy", 0.5),
                ("run-2", "loss", 0.5),
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
        self.assertEqual(monitor.job_id, "run-1")
        self.assertEqual(monitor.node_path, "main.layer")
        self.assertEqual(monitor.dataset, "Mnist")
        self.assertEqual(monitor.log_dir, str(first_path))
        self.assertEqual(
            [
                (
                    status.source_id,
                    status.preset,
                    status.dataset,
                    status.log_dir,
                )
                for status in statuses
            ],
            [
                ("run-1", "BASELINE", "Mnist", str(first_path)),
                ("run-2", "BASELINE", "Cifar10", str(second_path)),
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
            scanner = log_run_scanner(logs_root=logs_root)
            executor = LogRunDeletionExecutor(scanner=scanner)

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
                    with self.assertRaisesRegex(
                        RunHistoryFailure,
                        error_pattern,
                    ):
                        executor.delete_runs(
                            LogRunDeletePlan(
                                candidates=(_delete_candidate(relative_path),)
                            )
                        )
            self.assertTrue(valid_run.exists())
            self.assertTrue(invalid_run.exists())
            self.assertTrue(outside_run.exists())
            self.assertTrue(symlink_run.is_symlink())

            result = executor.delete_runs(
                LogRunDeletePlan(
                    candidates=(
                        _delete_candidate(valid_run.relative_to(logs_root).as_posix()),
                    )
                )
            )
            self.assertEqual(result.deleted_run_ids, ("run-1",))
            self.assertFalse(valid_run.exists())


if __name__ == "__main__":
    unittest.main()
