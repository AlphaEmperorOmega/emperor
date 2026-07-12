from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.run_history import RunHistoryService
from workbench.backend.run_history.errors import RunHistoryFailure
from workbench.backend.run_history.records import LogRun
from workbench.backend.schemas import LogExperimentsResponse, LogRunsResponse
from workbench.backend.tests.helpers import write_tensorboard_run


@dataclass(frozen=True, slots=True)
class _ActiveWriter:
    id: str
    status: str
    log_folder: str


def _service(
    logs_root: Path,
    *,
    writers: list[_ActiveWriter] | None = None,
) -> RunHistoryService:
    active_writers = writers or []
    return RunHistoryService(
        logs_root=logs_root,
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: list(active_writers),
    )


def _write_run(
    logs_root: Path,
    *,
    experiment: str,
    dataset: str,
    preset: str,
    run_name: str,
    with_events: bool = True,
) -> Path:
    parts = [
        experiment,
        "linear",
        preset,
        dataset,
        run_name,
        "version_0",
    ]
    if with_events:
        return write_tensorboard_run(logs_root, parts)
    run_dir = logs_root.joinpath(*parts)
    run_dir.mkdir(parents=True)
    return run_dir


class RunHistoryServiceListResponseTests(unittest.TestCase):
    def test_list_runs_filters_before_paginating_and_preserves_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            _write_run(
                logs_root,
                experiment="exp_a",
                dataset="Mnist",
                preset="BASELINE",
                run_name="first_20260711_010101",
            )
            _write_run(
                logs_root,
                experiment="exp_a",
                dataset="Cifar10",
                preset="BASELINE",
                run_name="second_20260711_020202",
            )
            _write_run(
                logs_root,
                experiment="exp_b",
                dataset="Mnist",
                preset="GATING",
                run_name="third_20260711_030303",
                with_events=False,
            )
            result = _service(logs_root).list_runs(
                limit=1,
                offset=0,
                model=["linear"],
                preset=["BASELINE"],
                dataset=["Mnist"],
                has_event_files=True,
            )

        response = LogRunsResponse.model_validate(result)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["limit"], 1)
        self.assertEqual(result["offset"], 0)
        self.assertFalse(result["hasMore"])
        self.assertEqual(response.runs[0].dataset, "Mnist")

    def test_summary_page_keeps_complete_facets_and_omits_expensive_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for dataset, timestamp in (
                ("Mnist", "010101"),
                ("Cifar10", "020202"),
            ):
                _write_run(
                    logs_root,
                    experiment="exp_a",
                    dataset=dataset,
                    preset="BASELINE",
                    run_name=f"run_20260711_{timestamp}",
                )
            result = _service(logs_root).list_runs(
                limit=1,
                offset=0,
                projection="summary",
            )

        response = LogRunsResponse.model_validate(result)
        self.assertEqual(response.runs[0].metrics, {})
        self.assertIsNone(response.runs[0].hasLayerMonitorData)
        self.assertEqual(response.total, 2)
        self.assertEqual(
            [facet.value for facet in response.facets.experiments[0].datasets],
            ["Cifar10", "Mnist"],
        )

    def test_list_runs_serializes_only_requested_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for index in range(5):
                _write_run(
                    logs_root,
                    experiment="exp_a",
                    dataset="Mnist",
                    preset="BASELINE",
                    run_name=f"run_20260711_0{index}0101",
                )
            service = _service(logs_root)
            serialized: list[str] = []
            original_to_response = LogRun.to_response

            def recording_to_response(run: LogRun):
                serialized.append(run.id)
                return original_to_response(run)

            with patch.object(LogRun, "to_response", recording_to_response):
                result = service.list_runs(limit=2, offset=1)

        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(len(serialized), 2)

    def test_list_experiments_returns_response_ready_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            _write_run(
                logs_root,
                experiment="exp_a",
                dataset="Mnist",
                preset="BASELINE",
                run_name="first_20260711_010101",
            )
            _write_run(
                logs_root,
                experiment="exp_b",
                dataset="Mnist",
                preset="BASELINE",
                run_name="second_20260711_020202",
            )
            result = _service(logs_root).list_experiments(limit=1, offset=0)

        response = LogExperimentsResponse.model_validate(result)
        self.assertEqual(result["total"], 2)
        self.assertTrue(result["hasMore"])
        self.assertEqual(
            [experiment.experiment for experiment in response.experiments],
            ["exp_a"],
        )


class RunHistoryServiceDeleteExperimentTests(unittest.TestCase):
    def test_delete_experiment_blocks_matching_active_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            experiment = logs_root / "test_model"
            experiment.mkdir(parents=True)
            service = _service(
                logs_root,
                writers=[_ActiveWriter("job-1", "Running", "test_model")],
            )

            with self.assertRaisesRegex(
                RunHistoryFailure,
                "A training job is still writing to this log folder",
            ):
                service.delete_experiment("test_model")

            self.assertTrue(experiment.is_dir())

    def test_delete_experiment_ignores_nonmatching_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            experiment = logs_root / "test_model"
            experiment.mkdir(parents=True)
            service = _service(
                logs_root,
                writers=[_ActiveWriter("job-1", "Running", "other_model")],
            )

            result = service.delete_experiment("test_model")

            self.assertEqual(result["experiment"], "test_model")
            self.assertFalse(experiment.exists())

    def test_empty_experiment_delete_payload_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            (logs_root / "new_empty").mkdir(parents=True)

            result = _service(logs_root).delete_experiment("new_empty")

        self.assertEqual(
            result,
            {
                "experiment": "new_empty",
                "deletedRunIds": [],
                "deletedRunCount": 0,
                "deletedRelativePath": "new_empty",
            },
        )


if __name__ == "__main__":
    unittest.main()
