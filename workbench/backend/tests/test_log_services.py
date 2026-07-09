from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.schemas import LogExperimentsResponse, LogRunsResponse
from workbench.backend.services.logs import LogRunService


class _DeleteResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to_response(self) -> dict[str, object]:
        return dict(self._payload)


class _ResponseItem:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __getattr__(self, name: str) -> object:
        try:
            return self._payload[name]
        except KeyError:
            raise AttributeError(name) from None

    def to_response(self) -> dict[str, object]:
        return dict(self._payload)


class ListingLogRunRepository:
    def __init__(
        self,
        *,
        runs: list[dict[str, object]] | None = None,
        experiments: list[dict[str, object]] | None = None,
    ) -> None:
        self._runs = runs or []
        self._experiments = experiments or []

    def list_runs(self) -> list[_ResponseItem]:
        return [_ResponseItem(run) for run in self._runs]

    def list_experiments(self) -> list[_ResponseItem]:
        return [
            _ResponseItem(experiment)
            for experiment in self._experiments
        ]


class RecordingLogRunRepository:
    def __init__(self, delete_payload: dict[str, object] | None = None) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.delete_payload = delete_payload or {
            "experiment": "test_model",
            "deletedRunIds": ["run-1"],
            "deletedRunCount": 1,
            "deletedRelativePath": "test_model",
        }

    def delete_experiment(self, experiment: str) -> _DeleteResult:
        self.calls.append(("delete_experiment", (experiment,), {}))
        return _DeleteResult(self.delete_payload)


def _run_payload(run_id: str, index: int) -> dict[str, object]:
    return {
        "id": run_id,
        "group": None,
        "experiment": "exp",
        "modelType": "models",
        "model": "model",
        "preset": "preset",
        "dataset": "dataset",
        "runName": f"run-{index}",
        "timestamp": None,
        "version": f"version_{index}",
        "relativePath": f"exp/version_{index}",
        "hasResult": False,
        "eventFileCount": 0,
        "checkpointCount": 0,
        "hasHparams": False,
        "metrics": {},
    }


class LogRunServiceListResponseTests(unittest.TestCase):
    def test_list_runs_returns_response_ready_page(self) -> None:
        repository = ListingLogRunRepository(
            runs=[
                _run_payload("run-1", 0),
                _run_payload("run-2", 1),
                _run_payload("run-3", 2),
            ],
        )
        service = LogRunService(repository)  # type: ignore[arg-type]

        result = service.list_runs(limit=1, offset=1)
        response = LogRunsResponse.model_validate(result)

        self.assertEqual(result["total"], 3)
        self.assertEqual(result["limit"], 1)
        self.assertEqual(result["offset"], 1)
        self.assertTrue(result["hasMore"])
        self.assertEqual([run.id for run in response.runs], ["run-2"])

    def test_list_runs_filters_before_paginating(self) -> None:
        first = {
            **_run_payload("run-1", 0),
            "modelType": "linears",
            "model": "linear",
            "preset": "BASELINE",
            "dataset": "Mnist",
            "eventFileCount": 1,
        }
        second = {
            **_run_payload("run-2", 1),
            "modelType": "linears",
            "model": "linear",
            "preset": "BASELINE",
            "dataset": "Cifar10",
            "eventFileCount": 1,
        }
        third = {
            **_run_payload("run-3", 2),
            "modelType": "linears",
            "model": "linear",
            "preset": "GATING",
            "dataset": "Mnist",
            "eventFileCount": 0,
        }
        repository = ListingLogRunRepository(runs=[first, second, third])
        service = LogRunService(repository)  # type: ignore[arg-type]

        result = service.list_runs(
            limit=5,
            offset=0,
            model=["linear"],
            preset=["BASELINE"],
            dataset=["Mnist"],
            has_event_files=True,
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["runs"][0]["id"], "run-1")

    def test_list_experiments_returns_response_ready_page(self) -> None:
        repository = ListingLogRunRepository(
            experiments=[
                {
                    "experiment": "exp-a",
                    "runCount": 2,
                    "relativePath": "exp-a",
                },
                {
                    "experiment": "exp-b",
                    "runCount": 1,
                    "relativePath": "exp-b",
                },
            ],
        )
        service = LogRunService(repository)  # type: ignore[arg-type]

        result = service.list_experiments(limit=1, offset=0)
        response = LogExperimentsResponse.model_validate(result)

        self.assertEqual(result["total"], 2)
        self.assertEqual(result["limit"], 1)
        self.assertEqual(result["offset"], 0)
        self.assertTrue(result["hasMore"])
        self.assertEqual(
            [experiment.experiment for experiment in response.experiments],
            ["exp-a"],
        )


class LogRunServiceDeleteExperimentTests(unittest.TestCase):
    def test_delete_experiment_blocks_matching_active_job(self) -> None:
        repository = RecordingLogRunRepository()
        service = LogRunService(repository)  # type: ignore[arg-type]

        with self.assertRaisesRegex(
            InspectorError,
            "A training job is still writing to this log folder.",
        ):
            service.delete_experiment(
                "test_model",
                active_jobs=[
                    {
                        "id": "job-1",
                        "logFolder": "test_model",
                        "status": "running",
                    }
                ],
            )

        self.assertEqual(repository.calls, [])

    def test_delete_experiment_delegates_for_non_matching_active_jobs(self) -> None:
        repository = RecordingLogRunRepository()
        service = LogRunService(repository)  # type: ignore[arg-type]

        result = service.delete_experiment(
            "test_model",
            active_jobs=[
                {
                    "id": "job-1",
                    "logFolder": "other_model",
                    "status": "running",
                }
            ],
        )

        self.assertEqual(
            repository.calls,
            [("delete_experiment", ("test_model",), {})],
        )
        self.assertEqual(result["experiment"], "test_model")

    def test_delete_experiment_success_response_payload_is_unchanged(self) -> None:
        expected = {
            "experiment": "new_empty",
            "deletedRunIds": [],
            "deletedRunCount": 0,
            "deletedRelativePath": "new_empty",
        }
        repository = RecordingLogRunRepository(delete_payload=expected)
        service = LogRunService(repository)  # type: ignore[arg-type]

        self.assertEqual(
            service.delete_experiment("new_empty", active_jobs=[]),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
