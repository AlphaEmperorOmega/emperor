from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from viewer.backend.inspector.errors import InspectorError
from viewer.backend.services.logs import LogRunService


class _DeleteResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to_response(self) -> dict[str, object]:
        return dict(self._payload)


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
