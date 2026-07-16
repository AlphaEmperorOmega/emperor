from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import get_type_hints

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi.routing import APIRoute, iter_route_contexts
from pydantic import ValidationError

from emperor_workbench.api.v1.run_plans import (
    SubmittedTrainingRunPlanRequest,
    TrainingRunPlanCreateRequest,
    TrainingRunPlanResponse,
)
from emperor_workbench.api.v1.training_jobs import (
    TrainingJobCreateRequest,
    TrainingJobReconcileRequest,
    TrainingJobResponse,
    TrainingProgressEventsResponse,
)
from emperor_workbench.run_plans import MAX_TRAINING_PLANNED_RUNS
from emperor_workbench.settings import WorkbenchApiSettings
from tests.support.training_jobs import (
    FakeProcess,
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
)


class TrainingApiContractTests(unittest.TestCase):
    def _create_test_app(
        self,
        root: Path,
        *,
        process: FakeProcess | None = None,
    ):
        logs_root = root / "logs"
        manager = TrainingJobServiceHarness(
            root=root / "jobs",
            logs_root=logs_root,
            runner=FakeRunner(process),
        )
        app = create_app_with_training_service(
            WorkbenchApiSettings(
                logs_root=str(logs_root),
                snapshots_root=str(root / "snapshots"),
                allow_unsafe_local_mutations=True,
            ),
            manager,
        )
        return app, manager

    def test_training_routes_preserve_paths_methods_and_schema_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))
            routes = {
                (tuple(sorted(route.methods or ())), route.path): route
                for route in iter_route_contexts(app.routes)
                if isinstance(route.original_route, APIRoute)
                and route.path is not None
                and route.path.startswith(("/training", "/v1/training"))
            }

        expected = {
            (("POST",), "/training/jobs"): (
                TrainingJobResponse,
                (TrainingJobCreateRequest,),
            ),
            (("POST",), "/training/run-plan"): (
                TrainingRunPlanResponse,
                (TrainingRunPlanCreateRequest,),
            ),
            (("GET",), "/training/jobs/{job_id}"): (
                TrainingJobResponse,
                (),
            ),
            (("GET",), "/training/jobs/{job_id}/events"): (
                TrainingProgressEventsResponse,
                (),
            ),
            (("GET",), "/training/jobs/{job_id}/monitor-data"): (
                "MonitorDataResponse",
                (),
            ),
            (("GET",), "/training/jobs/{job_id}/monitor-parameter-status"): (
                "ParameterStatusResponse",
                (),
            ),
            (("POST",), "/training/jobs/{job_id}/cancel"): (
                TrainingJobResponse,
                (),
            ),
            (("POST",), "/training/jobs/{job_id}/reconcile"): (
                TrainingJobResponse,
                (TrainingJobReconcileRequest,),
            ),
        }

        self.assertEqual(sorted(routes), sorted(expected))
        self.assertFalse(any(path.startswith("/v1/") for _methods, path in routes))
        for key, (response_model, body_models) in expected.items():
            with self.subTest(route=key):
                route = routes[key]
                endpoint = route.endpoint
                self.assertIsNotNone(endpoint)
                if isinstance(response_model, str):
                    self.assertEqual(route.response_model.__name__, response_model)
                else:
                    self.assertIs(route.response_model, response_model)
                self.assertEqual(
                    tuple(
                        get_type_hints(endpoint)[param.name]
                        for param in route.dependant.body_params
                    ),
                    body_models,
                )

    def test_training_create_request_overrides_are_config_values(self) -> None:
        base_payload = {
            "modelType": "linears",
            "model": "linear",
            "preset": "baseline",
            "datasets": ["Mnist"],
            "logFolder": "api_schema",
        }
        overrides = {
            "hidden_dim": 128,
            "learning_rate": 0.01,
            "use_bias": True,
            "activation": "RELU",
            "optional_layer": None,
        }

        job_request = TrainingJobCreateRequest.model_validate(
            {
                **base_payload,
                "overrides": overrides,
                "monitors": [],
            }
        )
        run_plan_request = TrainingRunPlanCreateRequest.model_validate(
            {
                **base_payload,
                "overrides": overrides,
            }
        )

        self.assertEqual(job_request.overrides, overrides)
        self.assertEqual(run_plan_request.overrides, overrides)
        for schema in (TrainingJobCreateRequest, TrainingRunPlanCreateRequest):
            with self.subTest(schema=schema.__name__):
                with self.assertRaises(ValidationError):
                    schema.model_validate(
                        {
                            **base_payload,
                            "overrides": {"scheduler": {"name": "cosine"}},
                            **(
                                {"monitors": []}
                                if schema is TrainingJobCreateRequest
                                else {}
                            ),
                        }
                    )

    def test_submitted_run_plan_overrides_are_config_values(self) -> None:
        base_payload = {
            "runs": [
                {
                    "id": "run-1",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "overrides": {"hidden_dim": 128, "use_bias": True},
                }
            ],
        }

        request = SubmittedTrainingRunPlanRequest.model_validate(base_payload)

        self.assertEqual(
            request.runs[0].overrides,
            {"hidden_dim": 128, "use_bias": True},
        )
        with self.assertRaises(ValidationError):
            SubmittedTrainingRunPlanRequest.model_validate(
                {
                    "runs": [
                        {
                            **base_payload["runs"][0],
                            "overrides": {"scheduler": {"name": "cosine"}},
                        }
                    ],
                }
            )

    def test_submitted_run_plan_rejects_response_only_projection_fields(self) -> None:
        row = {
            "id": "run-1",
            "preset": "baseline",
            "dataset": "Mnist",
            "overrides": {},
        }
        SubmittedTrainingRunPlanRequest.model_validate({"runs": [row]})

        for response_only_payload in (
            {"runs": [row], "summary": {"totalRuns": 1}},
            {"runs": [{**row, "status": "Pending"}]},
            {"runs": [{**row, "command": "client command"}]},
            {"runs": [{**row, "totalEpochs": 999}]},
            {"runs": [{**row, "currentEpoch": 1}]},
            {"runs": [{**row, "metrics": {"client": 1}}]},
        ):
            with self.subTest(payload=response_only_payload):
                with self.assertRaises(ValidationError):
                    SubmittedTrainingRunPlanRequest.model_validate(
                        response_only_payload
                    )

    def test_training_request_schemas_cap_preset_selection_count(self) -> None:
        presets = [f"preset-{index}" for index in range(MAX_TRAINING_PLANNED_RUNS + 1)]
        create_payload = {
            "modelType": "linears",
            "model": "linear",
            "preset": "baseline",
            "presets": presets,
            "datasets": ["Mnist"],
            "overrides": {},
            "logFolder": "api_schema",
        }
        with self.assertRaises(ValidationError):
            TrainingJobCreateRequest.model_validate({**create_payload, "monitors": []})
        with self.assertRaises(ValidationError):
            TrainingRunPlanCreateRequest.model_validate(create_payload)

    def test_training_responses_reject_nested_override_objects(self) -> None:
        summary = {
            "totalRuns": 0,
            "completedRuns": 0,
            "runningRuns": 0,
            "pendingRuns": 0,
            "failedRuns": 0,
            "cancelledRuns": 0,
            "skippedRuns": 0,
            "totalEpochs": 0,
            "completedEpochs": 0,
            "remainingEpochs": 0,
        }
        run_plan_payload = {
            "modelType": "linears",
            "model": "linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {"hidden_dim": 128},
            "search": None,
            "logFolder": "api_schema",
            "isRandomSearch": False,
            "runs": [],
            "summary": summary,
        }
        job_payload = {
            "id": "job-1",
            "status": "running",
            "modelType": "linears",
            "model": "linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {"hidden_dim": 128},
            "search": None,
            "plannedRunCount": 0,
            "runPlan": run_plan_payload,
            "monitors": [],
            "logFolder": "api_schema",
            "createdAt": "2026-06-09T00:00:00Z",
            "updatedAt": "2026-06-09T00:00:00Z",
            "exitCode": None,
            "pid": 123,
            "currentPreset": None,
            "currentDataset": None,
            "epoch": None,
            "step": None,
            "metrics": {},
            "logDir": None,
            "events": [],
            "logTail": [],
            "resultLinks": [],
        }

        self.assertEqual(
            TrainingRunPlanResponse.model_validate(run_plan_payload).overrides,
            {"hidden_dim": 128},
        )
        self.assertEqual(
            TrainingJobResponse.model_validate(job_payload).overrides,
            {"hidden_dim": 128},
        )
        with self.assertRaises(ValidationError):
            TrainingRunPlanResponse.model_validate(
                {
                    **run_plan_payload,
                    "overrides": {"scheduler": {"name": "cosine"}},
                }
            )
        with self.assertRaises(ValidationError):
            TrainingJobResponse.model_validate(
                {
                    **job_payload,
                    "overrides": {"scheduler": {"name": "cosine"}},
                }
            )

    def test_training_job_response_events_are_typed_progress_events(self) -> None:
        payload = {
            "id": "job-1",
            "status": "running",
            "modelType": "linears",
            "model": "linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {},
            "search": None,
            "plannedRunCount": 0,
            "runPlan": None,
            "monitors": [],
            "logFolder": "api_schema",
            "createdAt": "2026-06-09T00:00:00Z",
            "updatedAt": "2026-06-09T00:00:00Z",
            "exitCode": None,
            "pid": 123,
            "currentPreset": None,
            "currentDataset": None,
            "epoch": None,
            "step": None,
            "metrics": {},
            "logDir": None,
            "events": [
                {
                    "type": "validation",
                    "status": "running",
                    "jobId": "job-1",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": "run-1",
                    "runIndex": 1,
                    "epoch": 0,
                    "step": 7,
                    "metrics": {"validation/accuracy": 0.75},
                },
                {
                    "type": "cluster_initialized",
                    "node": "main.cluster",
                    "count": 2,
                    "capacity": [2, 2, 2],
                    "coordinates": [[0, 0, 0], [0, 0, 1]],
                },
                {
                    "type": "future_event",
                    "customField": {"still": "allowed"},
                },
            ],
            "logTail": [],
            "resultLinks": [],
        }

        response = TrainingJobResponse.model_validate(payload)

        self.assertEqual(response.events[0].type, "validation")
        self.assertEqual(response.events[0].metrics, {"validation/accuracy": 0.75})
        self.assertEqual(response.events[1].type, "cluster_initialized")
        self.assertEqual(response.events[1].capacity, [2, 2, 2])
        self.assertEqual(response.events[2].type, "future_event")
        self.assertEqual(
            response.events[2].model_extra,
            {"customField": {"still": "allowed"}},
        )
        with self.assertRaises(ValidationError):
            TrainingJobResponse.model_validate(
                {**payload, "events": [{"status": "running"}]}
            )


if __name__ == "__main__":
    unittest.main()
