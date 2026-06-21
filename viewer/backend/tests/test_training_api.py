from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi.routing import APIRoute
from pydantic import ValidationError

from viewer.backend.api import ViewerApiSettings, create_app
from viewer.backend.schemas import (
    MonitorDataResponse,
    ParameterStatusResponse,
    SubmittedTrainingRunPlanRequest,
    TrainingJobCreateRequest,
    TrainingJobResponse,
    TrainingProgressEventsResponse,
    TrainingRunPlanCreateRequest,
    TrainingRunPlanResponse,
)
from viewer.backend.tests.helpers import FakeProcess, FakeRunner
from viewer.backend.training_jobs import TrainingJobManager

EXPECTED_TRAINING_JOB_RESPONSE_FIELDS = (
    "id",
    "status",
    "modelType",
    "model",
    "preset",
    "presets",
    "datasets",
    "overrides",
    "search",
    "plannedRunCount",
    "runPlan",
    "monitors",
    "logFolder",
    "createdAt",
    "updatedAt",
    "exitCode",
    "pid",
    "cancellationMode",
    "currentPreset",
    "currentDataset",
    "epoch",
    "step",
    "metrics",
    "logDir",
    "events",
    "eventCount",
    "eventCounts",
    "eventsTruncated",
    "clusterGrowth",
    "logTail",
    "resultLinks",
)

EXPECTED_TRAINING_RUN_PLAN_RESPONSE_FIELDS = (
    "modelType",
    "model",
    "preset",
    "presets",
    "datasets",
    "overrides",
    "search",
    "logFolder",
    "isRandomSearch",
    "runs",
    "summary",
)

EXPECTED_TRAINING_RUN_RESPONSE_FIELDS = (
    "id",
    "index",
    "status",
    "preset",
    "snapshotId",
    "snapshotName",
    "dataset",
    "changes",
    "overrides",
    "command",
    "totalEpochs",
    "currentEpoch",
    "metrics",
    "logDir",
    "error",
    "errorTraceback",
)


class TrainingApiLifecycleTests(unittest.TestCase):
    def _create_test_app(
        self,
        root: Path,
        *,
        process: FakeProcess | None = None,
    ):
        logs_root = root / "logs"
        manager = TrainingJobManager(
            root=root / "jobs",
            logs_root=logs_root,
            runner=FakeRunner(process),
        )
        app = create_app(
            ViewerApiSettings(
                logs_root=str(logs_root),
                allow_unsafe_local_mutations=True,
            ),
            training_manager=manager,
        )
        return app, manager

    def test_training_routes_preserve_paths_methods_and_schema_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))
            routes = {
                (tuple(sorted(route.methods or ())), route.path): route
                for route in app.routes
                if isinstance(route, APIRoute)
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
                MonitorDataResponse,
                (),
            ),
            (("GET",), "/training/jobs/{job_id}/monitor-parameter-status"): (
                ParameterStatusResponse,
                (),
            ),
            (("POST",), "/training/jobs/{job_id}/cancel"): (
                TrainingJobResponse,
                (),
            ),
        }

        self.assertEqual(sorted(routes), sorted(expected))
        self.assertFalse(any(path.startswith("/v1/") for _methods, path in routes))
        for key, (response_model, body_models) in expected.items():
            with self.subTest(route=key):
                route = routes[key]
                self.assertIs(route.response_model, response_model)
                self.assertEqual(
                    tuple(param.type_ for param in route.dependant.body_params),
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
            "modelType": "linears",
            "model": "linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {"hidden_dim": 128, "use_bias": True},
            "search": None,
            "logFolder": "api_schema",
            "isRandomSearch": False,
            "runs": [],
            "summary": {
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
            },
        }

        request = SubmittedTrainingRunPlanRequest.model_validate(base_payload)

        self.assertEqual(request.overrides, {"hidden_dim": 128, "use_bias": True})
        with self.assertRaises(ValidationError):
            SubmittedTrainingRunPlanRequest.model_validate(
                {
                    **base_payload,
                    "overrides": {"scheduler": {"name": "cosine"}},
                }
            )

    def test_training_run_plan_rejects_nested_override_object_at_api_boundary(
        self,
    ) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/run-plan",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["Mnist"],
                            "overrides": {"scheduler": {"name": "cosine"}},
                            "logFolder": "api_schema",
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 422)
        self.assertIn("overrides", response.text)

    def test_training_run_plan_rejects_overlarge_search_axis(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/run-plan",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "search_limit",
                            "search": {
                                "mode": "grid",
                                "values": {
                                    "stack_hidden_dim": [128 for _ in range(51)]
                                },
                            },
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 400)
        self.assertIn("accepts at most 50", response.text)

    def test_training_run_plan_rejects_overlarge_grid_plan(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/run-plan",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "search_limit",
                            "search": {
                                "mode": "grid",
                                "values": {
                                    "learning_rate": [0.0001, 0.001, 0.01],
                                    "stack_hidden_dim": [16, 32, 64, 128, 256, 512],
                                    "stack_num_layers": [2, 4, 8, 16, 32],
                                    "stack_dropout_probability": [
                                        0.0,
                                        0.1,
                                        0.2,
                                        0.3,
                                        0.4,
                                        0.5,
                                    ],
                                    "stack_layer_norm_position": [
                                        "DISABLED",
                                        "DEFAULT",
                                        "BEFORE",
                                        "AFTER",
                                    ],
                                },
                            },
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 400)
        self.assertIn("planned runs exceeds 2000", response.text)

    def test_training_run_plan_deduplicates_search_values(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/run-plan",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "search_limit",
                            "search": {
                                "mode": "grid",
                                "values": {"stack_hidden_dim": [128, 128, 128]},
                            },
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["summary"]["totalRuns"], 1)
        self.assertEqual(response.json()["search"]["values"]["stack_hidden_dim"], [128])

    def test_training_random_search_samples_without_rejecting_large_grid(
        self,
    ) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/run-plan",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "search_limit",
                            "search": {
                                "mode": "random",
                                "randomSamples": 3,
                                "values": {
                                    "learning_rate": [0.0001, 0.001, 0.01],
                                    "stack_hidden_dim": [16, 32, 64, 128, 256, 512],
                                    "stack_num_layers": [2, 4, 8, 16, 32],
                                    "stack_dropout_probability": [
                                        0.0,
                                        0.1,
                                        0.2,
                                        0.3,
                                        0.4,
                                        0.5,
                                    ],
                                    "stack_layer_norm_position": [
                                        "DISABLED",
                                        "DEFAULT",
                                        "BEFORE",
                                        "AFTER",
                                    ],
                                },
                            },
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["summary"]["totalRuns"], 3)

    def test_training_run_plan_rejects_overlarge_aggregate_plan(
        self,
    ) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/run-plan",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "presets": ["baseline", "gating"],
                            "datasets": ["Mnist", "Cifar10"],
                            "overrides": {},
                            "logFolder": "search_limit",
                            "search": {
                                "mode": "random",
                                "randomSamples": 1000,
                                "values": {
                                    "learning_rate": [0.0001, 0.001, 0.01],
                                    "stack_hidden_dim": [16, 32, 64, 128, 256, 512],
                                    "stack_num_layers": [2, 4, 8, 16, 32],
                                    "stack_dropout_probability": [
                                        0.0,
                                        0.1,
                                        0.2,
                                        0.3,
                                        0.4,
                                        0.5,
                                    ],
                                    "stack_layer_norm_position": [
                                        "DISABLED",
                                        "DEFAULT",
                                        "BEFORE",
                                        "AFTER",
                                    ],
                                },
                            },
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 400)
        self.assertIn("4000 planned runs exceeds 2000", response.text)

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

    def test_training_create_and_run_plan_responses_preserve_public_shape(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    run_plan_response = await client.post(
                        "/training/run-plan",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "presets": ["baseline"],
                            "datasets": ["Mnist"],
                            "overrides": {"hidden_dim": "128"},
                            "logFolder": "api_schema",
                            "search": None,
                        },
                    )
                    create_response = await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "presets": ["baseline"],
                            "datasets": ["Mnist"],
                            "overrides": {"hidden_dim": "128"},
                            "logFolder": "api_schema",
                            "monitors": ["linear"],
                            "search": None,
                            "runPlan": run_plan_response.json(),
                        },
                    )
                    return run_plan_response, create_response

            run_plan_response, create_response = asyncio.run(call_api())

        self.assertEqual(run_plan_response.status_code, 200, run_plan_response.text)
        run_plan_payload = run_plan_response.json()
        self.assertEqual(
            tuple(run_plan_payload),
            EXPECTED_TRAINING_RUN_PLAN_RESPONSE_FIELDS,
        )
        self.assertEqual(run_plan_payload["modelType"], "linears")
        self.assertEqual(run_plan_payload["model"], "linear")
        self.assertEqual(run_plan_payload["preset"], "baseline")
        self.assertEqual(run_plan_payload["presets"], ["baseline"])
        self.assertEqual(run_plan_payload["datasets"], ["Mnist"])
        self.assertEqual(run_plan_payload["logFolder"], "api_schema")
        self.assertEqual(len(run_plan_payload["runs"]), 1)
        self.assertEqual(
            tuple(run_plan_payload["runs"][0]),
            EXPECTED_TRAINING_RUN_RESPONSE_FIELDS,
        )

        self.assertEqual(create_response.status_code, 200, create_response.text)
        create_payload = create_response.json()
        self.assertEqual(tuple(create_payload), EXPECTED_TRAINING_JOB_RESPONSE_FIELDS)
        self.assertEqual(create_payload["status"], "running")
        self.assertEqual(create_payload["modelType"], "linears")
        self.assertEqual(create_payload["model"], "linear")
        self.assertEqual(create_payload["preset"], "baseline")
        self.assertEqual(create_payload["presets"], ["baseline"])
        self.assertEqual(create_payload["datasets"], ["Mnist"])
        self.assertEqual(create_payload["monitors"], ["linear"])
        self.assertEqual(create_payload["logFolder"], "api_schema")
        self.assertEqual(create_payload["plannedRunCount"], 1)
        self.assertEqual(tuple(create_payload["runPlan"]), tuple(run_plan_payload))
        self.assertEqual(create_payload["events"][-1]["type"], "job_started")
        self.assertEqual(create_payload["resultLinks"], [])
        for internal_key in ("command", "root", "process"):
            self.assertNotIn(internal_key, create_payload)

    def test_training_job_events_endpoint_paginates_progress_history(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, manager = self._create_test_app(Path(tmp))

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    create_response = await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "history_api",
                            "monitors": [],
                        },
                    )
                    job_id = create_response.json()["id"]
                    job = manager.jobs[job_id]
                    for step in range(5):
                        manager._write_event(
                            job,
                            {
                                "type": "step",
                                "status": "running",
                                "dataset": "Mnist",
                                "preset": "baseline",
                                "runIndex": 1,
                                "step": step,
                            },
                        )
                    events_response = await client.get(
                        f"/training/jobs/{job_id}/events?offset=2&limit=3"
                    )
                    return create_response, events_response

            create_response, events_response = asyncio.run(call_api())

        self.assertEqual(create_response.status_code, 200, create_response.text)
        self.assertEqual(events_response.status_code, 200, events_response.text)
        payload = events_response.json()
        self.assertEqual(payload["offset"], 2)
        self.assertEqual(payload["limit"], 3)
        self.assertEqual(payload["totalCount"], 6)
        self.assertEqual(payload["nextOffset"], 5)
        self.assertEqual([event["step"] for event in payload["events"]], [1, 2, 3])

    def test_training_run_plan_rejects_path_like_dataset_input(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/run-plan",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["./Mnist"],
                            "overrides": {},
                            "logFolder": "path_like_dataset",
                            "search": None,
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 400)
        self.assertIn("./Mnist", response.json()["detail"])
        self.assertIn("filesystem path", response.json()["detail"])
        self.assertIn("server-known dataset name", response.json()["detail"])

    def test_training_job_rejects_path_like_dataset_input_before_side_effects(
        self,
    ) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app, manager = self._create_test_app(root)

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["./Mnist"],
                            "overrides": {},
                            "logFolder": "path_like_dataset",
                            "monitors": [],
                        },
                    )

            response = asyncio.run(call_api())

            self.assertEqual(manager.jobs, {})
            self.assertEqual(manager.active_jobs(), [])
            self.assertFalse((root / "jobs").exists())
            self.assertFalse((root / "logs" / "path_like_dataset").exists())

        self.assertEqual(response.status_code, 400)
        self.assertIn("./Mnist", response.json()["detail"])
        self.assertIn("filesystem path", response.json()["detail"])
        self.assertIn("server-known dataset name", response.json()["detail"])

    def test_training_cancel_endpoint_preserves_lifecycle_behavior(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess()
            app, _manager = self._create_test_app(Path(tmp), process=process)

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    create_response = await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "api_cancel",
                            "monitors": [],
                        },
                    )
                    cancel_response = await client.post(
                        f"/training/jobs/{create_response.json()['id']}/cancel"
                    )
                    return create_response, cancel_response

            create_response, cancel_response = asyncio.run(call_api())

        self.assertEqual(create_response.status_code, 200, create_response.text)
        self.assertEqual(cancel_response.status_code, 200, cancel_response.text)
        job_id = create_response.json()["id"]
        payload = cancel_response.json()
        self.assertEqual(payload["id"], job_id)
        self.assertEqual(payload["status"], "cancelled")
        self.assertTrue(process.terminated)
        self.assertEqual(payload["events"][-1]["type"], "cancelled")
        self.assertEqual(payload["events"][-1]["status"], "cancelled")
        self.assertEqual(payload["events"][-1]["jobId"], job_id)
        self.assertEqual(
            [run["status"] for run in payload["runPlan"]["runs"]],
            ["Skipped"],
        )
        self.assertEqual(payload["runPlan"]["summary"]["pendingRuns"], 0)
        self.assertEqual(payload["runPlan"]["summary"]["skippedRuns"], 1)

    def test_unknown_training_job_ids_remain_inspector_error_http_400(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            app, _manager = self._create_test_app(Path(tmp))

            async def call_api() -> tuple[
                httpx.Response,
                httpx.Response,
                httpx.Response,
            ]:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    get_response = await client.get("/training/jobs/missing")
                    cancel_response = await client.post(
                        "/training/jobs/missing/cancel"
                    )
                    prefixed_response = await client.get("/v1/training/jobs/missing")
                    return get_response, cancel_response, prefixed_response

            get_response, cancel_response, prefixed_response = asyncio.run(call_api())

        for response in (get_response, cancel_response):
            with self.subTest(path=str(response.request.url)):
                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.json(),
                    {"detail": "Unknown training job 'missing'."},
                )
        self.assertEqual(prefixed_response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
