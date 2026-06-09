from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi.routing import APIRoute

from viewer.backend.api import ViewerApiSettings, create_app
from viewer.backend.schemas import (
    MonitorDataResponse,
    ParameterStatusResponse,
    TrainingJobCreateRequest,
    TrainingJobResponse,
    TrainingRunPlanCreateRequest,
    TrainingRunPlanResponse,
)
from viewer.backend.tests.helpers import FakeProcess, FakeRunner
from viewer.backend.training_jobs import TrainingJobManager


EXPECTED_TRAINING_JOB_RESPONSE_FIELDS = (
    "id",
    "status",
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
    "currentPreset",
    "currentDataset",
    "epoch",
    "step",
    "metrics",
    "logDir",
    "events",
    "logTail",
    "resultLinks",
)

EXPECTED_TRAINING_RUN_PLAN_RESPONSE_FIELDS = (
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
            ViewerApiSettings(logs_root=str(logs_root)),
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
                            "model": "linears/linear",
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
                            "model": "linears/linear",
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
        self.assertEqual(run_plan_payload["model"], "linears/linear")
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
        self.assertEqual(create_payload["model"], "linears/linear")
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
                            "model": "linears/linear",
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
                            "model": "linears/linear",
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
                            "model": "linears/linear",
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
