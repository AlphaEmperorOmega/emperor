from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from tests.support import lifespan_client as _lifespan_client

PATH_LIKE_DATASET_FIELDS = {
    "path",
    "root",
    "dir",
    "file",
    "filename",
    "relativePath",
    "absolutePath",
}


class ApiIntegrationContractTests(unittest.TestCase):
    def test_capabilities_endpoint_exposes_local_defaults(self) -> None:
        import httpx

        from emperor_workbench.api import app
        from emperor_workbench.settings import get_workbench_api_settings

        async def call_api() -> httpx.Response:
            async with _lifespan_client(app) as client:
                return await client.get("/capabilities")

        response = asyncio.run(call_api())
        cancellation_capability = response.json()["trainingCancellationCapability"]

        self.assertEqual(response.request.url.path, "/capabilities")
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            cancellation_capability,
            {
                "process-group",
                "strict-cgroup",
                "windows-job-object",
                "unsupported",
            },
        )
        self.assertEqual(
            response.json(),
            {
                "authMode": "none",
                "trainingEnabled": False,
                "trainingCancellationCapability": cancellation_capability,
                "trainingResourceLimitsEnforced": (
                    cancellation_capability in {"strict-cgroup", "windows-job-object"}
                ),
                "logDeletionEnabled": False,
                "configSnapshotsEnabled": False,
                "historicalLogsEnabled": True,
                "liveMonitorDataEnabled": True,
                "historicalMonitorDataEnabled": True,
                "uploadsEnabled": False,
                "maxUploadSize": (
                    get_workbench_api_settings().effective_max_upload_size
                ),
                "maxActiveTrainingJobs": (
                    get_workbench_api_settings().max_active_training_jobs
                ),
                "trainingJobMemoryLimitBytes": (
                    get_workbench_api_settings().training_job_memory_limit_bytes
                ),
                "trainingJobCpuLimit": (
                    get_workbench_api_settings().training_job_cpu_limit
                ),
                "trainingJobProcessLimit": (
                    get_workbench_api_settings().training_job_process_limit
                ),
            },
        )

    def test_capabilities_endpoint_uses_app_scoped_training_interface(
        self,
    ) -> None:
        from unittest.mock import patch

        from emperor_workbench.settings import WorkbenchApiSettings
        from tests.support.training_jobs import (
            FakeRunner,
            TrainingJobServiceHarness,
            create_app_with_training_service,
        )

        training_jobs = TrainingJobServiceHarness(runner=FakeRunner())
        test_app = create_app_with_training_service(
            WorkbenchApiSettings(),
            training_jobs,
        )

        async def call_api():
            async with _lifespan_client(test_app) as client:
                with patch.object(
                    training_jobs.service,
                    "cancellation_capability",
                    return_value="process-group",
                ) as observe_capability:
                    response = await client.get("/capabilities")
                return response, observe_capability

        response, observe_capability = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["trainingCancellationCapability"],
            "process-group",
        )
        observe_capability.assert_called_once_with()

    def test_capabilities_endpoint_keeps_hosted_uploads_disabled_by_default(
        self,
    ) -> None:
        import httpx

        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

        app = create_app(WorkbenchApiSettings(auth_mode="bearer", token="secret-token"))

        async def call_api() -> httpx.Response:
            async with _lifespan_client(app) as client:
                return await client.get("/capabilities")

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["authMode"], "bearer")
        self.assertFalse(response.json()["uploadsEnabled"])

    def test_capabilities_endpoint_reports_hosted_upload_cap_when_enabled(
        self,
    ) -> None:
        import httpx

        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

        app = create_app(
            WorkbenchApiSettings(
                auth_mode="bearer",
                token="secret-token",
                allow_log_imports=True,
            )
        )

        async def call_api() -> httpx.Response:
            async with _lifespan_client(app) as client:
                return await client.get("/capabilities")

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["uploadsEnabled"])
        self.assertEqual(
            response.json()["maxUploadSize"],
            512 * 1024 * 1024,
        )

    def test_capabilities_endpoint_reports_local_mutation_features(self) -> None:
        import httpx

        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

        app = create_app(WorkbenchApiSettings(allow_unsafe_local_mutations=True))

        async def call_api() -> httpx.Response:
            async with _lifespan_client(app) as client:
                return await client.get("/capabilities")

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["trainingEnabled"])
        self.assertTrue(response.json()["logDeletionEnabled"])
        self.assertTrue(response.json()["configSnapshotsEnabled"])
        self.assertFalse(response.json()["uploadsEnabled"])

    def test_model_dataset_endpoint_exposes_path_free_dataset_metadata(self) -> None:
        import httpx

        from emperor_workbench.api import app

        async def call_api() -> httpx.Response:
            async with _lifespan_client(app) as client:
                return await client.get("/models/linears/linear/datasets")

        response = asyncio.run(call_api())

        self.assertEqual(response.request.url.path, "/models/linears/linear/datasets")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(
            tuple(payload),
            ("modelType", "model", "defaultExperimentTask", "datasetGroups"),
        )
        self.assertEqual(payload["modelType"], "linears")
        self.assertEqual(payload["model"], "linear")
        self.assertEqual(payload["defaultExperimentTask"], "image-classification")
        self.assertTrue(payload["datasetGroups"])

        group = payload["datasetGroups"][0]
        self.assertEqual(
            tuple(group),
            ("experimentTask", "label", "datasets"),
        )
        self.assertEqual(group["experimentTask"], "image-classification")
        self.assertEqual(group["label"], "Image Classification")
        self.assertTrue(group["datasets"])

        for dataset in group["datasets"]:
            with self.subTest(dataset=dataset.get("name")):
                self.assertEqual(
                    tuple(dataset),
                    ("name", "label", "inputDim", "outputDim"),
                )
                self.assertTrue(
                    PATH_LIKE_DATASET_FIELDS.isdisjoint(dataset),
                    f"Dataset payload exposed path-like fields: {dataset}",
                )

        dataset_by_name = {dataset["name"]: dataset for dataset in group["datasets"]}
        self.assertIn("Mnist", dataset_by_name)
        self.assertEqual(dataset_by_name["Mnist"]["inputDim"], 784)
        self.assertEqual(dataset_by_name["Mnist"]["outputDim"], 10)

    def test_retired_flat_model_routes_are_not_registered(self) -> None:
        import httpx

        from emperor_workbench.api import app

        async def call_api() -> list[httpx.Response]:
            async with _lifespan_client(app) as client:
                return [
                    await client.get(path)
                    for path in (
                        "/models/linear/presets",
                        "/models/linear/datasets",
                        "/models/linear/config-schema",
                    )
                ]

        responses = asyncio.run(call_api())

        self.assertTrue(all(response.status_code == 404 for response in responses))

    def test_api_health_and_inspect(self) -> None:
        import httpx

        from emperor_workbench.api import app

        async def call_api() -> tuple[
            httpx.Response,
            httpx.Response,
            httpx.Response,
            httpx.Response,
        ]:
            async with _lifespan_client(app) as client:
                health = await client.get("/health")
                monitors = await client.get("/models/linears/linear/monitors")
                search_space = await client.get(
                    "/models/linears/linear/search-space?preset=baseline"
                )
                inspection = await client.post(
                    "/inspect",
                    json={
                        "modelType": "linears",
                        "model": "linear",
                        "preset": "baseline",
                        "dataset": "Mnist",
                        "overrides": {"hidden_dim": "128"},
                    },
                )
                return health, monitors, search_space, inspection

        (
            health_response,
            monitors_response,
            search_space_response,
            inspection_response,
        ) = asyncio.run(call_api())
        self.assertEqual(inspection_response.status_code, 200)
        payload = inspection_response.json()
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(monitors_response.status_code, 200)
        self.assertEqual(monitors_response.json()["monitors"][0]["name"], "linear")
        self.assertEqual(search_space_response.status_code, 200)
        search_space_payload = search_space_response.json()
        self.assertIn(
            "HIDDEN_DIM", {axis["key"] for axis in search_space_payload["axes"]}
        )
        self.assertEqual(payload["modelType"], "linears")
        self.assertEqual(payload["model"], "linear")
        self.assertTrue(payload["nodes"])
        self.assertTrue(payload["edges"])
        self.assertIn("parameterCount", payload)
        self.assertIn("parameterSizeBytes", payload)
        self.assertIn("parameterCount", payload["nodes"][0])
        self.assertIn("parameterSizeBytes", payload["nodes"][0])

    def test_inspect_rejects_path_like_dataset_input(self) -> None:
        import httpx

        from emperor_workbench.api import app

        async def call_api() -> httpx.Response:
            async with _lifespan_client(app) as client:
                return await client.post(
                    "/inspect",
                    json={
                        "modelType": "linears",
                        "model": "linear",
                        "preset": "baseline",
                        "dataset": "./Mnist",
                        "overrides": {},
                    },
                )

        response = asyncio.run(call_api())
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertIn("./Mnist", detail)
        self.assertIn("filesystem path", detail)
        self.assertIn("server-known dataset name", detail)

    def test_log_scalars_rejects_more_than_max_request_run_ids(self) -> None:
        import httpx

        from emperor_workbench.api import app

        async def call_api() -> httpx.Response:
            async with _lifespan_client(app) as client:
                return await client.post(
                    "/logs/scalars",
                    json={
                        "runIds": [f"run-{index}" for index in range(94)],
                        "tags": ["train/loss"],
                        "maxPoints": 500,
                        "sampling": "tail",
                    },
                )

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"][0]
        self.assertEqual(detail["type"], "too_long")
        self.assertEqual(detail["loc"], ["body", "runIds"])
        self.assertEqual(detail["ctx"]["max_length"], 50)
        self.assertEqual(detail["ctx"]["actual_length"], 94)

    def test_api_inspector_errors_use_shared_handler(self) -> None:
        import httpx

        from emperor_workbench.api import app

        async def call_api() -> httpx.Response:
            async with _lifespan_client(app) as client:
                return await client.get("/models/unknown/model/presets")

        response = asyncio.run(call_api())
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown model", response.json()["detail"])

    def test_broken_model_package_routes_use_stable_http_errors(self) -> None:
        from unittest.mock import patch

        import httpx

        from emperor_workbench.api import create_app
        from emperor_workbench.project_adapter import (
            ProjectAdapterClient,
            ProjectAdapterFailure,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        paths = (
            "/models/broken/missing/presets",
            "/models/broken/missing/datasets",
            "/models/broken/missing/monitors",
            "/models/broken/missing/config-schema",
            "/models/broken/missing/search-space",
        )

        async def call_api(app, project_adapter) -> list[tuple[str, httpx.Response]]:
            async with _lifespan_client(app) as client:
                with patch.object(
                    project_adapter,
                    "package",
                    side_effect=ProjectAdapterFailure(
                        "Failed to import model package 'broken/missing': "
                        "missing module"
                    ),
                ):
                    return [(path, await client.get(path)) for path in paths]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_adapter = ProjectAdapterClient()
            app = create_app(
                WorkbenchApiSettings(
                    logs_root=str(root / "logs"),
                    snapshots_root=str(root / "snapshots"),
                    state_root=str(root / "state"),
                ),
                project_adapter=project_adapter,
            )
            responses = asyncio.run(call_api(app, project_adapter))

        for path, response in responses:
            with self.subTest(path=path):
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(
                    "Failed to import model package 'broken/missing'",
                    response.json()["detail"],
                )
