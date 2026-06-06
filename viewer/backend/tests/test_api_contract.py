from __future__ import annotations

import asyncio
import unittest

from fastapi.routing import APIRoute

from viewer.backend import schemas
from viewer.backend.api import app

EXPECTED_BUSINESS_ROUTES = [
    (("DELETE",), "/logs/experiments/{experiment}"),
    (("GET",), "/health"),
    (("GET",), "/logs/experiments"),
    (("GET",), "/logs/runs"),
    (("GET",), "/logs/runs/{run_id}/monitor-data"),
    (("GET",), "/models"),
    (("GET",), "/models/{model}/config-schema"),
    (("GET",), "/models/{model}/datasets"),
    (("GET",), "/models/{model}/monitors"),
    (("GET",), "/models/{model}/presets"),
    (("GET",), "/models/{model}/search-space"),
    (("GET",), "/training/jobs/{job_id}"),
    (("GET",), "/training/jobs/{job_id}/monitor-data"),
    (("POST",), "/inspect"),
    (("POST",), "/logs/runs/delete"),
    (("POST",), "/logs/runs/delete-plan"),
    (("POST",), "/logs/scalars"),
    (("POST",), "/logs/tags"),
    (("POST",), "/training/jobs"),
    (("POST",), "/training/jobs/{job_id}/cancel"),
    (("POST",), "/training/run-plan"),
]

FRONTEND_COUPLED_SCHEMA_FIELDS = (
    (
        schemas.TrainingJobCreateRequest,
        (
            "model",
            "preset",
            "presets",
            "datasets",
            "overrides",
            "logFolder",
            "monitors",
            "search",
            "runPlan",
        ),
    ),
    (
        schemas.TrainingRunPlanCreateRequest,
        (
            "model",
            "preset",
            "presets",
            "datasets",
            "overrides",
            "logFolder",
            "search",
        ),
    ),
    (
        schemas.TrainingRunResponse,
        (
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
        ),
    ),
    (
        schemas.TrainingRunPlanResponse,
        (
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
        ),
    ),
    (
        schemas.TrainingJobResponse,
        (
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
        ),
    ),
    (
        schemas.MonitorDataResponse,
        (
            "jobId",
            "nodePath",
            "preset",
            "dataset",
            "logDir",
            "scalarSeries",
            "histograms",
            "images",
        ),
    ),
    (
        schemas.LogRunResponse,
        (
            "id",
            "group",
            "experiment",
            "model",
            "preset",
            "dataset",
            "runName",
            "timestamp",
            "version",
            "relativePath",
            "hasResult",
            "eventFileCount",
            "checkpointCount",
            "hasHparams",
            "metrics",
        ),
    ),
    (
        schemas.LogRunDeleteFiltersRequest,
        (
            "experiments",
            "datasets",
            "models",
            "presets",
            "runIds",
        ),
    ),
    (
        schemas.LogRunDeletePlanResponse,
        (
            "candidateCount",
            "counts",
            "affected",
            "candidates",
            "blockedByActiveJobs",
            "canDelete",
        ),
    ),
    (
        schemas.LogRunDeleteResponse,
        (
            "candidateCount",
            "counts",
            "affected",
            "candidates",
            "blockedByActiveJobs",
            "canDelete",
            "deletedRunIds",
            "deletedRunCount",
            "deletedRelativePaths",
        ),
    ),
)


class ApiRouteContractTests(unittest.TestCase):
    def test_api_routes_declare_response_models(self) -> None:
        missing = [
            f"{sorted(route.methods)} {route.path}"
            for route in app.routes
            if isinstance(route, APIRoute) and route.response_model is None
        ]

        self.assertEqual(missing, [])

    def test_api_route_inventory_preserves_current_contract(self) -> None:
        business_prefixes = ("/health", "/models", "/inspect", "/logs", "/training")
        routes = sorted(
            (tuple(sorted(route.methods or ())), route.path)
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path.startswith(business_prefixes)
        )

        self.assertEqual(routes, EXPECTED_BUSINESS_ROUTES)
        self.assertFalse(
            any(path.startswith("/v1/") or path == "/v1" for _methods, path in routes)
        )


class ApiSchemaContractTests(unittest.TestCase):
    def test_frontend_coupled_schema_field_inventory_is_stable(self) -> None:
        for model, expected_fields in FRONTEND_COUPLED_SCHEMA_FIELDS:
            with self.subTest(model=model.__name__):
                self.assertEqual(tuple(model.model_fields), expected_fields)

    def test_frontend_coupled_schemas_reject_extra_fields(self) -> None:
        for model, _expected_fields in FRONTEND_COUPLED_SCHEMA_FIELDS:
            with self.subTest(model=model.__name__):
                self.assertEqual(model.model_config.get("extra"), "forbid")


class ApiIntegrationContractTests(unittest.TestCase):
    def test_api_health_and_inspect(self) -> None:
        import httpx
        from viewer.backend.api import app

        async def call_api() -> tuple[
            httpx.Response,
            httpx.Response,
            httpx.Response,
            httpx.Response,
        ]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                health = await client.get("/health")
                monitors = await client.get("/models/linear/monitors")
                search_space = await client.get(
                    "/models/linear/search-space?preset=baseline"
                )
                inspect_response = await client.post(
                    "/inspect",
                    json={
                        "model": "linear",
                        "preset": "baseline",
                        "dataset": "Mnist",
                        "overrides": {"hidden_dim": "128"},
                    },
                )
                return health, monitors, search_space, inspect_response

        health_response, monitors_response, search_space_response, response = (
            asyncio.run(call_api())
        )
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(monitors_response.status_code, 200)
        self.assertEqual(monitors_response.json()["monitors"][0]["name"], "linear")
        self.assertEqual(search_space_response.status_code, 200)
        search_space_payload = search_space_response.json()
        self.assertIn(
            "hidden_dim", {axis["key"] for axis in search_space_payload["axes"]}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "linear")
        self.assertTrue(payload["nodes"])
        self.assertTrue(payload["edges"])
        self.assertIn("parameterCount", payload)
        self.assertIn("parameterCount", payload["nodes"][0])

    def test_api_dependency_overrides_can_replace_route_services(self) -> None:
        import httpx
        from viewer.backend.api import create_app
        from viewer.backend.dependencies import get_model_catalog_service

        class FakeModelCatalogService:
            def list_models(self) -> list[str]:
                return ["override_model"]

        async def override_model_catalog_service() -> FakeModelCatalogService:
            return FakeModelCatalogService()

        test_app = create_app()
        test_app.dependency_overrides[get_model_catalog_service] = (
            override_model_catalog_service
        )

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=test_app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/models")

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": ["override_model"]})

    def test_api_inspector_errors_use_shared_handler(self) -> None:
        import httpx
        from viewer.backend.api import app

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/models/not_a_model/presets")

        response = asyncio.run(call_api())
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown model", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
