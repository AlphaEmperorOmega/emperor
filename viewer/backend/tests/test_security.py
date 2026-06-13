from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from viewer.backend.core.security import require_bearer_auth
from viewer.backend.settings import ViewerApiSettings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROUTE_AUTH_TOKEN = "server-secret"

PROTECTED_ROUTE_CASES = (
    ("capabilities", "GET", "/capabilities", None),
    ("models", "GET", "/models", None),
    (
        "inspection",
        "POST",
        "/inspect",
        {
            "model": "linears/linear",
            "preset": "baseline",
            "dataset": "Mnist",
            "overrides": {"hidden_dim": "128"},
        },
    ),
    (
        "training",
        "POST",
        "/training/run-plan",
        {
            "model": "linears/linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {"hidden_dim": "128"},
            "logFolder": "auth_route",
            "search": None,
        },
    ),
    (
        "training_job",
        "POST",
        "/training/jobs",
        {
            "model": "linears/linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {"hidden_dim": "128"},
            "logFolder": "auth_route",
            "monitors": [],
            "search": None,
            "runPlan": None,
        },
    ),
    ("logs", "GET", "/logs/runs", None),
    ("config_snapshots", "GET", "/config-snapshots?model=linears/linear", None),
)


def bearer_credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


class SecurityDependencyTests(unittest.TestCase):
    def test_auth_mode_none_bypasses_missing_authorization_header(self) -> None:
        asyncio.run(require_bearer_auth(ViewerApiSettings(auth_mode="none"), None))

    def test_bearer_mode_rejects_missing_malformed_and_invalid_tokens(self) -> None:
        settings = ViewerApiSettings(auth_mode="bearer", token="server-secret")

        cases = (
            None,
            HTTPAuthorizationCredentials(scheme="Token", credentials="server-secret"),
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=""),
            bearer_credentials("wrong-token"),
        )

        for credentials in cases:
            with self.subTest(credentials=credentials):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(require_bearer_auth(settings, credentials))

                self.assertEqual(raised.exception.status_code, 401)
                self.assertEqual(
                    raised.exception.headers,
                    {"WWW-Authenticate": "Bearer"},
                )
                self.assertNotIn("wrong-token", str(raised.exception.detail))

    def test_bearer_mode_accepts_configured_token(self) -> None:
        settings = ViewerApiSettings(auth_mode="bearer", token="server-secret")

        asyncio.run(require_bearer_auth(settings, bearer_credentials("server-secret")))


class RouteAuthIntegrationTests(unittest.TestCase):
    async def request(
        self,
        app,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        authorization: str | None = None,
    ):
        import httpx

        headers = {}
        if authorization is not None:
            headers["Authorization"] = authorization

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            kwargs = {"headers": headers}
            if payload is not None:
                kwargs["json"] = payload
            return await client.request(method, path, **kwargs)

    def create_test_app(self, root: Path, *, auth_mode: str = "bearer"):
        from viewer.backend.api import create_app
        from viewer.backend.dependencies import (
            get_inspection_service,
            get_log_run_service,
            get_model_catalog_service,
            get_training_job_service,
        )
        from viewer.backend.training_contracts import (
            TrainingJobView,
            TrainingRunPlanView,
        )

        class FakeModelCatalogService:
            def list_models(self) -> list[str]:
                return ["linear"]

        class FakeInspectionService:
            def inspect(
                self,
                *,
                model: str,
                preset: str,
                overrides: dict[str, object],
                dataset: str | None,
            ) -> dict[str, object]:
                return {
                    "model": model,
                    "preset": preset,
                    "parameterCount": 0,
                    "parameterSizeBytes": 0,
                    "nodes": [
                        {
                            "id": "root",
                            "label": "Root",
                            "typeName": "FakeModel",
                            "path": "root",
                            "graphRole": "architecture",
                            "parameterCount": 0,
                            "parameterSizeBytes": 0,
                            "details": {},
                            "config": None,
                        }
                    ],
                    "edges": [],
                }

        class FakeTrainingJobService:
            def create_job(self, command) -> TrainingJobView:
                return TrainingJobView.from_payload(
                    {
                        "id": "job-1",
                        "status": "running",
                        "model": command.model,
                        "preset": command.preset,
                        "presets": command.presets or [command.preset],
                        "datasets": command.datasets,
                        "overrides": command.overrides,
                        "search": (
                            command.search.to_api_payload()
                            if command.search is not None
                            else None
                        ),
                        "plannedRunCount": 0,
                        "runPlan": (
                            command.run_plan.to_api_payload()
                            if command.run_plan is not None
                            else None
                        ),
                        "monitors": command.monitors,
                        "logFolder": command.log_folder,
                        "createdAt": "2026-06-06T00:00:00Z",
                        "updatedAt": "2026-06-06T00:00:00Z",
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
                )

            def create_run_plan(self, command) -> TrainingRunPlanView:
                return TrainingRunPlanView.from_payload(
                    {
                        "model": command.model,
                        "preset": command.preset,
                        "presets": command.presets or [command.preset],
                        "datasets": command.datasets,
                        "overrides": command.overrides,
                        "search": (
                            command.search.to_api_payload()
                            if command.search is not None
                            else None
                        ),
                        "logFolder": command.log_folder,
                        "isRandomSearch": False,
                        "runs": [
                            {
                                "id": "run-1",
                                "index": 0,
                                "status": "Pending",
                                "preset": command.preset,
                                "dataset": (
                                    command.datasets[0] if command.datasets else ""
                                ),
                                "changes": [],
                                "overrides": {},
                                "command": "train",
                                "totalEpochs": 1,
                            }
                        ],
                        "summary": {
                            "totalRuns": 1,
                            "pendingRuns": 1,
                            "totalEpochs": 1,
                            "remainingEpochs": 1,
                        },
                    }
                )

        class FakeLogRunService:
            def list_runs(self, *, limit: int, offset: int) -> dict[str, object]:
                return {
                    "total": 0,
                    "limit": limit,
                    "offset": offset,
                    "hasMore": False,
                    "runs": [],
                }

        logs_root = root / "logs"
        logs_root.mkdir()
        token = ROUTE_AUTH_TOKEN if auth_mode == "bearer" else None
        app = create_app(
            ViewerApiSettings(
                logs_root=str(logs_root),
                snapshots_root=str(root / "snapshots"),
                auth_mode=auth_mode,
                token=token,
            )
        )
        model_catalog_service = FakeModelCatalogService()
        inspection_service = FakeInspectionService()
        training_job_service = FakeTrainingJobService()
        log_run_service = FakeLogRunService()

        async def override_model_catalog_service() -> FakeModelCatalogService:
            return model_catalog_service

        async def override_inspection_service() -> FakeInspectionService:
            return inspection_service

        async def override_training_job_service() -> FakeTrainingJobService:
            return training_job_service

        async def override_log_run_service() -> FakeLogRunService:
            return log_run_service

        app.dependency_overrides[get_model_catalog_service] = (
            override_model_catalog_service
        )
        app.dependency_overrides[get_inspection_service] = override_inspection_service
        app.dependency_overrides[get_training_job_service] = (
            override_training_job_service
        )
        app.dependency_overrides[get_log_run_service] = override_log_run_service
        return app

    def test_health_remains_open_without_token_in_bearer_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(Path(tmp), auth_mode="bearer")

            response = asyncio.run(self.request(app, "GET", "/health"))

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_bearer_mode_rejects_missing_and_invalid_tokens_on_non_health_routes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(Path(tmp), auth_mode="bearer")

            for route_name, method, path, payload in PROTECTED_ROUTE_CASES:
                for authorization in (None, "Bearer wrong-token"):
                    with self.subTest(route=route_name, authorization=authorization):
                        response = asyncio.run(
                            self.request(
                                app,
                                method,
                                path,
                                payload=payload,
                                authorization=authorization,
                            )
                        )

                        self.assertEqual(response.status_code, 401, response.text)
                        self.assertEqual(
                            response.headers["www-authenticate"],
                            "Bearer",
                        )
                        self.assertEqual(
                            response.json(),
                            {"detail": "Missing or invalid bearer credentials"},
                        )

    def test_bearer_mode_accepts_valid_token_on_non_health_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(Path(tmp), auth_mode="bearer")

            for route_name, method, path, payload in PROTECTED_ROUTE_CASES:
                with self.subTest(route=route_name):
                    response = asyncio.run(
                        self.request(
                            app,
                            method,
                            path,
                            payload=payload,
                            authorization=f"Bearer {ROUTE_AUTH_TOKEN}",
                        )
                    )

                    self.assertEqual(response.status_code, 200, response.text)

    def test_openapi_declares_bearer_scheme_for_protected_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(Path(tmp), auth_mode="bearer")
            openapi = app.openapi()

        bearer_scheme = openapi["components"]["securitySchemes"]["HTTPBearer"]
        self.assertEqual(bearer_scheme["type"], "http")
        self.assertEqual(bearer_scheme["scheme"], "bearer")
        self.assertNotIn("security", openapi["paths"]["/health"]["get"])

        for route_name, method, path, _payload in PROTECTED_ROUTE_CASES:
            with self.subTest(route=route_name):
                operation = openapi["paths"][path.split("?")[0]][method.lower()]
                self.assertIn({"HTTPBearer": []}, operation["security"])

    def test_local_default_auth_mode_allows_non_health_routes_without_token(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(Path(tmp), auth_mode="none")

            for route_name, method, path, payload in PROTECTED_ROUTE_CASES:
                with self.subTest(route=route_name):
                    response = asyncio.run(
                        self.request(app, method, path, payload=payload)
                    )

                    self.assertEqual(response.status_code, 200, response.text)


if __name__ == "__main__":
    unittest.main()
