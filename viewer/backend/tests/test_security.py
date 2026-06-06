from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from fastapi import HTTPException
from starlette.datastructures import Headers

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
            "model": "linear",
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
            "model": "linear",
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
            "model": "linear",
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
    ("config_snapshots", "GET", "/config-snapshots?model=linear", None),
)


def request_with_settings(
    settings: ViewerApiSettings,
    authorization: str | None = None,
) -> SimpleNamespace:
    headers = Headers(
        {} if authorization is None else {"Authorization": authorization}
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(settings=settings)),
        headers=headers,
    )


class SecurityDependencyTests(unittest.TestCase):
    def test_auth_mode_none_bypasses_missing_authorization_header(self) -> None:
        request = request_with_settings(ViewerApiSettings(auth_mode="none"))

        require_bearer_auth(request)

    def test_bearer_mode_rejects_missing_malformed_and_invalid_tokens(self) -> None:
        settings = ViewerApiSettings(auth_mode="bearer", token="server-secret")

        cases = (
            None,
            "",
            "Bearer",
            "Bearer ",
            "Bearer server-secret extra",
            "Token server-secret",
            "Bearer wrong-token",
        )

        for authorization in cases:
            with self.subTest(authorization=authorization):
                request = request_with_settings(settings, authorization)

                with self.assertRaises(HTTPException) as raised:
                    require_bearer_auth(request)

                self.assertEqual(raised.exception.status_code, 401)
                self.assertNotIn("wrong-token", str(raised.exception.detail))

    def test_bearer_mode_accepts_configured_token(self) -> None:
        settings = ViewerApiSettings(auth_mode="bearer", token="server-secret")
        request = request_with_settings(settings, "Bearer server-secret")

        require_bearer_auth(request)


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
                    "nodes": [
                        {
                            "id": "root",
                            "label": "Root",
                            "typeName": "FakeModel",
                            "path": "root",
                            "graphRole": "architecture",
                            "parameterCount": 0,
                            "details": {},
                            "config": None,
                        }
                    ],
                    "edges": [],
                }

        class FakeTrainingJobService:
            def create_job(
                self,
                *,
                model: str,
                preset: str,
                presets: list[str] | None,
                datasets: list[str],
                overrides: dict[str, object],
                log_folder: str,
                monitors: list[str],
                search: dict[str, object] | None,
                run_plan: dict[str, object] | None,
            ) -> dict[str, object]:
                return {
                    "id": "job-1",
                    "status": "running",
                    "model": model,
                    "preset": preset,
                    "presets": presets or [preset],
                    "datasets": datasets,
                    "overrides": overrides,
                    "search": search,
                    "plannedRunCount": 0,
                    "runPlan": run_plan,
                    "monitors": monitors,
                    "logFolder": log_folder,
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

            def create_run_plan(
                self,
                *,
                model: str,
                preset: str,
                presets: list[str] | None,
                datasets: list[str],
                overrides: dict[str, object],
                log_folder: str,
                search: dict[str, object] | None,
            ) -> dict[str, object]:
                return {
                    "model": model,
                    "preset": preset,
                    "presets": presets or [preset],
                    "datasets": datasets,
                    "overrides": overrides,
                    "search": search,
                    "logFolder": log_folder,
                    "isRandomSearch": False,
                    "runs": [
                        {
                            "id": "run-1",
                            "index": 0,
                            "status": "Pending",
                            "preset": preset,
                            "dataset": datasets[0] if datasets else "",
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
        app.dependency_overrides[get_model_catalog_service] = FakeModelCatalogService
        app.dependency_overrides[get_inspection_service] = FakeInspectionService
        app.dependency_overrides[get_training_job_service] = FakeTrainingJobService
        app.dependency_overrides[get_log_run_service] = FakeLogRunService
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

    def test_local_default_auth_mode_allows_non_health_routes_without_token(self) -> None:
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
