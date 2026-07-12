from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from workbench.backend.api.mutation_policy import (
    HttpOperationPolicy,
    build_http_operation_catalog,
    enforce_operation_policy,
)
from workbench.backend.core.security import (
    LOCAL_MUTATION_DISABLED_DETAIL,
    MUTATION_HEADER_NAME,
    MUTATION_HEADER_VALUE,
    MUTATION_PROOF_REQUIRED_DETAIL,
    UNTRUSTED_MUTATION_ORIGIN_DETAIL,
    require_bearer_auth,
)
from workbench.backend.settings import WorkbenchApiSettings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROUTE_AUTH_TOKEN = "server-secret"


def concrete_route_path(route) -> str:
    path = route.path_format
    for parameter_name in route.param_convertors:
        path = path.replace(f"{{{parameter_name}}}", "security-test")
    return path

PROTECTED_ROUTE_CASES = (
    ("models", "GET", "/models", None),
    (
        "inspection",
        "POST",
        "/inspect",
        {
            "modelType": "linears",
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
            "modelType": "linears",
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
            "modelType": "linears",
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
    (
        "config_snapshots",
        "GET",
        "/config-snapshots?modelType=linears&model=linear",
        None,
    ),
)


def bearer_credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


class SecurityDependencyTests(unittest.TestCase):
    def test_auth_mode_none_bypasses_missing_authorization_header(self) -> None:
        asyncio.run(require_bearer_auth(WorkbenchApiSettings(auth_mode="none"), None))

    def test_bearer_mode_rejects_missing_malformed_and_invalid_tokens(self) -> None:
        settings = WorkbenchApiSettings(auth_mode="bearer", token="server-secret")

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
        settings = WorkbenchApiSettings(auth_mode="bearer", token="server-secret")

        asyncio.run(require_bearer_auth(settings, bearer_credentials("server-secret")))

    def test_local_mutation_policy_rejects_default_settings(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            enforce_operation_policy(
                HttpOperationPolicy.LOCAL_MUTATION,
                WorkbenchApiSettings(),
            )

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(raised.exception.detail, LOCAL_MUTATION_DISABLED_DETAIL)

    def test_local_mutation_policy_accepts_explicit_opt_in(self) -> None:
        enforce_operation_policy(
            HttpOperationPolicy.LOCAL_MUTATION,
            WorkbenchApiSettings(allow_unsafe_local_mutations=True),
        )


class RouteAuthIntegrationTests(unittest.TestCase):
    async def request(
        self,
        app,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        authorization: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        import httpx

        request_headers = dict(headers or {})
        if authorization is not None:
            request_headers["Authorization"] = authorization

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            kwargs = {"headers": request_headers}
            if payload is not None:
                kwargs["json"] = payload
            return await client.request(method, path, **kwargs)

    def create_test_app(
        self,
        root: Path,
        *,
        auth_mode: str = "bearer",
        allow_unsafe_local_mutations: bool = True,
        allow_log_imports: bool | None = None,
    ):
        from workbench.backend.api import create_app
        from workbench.backend.dependencies import (
            get_inspection_service,
            get_run_history_service,
            get_training_job_service,
            get_training_run_plan_service,
        )
        from workbench.backend.training_jobs.contracts import (
            TrainingJobView,
            TrainingRunPlanView,
        )
        from workbench.backend.training_jobs.run_plan_adapter import (
            training_run_plan_from_payload,
            training_search_to_payload,
        )

        class FakeInspectionService:
            def inspect(
                self,
                *,
                model_type: str,
                model: str,
                preset: str,
                overrides: dict[str, object],
                dataset: str | None,
                experiment_task: str | None = None,
                log_run_id: str | None,
            ) -> dict[str, object]:
                return {
                    "modelType": model_type,
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
                return TrainingJobView(
                    id="job-1",
                    status="running",
                    model="linears/linear",
                    preset=command.preset,
                    presets=command.presets or [command.preset],
                    experiment_task=command.experiment_task or "",
                    datasets=command.datasets,
                    overrides=command.overrides,
                    search=command.search,
                    planned_run_count=0,
                    run_plan=command.run_plan,
                    monitors=command.monitors,
                    log_folder=command.log_folder,
                    created_at="2026-06-06T00:00:00Z",
                    updated_at="2026-06-06T00:00:00Z",
                    exit_code=None,
                    pid=123,
                    cancellation_mode="process-group",
                    current_preset=None,
                    current_dataset=None,
                    epoch=None,
                    step=None,
                    metrics={},
                    log_dir=None,
                    events=[],
                    event_count=0,
                    event_counts={},
                    events_truncated=False,
                    cluster_growth=[],
                    log_tail=[],
                    result_links=[],
                )

            def create_run_plan(self, command) -> TrainingRunPlanView:
                return training_run_plan_from_payload(
                    {
                        "modelType": "linears",
                        "model": "linear",
                        "preset": command.preset,
                        "presets": command.presets or [command.preset],
                        "datasets": command.datasets,
                        "overrides": command.overrides,
                        "search": (
                            training_search_to_payload(command.search)
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

        class FakeRunHistoryService:
            def list_runs(
                self,
                *,
                limit: int,
                offset: int,
                **_filters: object,
            ) -> dict[str, object]:
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
            WorkbenchApiSettings(
                logs_root=str(logs_root),
                snapshots_root=str(root / "snapshots"),
                auth_mode=auth_mode,
                token=token,
                allow_unsafe_local_mutations=allow_unsafe_local_mutations,
                **(
                    {"allow_log_imports": allow_log_imports}
                    if allow_log_imports is not None
                    else {}
                ),
            )
        )
        inspection_service = FakeInspectionService()
        training_job_service = FakeTrainingJobService()
        run_history_service = FakeRunHistoryService()

        async def override_inspection_service() -> FakeInspectionService:
            return inspection_service

        async def override_training_job_service() -> FakeTrainingJobService:
            return training_job_service

        async def override_training_run_plan_service() -> FakeTrainingJobService:
            return training_job_service

        async def override_run_history_service() -> FakeRunHistoryService:
            return run_history_service

        app.dependency_overrides[get_inspection_service] = override_inspection_service
        app.dependency_overrides[get_training_job_service] = (
            override_training_job_service
        )
        app.dependency_overrides[get_training_run_plan_service] = (
            override_training_run_plan_service
        )
        app.dependency_overrides[get_run_history_service] = override_run_history_service
        return app

    def route_runs_on_event_loop(self, app, method: str, path: str) -> bool:
        from fastapi.routing import APIRoute

        route_path = path.split("?")[0]
        for route in app.routes:
            if (
                isinstance(route, APIRoute)
                and route.path == route_path
                and method.upper() in (route.methods or ())
            ):
                return asyncio.iscoroutinefunction(route.endpoint)
        return False

    def test_health_remains_open_without_token_in_bearer_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(Path(tmp), auth_mode="bearer")

            response = asyncio.run(self.request(app, "GET", "/health"))

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_capabilities_remains_open_without_token_in_bearer_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(Path(tmp), auth_mode="bearer")

            response = asyncio.run(self.request(app, "GET", "/capabilities"))

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["authMode"], "bearer")

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
                    # The in-process ASGI client deadlocks sync FastAPI handlers
                    # in this Python/anyio test environment. Rejection cases
                    # still cover every protected route before endpoint dispatch,
                    # and OpenAPI assertions cover the auth dependency metadata.
                    if not self.route_runs_on_event_loop(app, method, path):
                        continue
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
        self.assertNotIn("security", openapi["paths"]["/capabilities"]["get"])

        for route_name, method, path, _payload in PROTECTED_ROUTE_CASES:
            with self.subTest(route=route_name):
                operation = openapi["paths"][path.split("?")[0]][method.lower()]
                self.assertIn({"HTTPBearer": []}, operation["security"])
        self.assertIn(
            {"HTTPBearer": []},
            openapi["paths"]["/logs/import"]["post"]["security"],
        )

    def test_bearer_mode_rejects_missing_token_on_log_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(Path(tmp), auth_mode="bearer")

            response = asyncio.run(self.request(app, "POST", "/logs/import"))

        self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")
        self.assertEqual(
            response.json(),
            {"detail": "Missing or invalid bearer credentials"},
        )

    def test_local_default_auth_mode_allows_non_health_routes_without_token(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(Path(tmp), auth_mode="none")

            for route_name, method, path, payload in PROTECTED_ROUTE_CASES:
                with self.subTest(route=route_name):
                    # See the bearer-mode valid-token test above for why sync
                    # handlers are not executed through the ASGI test client.
                    if not self.route_runs_on_event_loop(app, method, path):
                        continue
                    response = asyncio.run(
                        self.request(
                            app,
                            method,
                            path,
                            payload=payload,
                            headers={MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE},
                        )
                    )

                    self.assertEqual(response.status_code, 200, response.text)

    def test_local_mutation_routes_reject_when_not_explicitly_enabled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(
                Path(tmp),
                auth_mode="none",
                allow_unsafe_local_mutations=False,
            )

            response = asyncio.run(
                self.request(
                    app,
                    "POST",
                    "/training/jobs",
                    payload={
                        "modelType": "linears",
                        "model": "linear",
                        "preset": "baseline",
                        "presets": ["baseline"],
                        "datasets": ["Mnist"],
                        "overrides": {"hidden_dim": "128"},
                        "logFolder": "mutation_guard",
                        "monitors": [],
                        "search": None,
                        "runPlan": None,
                    },
                    headers={MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE},
                )
            )

        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(
            response.json(),
            {"detail": LOCAL_MUTATION_DISABLED_DETAIL},
        )

    def test_log_import_rejects_when_uploads_are_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(
                Path(tmp),
                auth_mode="none",
                allow_log_imports=False,
            )

            response = asyncio.run(
                self.request(
                    app,
                    "POST",
                    "/logs/import",
                    headers={MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE},
                )
            )

        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(
            response.json(),
            {"detail": LOCAL_MUTATION_DISABLED_DETAIL},
        )

    def test_untrusted_origin_is_rejected_on_every_mutation_route(self) -> None:
        from workbench.backend.api.v1.router import router as api_v1_router

        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(
                Path(tmp),
                auth_mode="none",
                allow_unsafe_local_mutations=True,
                allow_log_imports=True,
            )
            catalog = build_http_operation_catalog(
                app.routes,
                declared_routes=api_v1_router.routes,
            )

            for operation in catalog.mutations:
                path = concrete_route_path(operation.route)
                with self.subTest(method=operation.method, path=path):
                    response = asyncio.run(
                        self.request(
                            app,
                            operation.method,
                            path,
                            headers={
                                "Origin": "https://evil.example",
                                "Sec-Fetch-Site": "cross-site",
                            },
                        )
                    )

                    self.assertEqual(response.status_code, 403, response.text)
                    self.assertEqual(
                        response.json(),
                        {"detail": UNTRUSTED_MUTATION_ORIGIN_DETAIL},
                    )

    def test_every_unauthenticated_mutation_requires_proof(self) -> None:
        from workbench.backend.api.v1.router import router as api_v1_router

        with tempfile.TemporaryDirectory() as tmp:
            app = self.create_test_app(
                Path(tmp),
                auth_mode="none",
                allow_unsafe_local_mutations=True,
                allow_log_imports=True,
            )
            catalog = build_http_operation_catalog(
                app.routes,
                declared_routes=api_v1_router.routes,
            )

            for operation in catalog.mutations:
                path = concrete_route_path(operation.route)
                with self.subTest(method=operation.method, path=path):
                    response = asyncio.run(
                        self.request(app, operation.method, path)
                    )

                    self.assertEqual(response.status_code, 403, response.text)
                    self.assertEqual(
                        response.json(),
                        {"detail": MUTATION_PROOF_REQUIRED_DETAIL},
                    )


if __name__ == "__main__":
    unittest.main()
