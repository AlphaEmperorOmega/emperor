from __future__ import annotations

import asyncio
import importlib
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import httpx
from fastapi import FastAPI
from fastapi.routing import APIRoute

from workbench.backend.core.errors import ApiError
from workbench.backend.exceptions import api_error_handler
from workbench.backend.inspector.errors import InspectorError

EXPECTED_ROOT_ROUTE_PAIRS = {
    ("GET", "/health"),
    ("GET", "/models"),
    ("POST", "/inspect"),
    ("GET", "/logs/runs"),
    ("POST", "/logs/tags"),
    ("POST", "/training/jobs"),
}

CORS_PREFLIGHT_METHODS = ("GET", "POST", "PATCH", "DELETE")
CORS_PREFLIGHT_REQUEST_HEADERS = "authorization,content-type,x-workbench-mutation"
CORS_PREFLIGHT_ALLOWED_HEADERS = {
    "authorization",
    "content-type",
    "x-workbench-mutation",
}

# Routes that do blocking CPU or file work stay async at the API boundary.
# Relying on FastAPI sync-route dispatch deadlocks under ASGITransport in this
# test environment.
EXPECTED_ASYNC_BOUNDARY_ROUTE_PAIRS = {
    ("POST", "/inspect"),
    ("GET", "/logs/runs"),
    ("GET", "/logs/experiments"),
    ("POST", "/logs/import"),
    ("POST", "/logs/checkpoints"),
    ("DELETE", "/logs/experiments/{experiment}"),
    ("POST", "/logs/runs/delete"),
    ("POST", "/logs/runs/delete-plan"),
    ("POST", "/logs/tags"),
    ("POST", "/logs/scalars"),
    ("POST", "/logs/media"),
    ("POST", "/logs/parameter-status"),
    ("GET", "/logs/runs/{run_id}/artifacts"),
    ("GET", "/logs/runs/{run_id}/monitor-data"),
    ("GET", "/models"),
    ("GET", "/models/{modelType}/{model}/config-schema"),
    ("GET", "/models/{modelType}/{model}/datasets"),
    ("GET", "/models/{modelType}/{model}/monitors"),
    ("GET", "/models/{modelType}/{model}/presets"),
    ("GET", "/models/{modelType}/{model}/search-space"),
    ("GET", "/training/jobs/{job_id}"),
    ("GET", "/training/jobs/{job_id}/events"),
    ("GET", "/training/jobs/{job_id}/monitor-data"),
    ("GET", "/training/jobs/{job_id}/monitor-parameter-status"),
    ("POST", "/training/jobs"),
    ("POST", "/training/jobs/{job_id}/cancel"),
    ("POST", "/training/run-plan"),
    ("GET", "/config-snapshots"),
    ("GET", "/config-snapshots/library"),
    ("POST", "/config-snapshots"),
    ("PATCH", "/config-snapshots/{snapshot_id}"),
    ("DELETE", "/config-snapshots/{snapshot_id}"),
}


def business_route_pairs(api: FastAPI) -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in api.routes
        if isinstance(route, APIRoute)
        for method in route.methods or ()
    }


async def cors_preflight(
    api: FastAPI,
    *,
    origin: str,
    method: str,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=api)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
    ) as client:
        return await client.options(
            "/health",
            headers={
                "origin": origin,
                "access-control-request-method": method,
                "access-control-request-headers": CORS_PREFLIGHT_REQUEST_HEADERS,
            },
        )


def header_values(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",")}


class AppFactoryTests(unittest.TestCase):
    def test_large_json_responses_are_compressed_without_compressing_small_ones(
        self,
    ) -> None:
        from workbench.backend.api import WorkbenchApiSettings
        from workbench.backend.api.mutation_policy import (
            build_http_operation_catalog,
        )
        from workbench.backend.middleware import configure_middleware

        async def call_api() -> tuple[httpx.Response, httpx.Response]:
            api = FastAPI()
            configure_middleware(
                api,
                WorkbenchApiSettings(),
                build_http_operation_catalog(api.routes),
            )

            @api.get("/small")
            async def small() -> dict[str, str]:
                return {"data": "small"}

            @api.get("/large")
            async def large() -> dict[str, str]:
                return {"data": "scalar-data-" * 20_000}

            transport = httpx.ASGITransport(app=api)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://localhost",
                headers={"accept-encoding": "gzip"},
            ) as client:
                return await client.get("/small"), await client.get("/large")

        small_response, large_response = asyncio.run(call_api())

        self.assertNotIn("content-encoding", small_response.headers)
        self.assertEqual(large_response.headers["content-encoding"], "gzip")
        self.assertLess(
            int(large_response.headers["content-length"]),
            len(large_response.content),
        )

    def test_blocking_route_handlers_are_async_boundary_handlers(self) -> None:
        from workbench.backend.api import create_app

        test_app = create_app()
        async_route_pairs = {
            (method, route.path)
            for route in test_app.routes
            if isinstance(route, APIRoute)
            and asyncio.iscoroutinefunction(route.endpoint)
            for method in route.methods or ()
        }
        missing_async_handlers = sorted(
            EXPECTED_ASYNC_BOUNDARY_ROUTE_PAIRS - async_route_pairs
        )
        self.assertEqual(missing_async_handlers, [])

    def test_health_responds_while_log_scalars_read_is_blocked(self) -> None:
        from workbench.backend.blocking import run_blocking_io

        def slow_scalars(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
            time.sleep(0.2)
            return []

        async def call_api() -> tuple[httpx.Response, httpx.Response, bool]:
            api = FastAPI()

            @api.get("/health")
            async def health() -> dict[str, str]:
                return {"status": "ok"}

            @api.post("/logs/scalars")
            async def scalars() -> dict[str, list[object]]:
                await run_blocking_io(slow_scalars)
                return {"series": []}

            transport = httpx.ASGITransport(app=api)
            async with (
                httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                ) as scalar_client,
                httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                ) as health_client,
            ):
                scalar_task = asyncio.create_task(
                    scalar_client.post(
                        "/logs/scalars",
                        json={"runIds": ["run-1"], "tags": ["train/loss"]},
                    )
                )
                await asyncio.sleep(0.02)
                scalar_pending_before_health = not scalar_task.done()
                health_started_at = time.perf_counter()
                health_response = await asyncio.wait_for(
                    health_client.get("/health"),
                    1,
                )
                health_elapsed = time.perf_counter() - health_started_at
                scalar_response = await scalar_task
                return (
                    health_response,
                    scalar_response,
                    scalar_pending_before_health,
                    health_elapsed,
                )

        (
            health_response,
            scalar_response,
            scalar_pending_before_health,
            health_elapsed,
        ) = asyncio.run(call_api())

        self.assertTrue(scalar_pending_before_health)
        self.assertLess(health_elapsed, 0.15)
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(scalar_response.status_code, 200)

    def test_health_responds_while_log_delete_is_blocked(self) -> None:
        from workbench.backend.api import WorkbenchApiSettings, create_app
        from workbench.backend.dependencies import (
            get_run_history_service,
        )

        class FakeRunHistoryService:
            def delete_experiment(
                self,
                experiment: str,
            ) -> dict[str, object]:
                time.sleep(0.2)
                return {
                    "experiment": experiment,
                    "deletedRunIds": [],
                    "deletedRunCount": 0,
                    "deletedRelativePath": experiment,
                }

        async def call_api() -> tuple[httpx.Response, httpx.Response, float]:
            with tempfile.TemporaryDirectory() as tmp:
                test_app = create_app(
                    WorkbenchApiSettings(
                        logs_root=str(Path(tmp) / "logs"),
                        allow_unsafe_local_mutations=True,
                    )
                )

                async def override_run_history_service() -> FakeRunHistoryService:
                    return FakeRunHistoryService()

                test_app.dependency_overrides[get_run_history_service] = (
                    override_run_history_service
                )

                transport = httpx.ASGITransport(app=test_app)
                async with (
                    httpx.AsyncClient(
                        transport=transport,
                        base_url="http://localhost",
                    ) as delete_client,
                    httpx.AsyncClient(
                        transport=transport,
                        base_url="http://localhost",
                    ) as health_client,
                ):
                    started_at = time.perf_counter()
                    delete_task = asyncio.create_task(
                        delete_client.delete(
                            "/logs/experiments/slow",
                            headers={
                                "X-Workbench-Mutation": "true",
                                "Idempotency-Key": uuid.uuid4().hex,
                            },
                        )
                    )
                    await asyncio.sleep(0.02)
                    health_response = await asyncio.wait_for(
                        health_client.get("/health"),
                        1,
                    )
                    health_elapsed = time.perf_counter() - started_at
                    delete_response = await delete_task
                    return health_response, delete_response, health_elapsed

        health_response, delete_response, health_elapsed = asyncio.run(call_api())

        self.assertLess(health_elapsed, 0.15)
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(delete_response.status_code, 200)

    def test_blocking_io_capacity_limiter_bounds_parallel_work(self) -> None:
        import threading

        from workbench.backend.blocking import run_blocking_io

        active = 0
        max_active = 0
        lock = threading.Lock()

        def slow_work() -> str:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return "ok"

        async def run_many() -> list[str]:
            limiter = asyncio.Semaphore(1)
            return await asyncio.gather(
                run_blocking_io(slow_work, limiter=limiter),
                run_blocking_io(slow_work, limiter=limiter),
                run_blocking_io(slow_work, limiter=limiter),
            )

        self.assertEqual(asyncio.run(run_many()), ["ok", "ok", "ok"])
        self.assertEqual(max_active, 1)

    def test_named_blocking_io_limiters_isolate_inspection_work(self) -> None:
        import threading

        from workbench.backend.blocking import (
            named_blocking_work_limiter,
            run_blocking_io,
        )

        started = threading.Event()
        release = threading.Event()

        def slow_log_work() -> str:
            started.set()
            release.wait(timeout=1)
            return "logs"

        async def run_sequence() -> tuple[str, str, float]:
            log_limiter = named_blocking_work_limiter("test-logs", 1)
            inspection_limiter = named_blocking_work_limiter("inspection", 1)
            log_task = asyncio.create_task(
                run_blocking_io(
                    slow_log_work,
                    limiter=log_limiter,
                    timeout_seconds=1,
                )
            )
            while not started.is_set():
                await asyncio.sleep(0.001)
            started_at = time.perf_counter()
            inspection_result = await run_blocking_io(
                lambda: "inspect",
                limiter=inspection_limiter,
                timeout_seconds=1,
            )
            inspection_elapsed = time.perf_counter() - started_at
            release.set()
            log_result = await log_task
            return log_result, inspection_result, inspection_elapsed

        log_result, inspection_result, inspection_elapsed = asyncio.run(run_sequence())

        self.assertEqual(log_result, "logs")
        self.assertEqual(inspection_result, "inspect")
        self.assertLess(inspection_elapsed, 0.15)

    def test_blocking_io_timeout_holds_capacity_until_worker_finishes(
        self,
    ) -> None:
        import threading

        from workbench.backend.blocking import run_blocking_io

        started = threading.Event()
        release = threading.Event()

        def slow_work() -> str:
            started.set()
            release.wait(timeout=1)
            return "slow"

        async def run_sequence() -> str:
            limiter = asyncio.Semaphore(1)
            with self.assertRaises(ApiError):
                await run_blocking_io(
                    slow_work,
                    limiter=limiter,
                    timeout_seconds=0.01,
                )
            self.assertTrue(started.wait(timeout=1))
            with self.assertRaises(ApiError):
                await run_blocking_io(
                    lambda: "blocked",
                    limiter=limiter,
                    timeout_seconds=0.02,
                )
            release.set()
            await asyncio.sleep(0.05)
            return await run_blocking_io(
                lambda: "available",
                limiter=limiter,
                timeout_seconds=1,
            )

        self.assertEqual(asyncio.run(run_sequence()), "available")

    def test_main_app_factory_imports_without_public_api_import_order(self) -> None:
        env = {
            **os.environ,
            "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from workbench.backend.main import create_app; "
                    "print(create_app().title)"
                ),
            ],
            check=False,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Emperor Model Workbench API", result.stdout)

    def test_public_api_app_reexports_main_asgi_target(self) -> None:
        api = importlib.import_module("workbench.backend.api")
        main = importlib.import_module("workbench.backend.main")
        from workbench.backend.api import app as public_app

        self.assertIs(public_app, main.app)
        self.assertIs(api.app, main.app)

    def test_create_app_uses_controlled_settings_and_root_business_routes(self) -> None:
        from workbench.backend.api import WorkbenchApiSettings, create_app
        from workbench.backend.api.v1.router import (
            INTERNAL_API_VERSION_NAMESPACE,
            PUBLIC_API_PREFIX,
        )
        from workbench.backend.dependencies import WorkbenchServices

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            settings = WorkbenchApiSettings(logs_root=str(logs_root))
            test_app = create_app(settings)

            self.assertIsInstance(test_app.state.workbench_services, WorkbenchServices)
            self.assertIs(test_app.state.workbench_services.settings, settings)
            routes = business_route_pairs(test_app)

        missing_routes = sorted(EXPECTED_ROOT_ROUTE_PAIRS - routes)
        unexpected_v1_routes = sorted(
            (method, f"/v1{path}")
            for method, path in EXPECTED_ROOT_ROUTE_PAIRS
            if (method, f"/v1{path}") in routes
        )

        self.assertEqual(missing_routes, [])
        self.assertEqual(unexpected_v1_routes, [])
        self.assertEqual(INTERNAL_API_VERSION_NAMESPACE, "v1")
        self.assertEqual(PUBLIC_API_PREFIX, "")

    def test_route_modules_do_not_read_app_state_directly(self) -> None:
        route_root = Path("workbench/backend/api/v1/routers")

        for path in sorted(route_root.glob("*.py")):
            if path.name == "__init__.py":
                continue
            with self.subTest(path=str(path)):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(".app.state", source)

    def test_create_app_registers_api_error_handler(self) -> None:
        from workbench.backend.api import WorkbenchApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            test_app = create_app(
                WorkbenchApiSettings(logs_root=str(Path(tmp) / "logs"))
            )

        self.assertIs(
            test_app.exception_handlers.get(ApiError),
            api_error_handler,
        )

    def test_inspector_error_response_shape_is_preserved(self) -> None:
        from workbench.backend.api import WorkbenchApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            test_app = create_app(
                WorkbenchApiSettings(logs_root=str(Path(tmp) / "logs"))
            )

            @test_app.get("/raises-inspector-error")
            async def raises_inspector_error() -> None:
                raise InspectorError("bad model input")

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=test_app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                ) as client:
                    return await client.get("/raises-inspector-error")

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "bad model input"})

    def test_api_default_cors_settings_allow_local_dev_frontends(self) -> None:
        from workbench.backend.api import create_app

        test_app = create_app()

        for origin in (
            "http://localhost:9000",
            "http://127.0.0.1:9000",
            "http://0.0.0.0:9000",
            "http://localhost:9001",
            "http://127.0.0.1:9001",
            "http://0.0.0.0:9001",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://0.0.0.0:3000",
        ):
            for method in CORS_PREFLIGHT_METHODS:
                with self.subTest(origin=origin, method=method):
                    response = asyncio.run(
                        cors_preflight(test_app, origin=origin, method=method)
                    )
                    self.assert_allowed_authorization_preflight(
                        response,
                        origin=origin,
                        method=method,
                    )

    def test_api_factory_applies_hosted_cors_settings_to_preflights(self) -> None:
        from workbench.backend.api import WorkbenchApiSettings, create_app

        origin = "https://workbench.example.com"
        test_app = create_app(WorkbenchApiSettings(cors_origins=[origin]))

        for method in CORS_PREFLIGHT_METHODS:
            with self.subTest(method=method):
                response = asyncio.run(
                    cors_preflight(test_app, origin=origin, method=method)
                )
                self.assert_allowed_authorization_preflight(
                    response,
                    origin=origin,
                    method=method,
                )

    def test_api_factory_disallowed_cors_origin_gets_no_allow_origin(self) -> None:
        from workbench.backend.api import WorkbenchApiSettings, create_app

        test_app = create_app(
            WorkbenchApiSettings(cors_origins=["https://workbench.example.com"])
        )

        for method in CORS_PREFLIGHT_METHODS:
            with self.subTest(method=method):
                response = asyncio.run(
                    cors_preflight(
                        test_app,
                        origin="https://evil.example.com",
                        method=method,
                    )
                )
                self.assertEqual(response.status_code, 400)
                self.assertNotIn("access-control-allow-origin", response.headers)

    def assert_allowed_authorization_preflight(
        self,
        response: httpx.Response,
        *,
        origin: str,
        method: str,
    ) -> None:
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            origin,
        )
        self.assertIn(
            method,
            response.headers["access-control-allow-methods"],
        )
        self.assertGreaterEqual(
            header_values(response.headers["access-control-allow-headers"]),
            CORS_PREFLIGHT_ALLOWED_HEADERS,
        )


if __name__ == "__main__":
    unittest.main()
