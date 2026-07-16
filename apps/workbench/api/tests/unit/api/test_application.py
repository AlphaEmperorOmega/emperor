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
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import httpx
from fastapi import FastAPI
from fastapi.datastructures import DefaultPlaceholder
from fastapi.routing import APIRoute, iter_route_contexts

from emperor_workbench.api._errors import ApiError, api_error_handler
from emperor_workbench.inspection import InspectionFailure
from tests.support import API_ROOT, lifespan_client

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
        for route in iter_route_contexts(api.routes)
        if isinstance(route.original_route, APIRoute) and route.path is not None
        for method in route.methods or ()
    }


async def cors_preflight(
    api: FastAPI,
    *,
    origin: str,
    method: str,
) -> httpx.Response:
    async with lifespan_client(
        api,
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
    def test_application_exposes_only_the_semantic_inspection_service(self) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.inspection import InspectionService
        from emperor_workbench.settings import WorkbenchApiSettings

        async def inspection_service(app) -> InspectionService:
            async with app.router.lifespan_context(app):
                return app.state.workbench_container.inspection

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_app(
                WorkbenchApiSettings(
                    logs_root=str(root / "logs"),
                    snapshots_root=str(root / "snapshots"),
                    state_root=str(root / "state"),
                )
            )
            service = asyncio.run(inspection_service(app))

        self.assertIsInstance(service, InspectionService)
        self.assertFalse(hasattr(service, "executor"))

    def test_response_models_use_fastapi_native_pydantic_json_serializer(self) -> None:
        from emperor_workbench.api import create_app

        api = create_app()
        business_routes = [
            route
            for route in iter_route_contexts(api.routes)
            if isinstance(route.original_route, APIRoute)
        ]

        self.assertTrue(business_routes)
        self.assertTrue(
            all(route.response_model is not None for route in business_routes)
        )
        self.assertTrue(
            all(
                isinstance(route.response_class, DefaultPlaceholder)
                for route in business_routes
            )
        )

    def test_large_json_responses_are_compressed_without_compressing_small_ones(
        self,
    ) -> None:
        from emperor_workbench.api._container import WorkbenchContainerSlot
        from emperor_workbench.api._middleware import configure_middleware
        from emperor_workbench.api._mutations import (
            build_http_operation_catalog,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        async def call_api() -> tuple[httpx.Response, httpx.Response]:
            api = FastAPI()
            configure_middleware(
                api,
                WorkbenchApiSettings(),
                build_http_operation_catalog(api.routes),
                container_slot=WorkbenchContainerSlot(),
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
        from emperor_workbench.api import create_app

        test_app = create_app()
        async_route_pairs = {
            (method, route.path)
            for route in iter_route_contexts(test_app.routes)
            if isinstance(route.original_route, APIRoute)
            and route.path is not None
            and asyncio.iscoroutinefunction(route.endpoint)
            for method in route.methods or ()
        }
        missing_async_handlers = sorted(
            EXPECTED_ASYNC_BOUNDARY_ROUTE_PAIRS - async_route_pairs
        )
        self.assertEqual(missing_async_handlers, [])

    def test_health_responds_while_log_scalars_read_is_blocked(self) -> None:
        from emperor_workbench.api._blocking import run_blocking_io

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
        from emperor_workbench.api import create_app
        from emperor_workbench.api._dependencies import (
            get_run_history_service,
        )
        from emperor_workbench.run_history import LogExperimentDeleteResult
        from emperor_workbench.settings import WorkbenchApiSettings

        class FakeRunHistoryService:
            def delete_experiment(
                self,
                experiment: str,
            ) -> LogExperimentDeleteResult:
                time.sleep(0.2)
                return LogExperimentDeleteResult(
                    experiment=experiment,
                    deleted_run_ids=(),
                    deleted_run_count=0,
                    deleted_relative_path=experiment,
                )

        async def call_api() -> tuple[httpx.Response, httpx.Response, float]:
            with tempfile.TemporaryDirectory() as tmp:
                test_app = create_app(
                    WorkbenchApiSettings(
                        logs_root=str(Path(tmp) / "logs"),
                        snapshots_root=str(Path(tmp) / "snapshots"),
                        state_root=str(Path(tmp) / "state"),
                        allow_unsafe_local_mutations=True,
                        training_cancellation_mode="process-group",
                    )
                )

                async def override_run_history_service() -> FakeRunHistoryService:
                    return FakeRunHistoryService()

                test_app.dependency_overrides[get_run_history_service] = (
                    override_run_history_service
                )

                async with lifespan_client(test_app) as client:
                    started_at = time.perf_counter()
                    delete_task = asyncio.create_task(
                        client.delete(
                            "/logs/experiments/slow",
                            headers={
                                "X-Workbench-Mutation": "true",
                                "Idempotency-Key": uuid.uuid4().hex,
                            },
                        )
                    )
                    await asyncio.sleep(0.02)
                    health_response = await asyncio.wait_for(
                        client.get("/health"),
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

        from emperor_workbench.api._blocking import run_blocking_io

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

        from emperor_workbench.api._blocking import (
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

        from emperor_workbench.api._blocking import run_blocking_io

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
                    "from emperor_workbench.api import create_app; "
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

    def test_public_api_is_the_canonical_asgi_target(self) -> None:
        api = importlib.import_module("emperor_workbench.api")
        from emperor_workbench.api import app as public_app

        self.assertIs(public_app, api.app)

    def test_create_app_uses_controlled_settings_and_root_business_routes(self) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.api._container import WorkbenchContainer
        from emperor_workbench.api.v1 import (
            INTERNAL_API_VERSION_NAMESPACE,
            PUBLIC_API_PREFIX,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = WorkbenchApiSettings(
                logs_root=str(root / "logs"),
                snapshots_root=str(root / "snapshots"),
                state_root=str(root / "state"),
                training_cancellation_mode="process-group",
            )
            test_app = create_app(settings)
            self.assertFalse(hasattr(test_app.state, "workbench_container"))

            async def inspect_lifespan() -> None:
                async with test_app.router.lifespan_context(test_app):
                    container = test_app.state.workbench_container
                    self.assertIsInstance(container, WorkbenchContainer)
                    self.assertIs(container.settings, settings)

            asyncio.run(inspect_lifespan())
            self.assertFalse(hasattr(test_app.state, "workbench_container"))
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

    def test_bootstrap_failure_closes_all_acquired_resources_and_reraises(
        self,
    ) -> None:
        from emperor_workbench.api._bootstrap import acquire_container
        from emperor_workbench.settings import WorkbenchApiSettings

        class BootstrapFailure(RuntimeError):
            pass

        failure = BootstrapFailure("inspection construction failed")
        project_adapter = Mock()
        blocking_work = Mock()
        mutation_execution = Mock()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = WorkbenchApiSettings(
                logs_root=str(root / "logs"),
                snapshots_root=str(root / "snapshots"),
                state_root=str(root / "state"),
                training_cancellation_mode="process-group",
            )
            with (
                patch(
                    "emperor_workbench.api._bootstrap.BlockingWorkRuntime",
                    return_value=blocking_work,
                ),
                patch(
                    "emperor_workbench.api._bootstrap.MutationExecutionRuntime",
                    return_value=mutation_execution,
                ),
                patch(
                    "emperor_workbench.api._bootstrap.InspectionService",
                    side_effect=failure,
                ),
                self.assertRaises(BootstrapFailure) as raised,
            ):
                acquire_container(
                    settings,
                    project_adapter=project_adapter,
                )

        self.assertIs(raised.exception, failure)
        mutation_execution.close_executor.assert_called_once_with()
        blocking_work.close.assert_called_once_with()
        project_adapter.close.assert_called_once_with()

    def test_lifespan_cleanup_failure_still_unpublishes_and_closes_later_resources(
        self,
    ) -> None:
        from emperor_workbench.api._container import WorkbenchContainerSlot
        from emperor_workbench.api._lifespan import create_lifespan
        from emperor_workbench.settings import WorkbenchApiSettings

        class CleanupFailure(RuntimeError):
            pass

        failure = CleanupFailure("mutation cleanup failed")
        container = Mock()
        container.mutation_execution.close = AsyncMock(side_effect=failure)
        container.blocking_work.close = Mock()
        container.project_adapter.close = Mock()
        container_slot = WorkbenchContainerSlot()
        settings = WorkbenchApiSettings()
        test_app = FastAPI(
            lifespan=create_lifespan(
                settings,
                container_slot=container_slot,
                project_adapter=None,
            )
        )

        async def exercise_lifespan() -> None:
            with patch(
                "emperor_workbench.api._lifespan.acquire_container",
                return_value=container,
            ):
                async with test_app.router.lifespan_context(test_app):
                    self.assertIs(container_slot.get(), container)
                    self.assertIs(
                        test_app.state.workbench_container,
                        container,
                    )

        with self.assertRaises(CleanupFailure) as raised:
            asyncio.run(exercise_lifespan())

        self.assertIs(raised.exception, failure)
        container.mutation_execution.close.assert_awaited_once_with()
        container.blocking_work.close.assert_called_once_with()
        container.project_adapter.close.assert_called_once_with()
        self.assertFalse(hasattr(test_app.state, "workbench_container"))
        with self.assertRaises(RuntimeError):
            container_slot.get()

    def test_route_modules_do_not_read_app_state_directly(self) -> None:
        route_root = API_ROOT / "src" / "emperor_workbench" / "api" / "v1"

        for path in sorted(route_root.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            with self.subTest(path=str(path)):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(".app.state", source)

    def test_create_app_registers_api_error_handler(self) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

        with tempfile.TemporaryDirectory() as tmp:
            test_app = create_app(
                WorkbenchApiSettings(logs_root=str(Path(tmp) / "logs"))
            )

        self.assertIs(
            test_app.exception_handlers.get(ApiError),
            api_error_handler,
        )

    def test_inspection_failure_response_shape_is_preserved(self) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

        with tempfile.TemporaryDirectory() as tmp:
            test_app = create_app(
                WorkbenchApiSettings(logs_root=str(Path(tmp) / "logs"))
            )

            @test_app.get("/raises-inspection-failure")
            async def raises_inspection_failure() -> None:
                raise InspectionFailure("bad model input")

            async def call_api() -> httpx.Response:
                async with lifespan_client(test_app) as client:
                    return await client.get("/raises-inspection-failure")

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "bad model input"})

    def test_api_default_cors_settings_allow_local_dev_frontends(self) -> None:
        from emperor_workbench.api import create_app

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
        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

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
        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

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
