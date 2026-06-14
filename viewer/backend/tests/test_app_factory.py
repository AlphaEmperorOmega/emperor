from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import httpx
from fastapi import FastAPI
from fastapi.routing import APIRoute

from viewer.backend.core.errors import ApiError
from viewer.backend.exceptions import api_error_handler
from viewer.backend.inspector.errors import InspectorError

EXPECTED_ROOT_ROUTE_PAIRS = {
    ("GET", "/health"),
    ("GET", "/models"),
    ("POST", "/inspect"),
    ("POST", "/inspect/operation-graph"),
    ("GET", "/logs/runs"),
    ("POST", "/logs/tags"),
    ("POST", "/training/jobs"),
}

CORS_PREFLIGHT_METHODS = ("GET", "POST", "PATCH", "DELETE")
CORS_PREFLIGHT_REQUEST_HEADERS = "authorization,content-type"
CORS_PREFLIGHT_ALLOWED_HEADERS = {"authorization", "content-type"}

# Routes that do blocking CPU or file work stay async at the API boundary.
# Relying on FastAPI sync-route dispatch deadlocks under ASGITransport in this
# test environment.
EXPECTED_ASYNC_BOUNDARY_ROUTE_PAIRS = {
    ("POST", "/inspect"),
    ("POST", "/inspect/operation-graph"),
    ("GET", "/logs/runs"),
    ("GET", "/logs/experiments"),
    ("POST", "/logs/checkpoints"),
    ("POST", "/logs/tags"),
    ("POST", "/logs/scalars"),
    ("POST", "/logs/media"),
    ("POST", "/logs/parameter-status"),
    ("GET", "/logs/runs/{run_id}/artifacts"),
    ("GET", "/logs/runs/{run_id}/monitor-data"),
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
        base_url="http://testserver",
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
    def test_blocking_route_handlers_are_async_boundary_handlers(self) -> None:
        from viewer.backend.api import create_app

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

    def test_public_api_app_reexports_main_asgi_target(self) -> None:
        api = importlib.import_module("viewer.backend.api")
        main = importlib.import_module("viewer.backend.main")
        from viewer.backend.api import app as public_app

        self.assertIs(public_app, main.app)
        self.assertIs(api.app, main.app)

    def test_create_app_uses_controlled_settings_and_root_business_routes(self) -> None:
        from viewer.backend.api import ViewerApiSettings, create_app
        from viewer.backend.api.v1.router import (
            INTERNAL_API_VERSION_NAMESPACE,
            PUBLIC_API_PREFIX,
        )
        from viewer.backend.dependencies import ViewerServices

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            settings = ViewerApiSettings(logs_root=str(logs_root))
            test_app = create_app(settings)

            self.assertIsInstance(test_app.state.viewer_services, ViewerServices)
            self.assertIs(test_app.state.viewer_services.settings, settings)
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
        route_root = Path("viewer/backend/api/v1/routers")

        for path in sorted(route_root.glob("*.py")):
            if path.name == "__init__.py":
                continue
            with self.subTest(path=str(path)):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(".app.state", source)

    def test_create_app_registers_api_error_handler(self) -> None:
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            test_app = create_app(
                ViewerApiSettings(logs_root=str(Path(tmp) / "logs"))
            )

        self.assertIs(
            test_app.exception_handlers.get(ApiError),
            api_error_handler,
        )

    def test_inspector_error_response_shape_is_preserved(self) -> None:
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            test_app = create_app(
                ViewerApiSettings(logs_root=str(Path(tmp) / "logs"))
            )

            @test_app.get("/raises-inspector-error")
            async def raises_inspector_error() -> None:
                raise InspectorError("bad model input")

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=test_app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.get("/raises-inspector-error")

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "bad model input"})

    def test_api_default_cors_settings_allow_local_dev_frontends(self) -> None:
        from viewer.backend.api import create_app

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
        from viewer.backend.api import ViewerApiSettings, create_app

        origin = "https://viewer.example.com"
        test_app = create_app(ViewerApiSettings(cors_origins=[origin]))

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
        from viewer.backend.api import ViewerApiSettings, create_app

        test_app = create_app(
            ViewerApiSettings(cors_origins=["https://viewer.example.com"])
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
