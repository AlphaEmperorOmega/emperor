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

from viewer.backend.exceptions import inspector_error_handler
from viewer.backend.inspector.errors import InspectorError


EXPECTED_ROOT_ROUTE_PAIRS = {
    ("GET", "/health"),
    ("GET", "/models"),
    ("POST", "/inspect"),
    ("GET", "/logs/runs"),
    ("POST", "/logs/tags"),
    ("POST", "/training/jobs"),
}


def business_route_pairs(api: FastAPI) -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in api.routes
        if isinstance(route, APIRoute)
        for method in route.methods or ()
    }


class AppFactoryTests(unittest.TestCase):
    def test_public_api_app_reexports_main_asgi_target(self) -> None:
        api = importlib.import_module("viewer.backend.api")
        main = importlib.import_module("viewer.backend.main")
        from viewer.backend.api import app as public_app

        self.assertIs(public_app, main.app)
        self.assertIs(api.app, main.app)

    def test_create_app_uses_controlled_settings_and_root_business_routes(self) -> None:
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            settings = ViewerApiSettings(logs_root=str(logs_root))
            test_app = create_app(settings)

            self.assertIs(test_app.state.settings, settings)
            routes = business_route_pairs(test_app)

        missing_routes = sorted(EXPECTED_ROOT_ROUTE_PAIRS - routes)
        unexpected_v1_routes = sorted(
            (method, f"/v1{path}")
            for method, path in EXPECTED_ROOT_ROUTE_PAIRS
            if (method, f"/v1{path}") in routes
        )

        self.assertEqual(missing_routes, [])
        self.assertEqual(unexpected_v1_routes, [])

    def test_create_app_registers_inspector_error_handler(self) -> None:
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            test_app = create_app(
                ViewerApiSettings(logs_root=str(Path(tmp) / "logs"))
            )

        self.assertIs(
            test_app.exception_handlers.get(InspectorError),
            inspector_error_handler,
        )

    def test_api_default_cors_settings_allow_local_dev_frontends(self) -> None:
        from viewer.backend.api import create_app

        async def call_api(origin: str) -> httpx.Response:
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.options(
                    "/health",
                    headers={
                        "origin": origin,
                        "access-control-request-method": "GET",
                    },
                )

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
            with self.subTest(origin=origin):
                response = asyncio.run(call_api(origin))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.headers["access-control-allow-origin"],
                    origin,
                )
                self.assertIn(
                    "DELETE",
                    response.headers["access-control-allow-methods"],
                )

    def test_api_factory_applies_custom_cors_settings(self) -> None:
        from viewer.backend.api import ViewerApiSettings, create_app

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(
                app=create_app(ViewerApiSettings(cors_origins=["http://frontend.test"]))
            )
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.options(
                    "/health",
                    headers={
                        "origin": "http://frontend.test",
                        "access-control-request-method": "GET",
                    },
                )

        response = asyncio.run(call_api())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://frontend.test",
        )


if __name__ == "__main__":
    unittest.main()
