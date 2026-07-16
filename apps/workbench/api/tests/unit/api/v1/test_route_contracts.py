from __future__ import annotations

import asyncio
import unittest

from fastapi.routing import APIRoute, iter_route_contexts

from emperor_workbench.api import app
from tests.support import lifespan_client as _lifespan_client

from ._contract_support import (
    ENDPOINT_SCHEMA_MAPPINGS,
    EXPECTED_BUSINESS_ROUTES,
    SCHEMA_PARITY_BY_BACKEND_AND_FRONTEND_CONTRACT,
    SCHEMA_PARITY_BY_BACKEND_SCHEMA,
    _body_request_schemas,
    _business_routes_by_key,
)


class ApiRouteContractTests(unittest.TestCase):
    def test_api_routes_declare_response_models(self) -> None:
        missing = [
            f"{sorted(route.methods)} {route.path}"
            for route in iter_route_contexts(app.routes)
            if isinstance(route.original_route, APIRoute)
            and route.response_model is None
        ]

        self.assertEqual(missing, [])

    def test_api_route_inventory_preserves_current_contract(self) -> None:
        business_prefixes = (
            "/capabilities",
            "/config-snapshots",
            "/health",
            "/models",
            "/inspect",
            "/logs",
            "/training",
        )
        routes = sorted(
            (tuple(sorted(route.methods or ())), route.path)
            for route in iter_route_contexts(app.routes)
            if isinstance(route.original_route, APIRoute)
            and route.path is not None
            and route.path.startswith(business_prefixes)
        )

        self.assertEqual(routes, EXPECTED_BUSINESS_ROUTES)
        self.assertFalse(
            any(path.startswith("/v1/") or path == "/v1" for _methods, path in routes)
        )

    def test_endpoint_schema_mapping_covers_public_routes(self) -> None:
        self.assertEqual(
            sorted(ENDPOINT_SCHEMA_MAPPINGS),
            EXPECTED_BUSINESS_ROUTES,
        )

        routes_by_key = _business_routes_by_key()
        self.assertEqual(sorted(routes_by_key), EXPECTED_BUSINESS_ROUTES)
        self.assertEqual(set(routes_by_key), set(ENDPOINT_SCHEMA_MAPPINGS))

        for route_key, mapping in ENDPOINT_SCHEMA_MAPPINGS.items():
            with self.subTest(route=route_key):
                route = routes_by_key[route_key]
                self.assertIs(route.response_model, mapping.backend_response_schema)
                self.assertEqual(
                    _body_request_schemas(route),
                    mapping.backend_body_request_schemas,
                )
                self.assertIsInstance(mapping.frontend_api_function, str)
                self.assertTrue(mapping.frontend_api_function)
                self.assertIsInstance(mapping.frontend_response_schema, str)
                self.assertTrue(mapping.frontend_response_schema)

    def test_endpoint_schema_mapping_has_explicit_schema_parity_cases(self) -> None:
        for route_key, mapping in ENDPOINT_SCHEMA_MAPPINGS.items():
            with self.subTest(route=route_key, schema="response"):
                self.assertIn(
                    (
                        mapping.backend_response_schema,
                        mapping.frontend_response_schema,
                    ),
                    SCHEMA_PARITY_BY_BACKEND_AND_FRONTEND_CONTRACT,
                )

            for request_schema in mapping.backend_body_request_schemas:
                with self.subTest(
                    route=route_key,
                    schema=request_schema.__name__,
                ):
                    self.assertIn(request_schema, SCHEMA_PARITY_BY_BACKEND_SCHEMA)


class ApiPrivateIntegrationContractTests(unittest.TestCase):
    def test_api_dependency_overrides_can_replace_route_services(self) -> None:
        import httpx

        from emperor_workbench.api import create_app
        from emperor_workbench.api._dependencies import get_run_history_service
        from emperor_workbench.run_history import LogRunFacets, LogRunPage

        class FakeRunHistoryService:
            def list_runs(self, **kwargs: object) -> LogRunPage:
                return LogRunPage(
                    total=0,
                    limit=int(kwargs["limit"]),
                    offset=int(kwargs["offset"]),
                    has_more=False,
                    runs=(),
                    facets=LogRunFacets(experiments=()),
                )

        async def override_run_history_service() -> FakeRunHistoryService:
            return FakeRunHistoryService()

        test_app = create_app()
        test_app.dependency_overrides[get_run_history_service] = (
            override_run_history_service
        )

        async def call_api() -> httpx.Response:
            async with _lifespan_client(test_app) as client:
                return await client.get("/logs/runs")

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "total": 0,
                "limit": 500,
                "offset": 0,
                "hasMore": False,
                "facets": {"experiments": []},
                "runs": [],
            },
        )
