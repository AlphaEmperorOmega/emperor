from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from starlette.routing import Match

from tests.support import lifespan_client


class HttpMutationPolicyTests(unittest.TestCase):
    def test_policy_preserves_string_and_json_compatibility(self) -> None:
        from emperor_workbench.api._mutations import HttpOperationPolicy

        policy = HttpOperationPolicy.READ_ONLY

        self.assertEqual(policy, "read-only")
        self.assertEqual(str(policy), "HttpOperationPolicy.READ_ONLY")
        self.assertEqual(f"{policy:>30}", " HttpOperationPolicy.READ_ONLY")
        self.assertEqual(
            json.loads(json.dumps({"policy": policy})),
            {"policy": "read-only"},
        )

    def test_configured_app_classifies_every_non_safe_operation(self) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            build_http_operation_catalog,
        )
        from emperor_workbench.api.v1 import router as api_v1_router

        app = create_app()
        catalog = build_http_operation_catalog(
            app.routes,
            declared_routes=api_v1_router.routes,
        )
        registered_non_safe_operations = {
            (method, route.path)
            for route in iter_route_contexts(app.routes)
            if isinstance(route.original_route, APIRoute) and route.path is not None
            for method in route.methods or ()
            if method not in {"GET", "HEAD", "OPTIONS", "TRACE"}
        }

        self.assertEqual(
            {(operation.method, operation.route.path) for operation in catalog},
            registered_non_safe_operations,
        )
        self.assertEqual(
            {operation.policy for operation in catalog},
            {
                HttpOperationPolicy.READ_ONLY,
                HttpOperationPolicy.LOCAL_MUTATION,
                HttpOperationPolicy.LOG_IMPORT,
            },
        )
        self.assertEqual(len(catalog), 19)

    def test_catalog_discovers_effective_nested_prefixed_route_context(self) -> None:
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            build_http_operation_catalog,
            declare_http_operation,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        async def outer_dependency() -> None:
            pass

        async def inner_dependency() -> None:
            pass

        async def get_settings() -> WorkbenchApiSettings:
            return WorkbenchApiSettings(allow_unsafe_local_mutations=True)

        settings_dependency = Depends(get_settings)
        child = APIRouter()

        @child.post("/operation")
        @declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
        async def operation(settings=settings_dependency) -> dict[str, bool]:
            return {"called": True}

        parent = APIRouter()
        parent.include_router(
            child,
            prefix="/child",
            dependencies=[Depends(inner_dependency)],
        )
        app = FastAPI()
        app.include_router(
            parent,
            prefix="/api",
            dependencies=[Depends(outer_dependency)],
        )

        catalog = build_http_operation_catalog(
            app.routes,
            declared_routes=parent.routes,
        )

        self.assertEqual(len(catalog), 1)
        operation_context = catalog[0]
        self.assertEqual(operation_context.method, "POST")
        self.assertEqual(operation_context.route.path, "/api/child/operation")
        self.assertIsInstance(operation_context.route.original_route, APIRoute)
        self.assertEqual(
            {
                dependency.dependency
                for dependency in operation_context.route.dependencies
            },
            {outer_dependency, inner_dependency},
        )
        match, _child_scope = operation_context.route.matches(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/child/operation",
                "root_path": "",
            }
        )
        self.assertIs(match, Match.FULL)

    def test_mounted_mutation_requires_proof_and_cannot_write(self) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.api._security import (
            MUTATION_PROOF_REQUIRED_DETAIL,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        async def request(app: FastAPI) -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://localhost",
            ) as client:
                return await client.post(
                    "/workbench/config-snapshots",
                    json={
                        "modelType": "linears",
                        "model": "linear",
                        "preset": "baseline",
                        "name": "mounted-prefix-probe",
                        "overrides": {},
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            snapshots_root = Path(tmp) / "snapshots"
            child = create_app(
                WorkbenchApiSettings(
                    logs_root=str(Path(tmp) / "logs"),
                    snapshots_root=str(snapshots_root),
                    allow_unsafe_local_mutations=True,
                )
            )
            mounted = FastAPI()
            mounted.mount("/workbench", child)

            response = asyncio.run(request(mounted))

            self.assertFalse(snapshots_root.exists())

        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(
            response.json(),
            {"detail": MUTATION_PROOF_REQUIRED_DETAIL},
        )

    def test_local_mutation_declaration_enforces_operational_opt_in(self) -> None:
        from emperor_workbench.api._container import WorkbenchContainerSlot
        from emperor_workbench.api._middleware import configure_middleware
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            build_http_operation_catalog,
            declare_http_operation,
        )
        from emperor_workbench.api._security import (
            LOCAL_MUTATION_DISABLED_DETAIL,
            MUTATION_HEADER_NAME,
            MUTATION_HEADER_VALUE,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        settings = WorkbenchApiSettings(allow_unsafe_local_mutations=False)
        app = FastAPI()

        async def get_settings() -> WorkbenchApiSettings:
            return settings

        settings_dependency = Depends(get_settings)

        @app.post("/operation")
        @declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
        async def operation(settings=settings_dependency) -> dict[str, bool]:
            return {"called": True}

        configure_middleware(
            app,
            settings,
            build_http_operation_catalog(app.routes),
            container_slot=WorkbenchContainerSlot(),
        )

        async def request() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://localhost",
            ) as client:
                return await client.post(
                    "/operation",
                    headers={
                        MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                )

        response = asyncio.run(request())

        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(
            response.json(),
            {"detail": LOCAL_MUTATION_DISABLED_DETAIL},
        )

    def test_catalog_rejects_opaque_child_mounts(self) -> None:
        from emperor_workbench.api._mutations import (
            MutationPolicyConfigurationError,
            build_http_operation_catalog,
        )

        app = FastAPI()
        app.mount("/opaque", FastAPI())

        with self.assertRaises(MutationPolicyConfigurationError):
            build_http_operation_catalog(app.routes)

    def test_catalog_rejects_mutation_without_operational_settings(self) -> None:
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            MutationPolicyConfigurationError,
            build_http_operation_catalog,
            declare_http_operation,
        )

        app = FastAPI()

        @app.post("/operation")
        @declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
        async def operation() -> dict[str, bool]:
            return {"called": True}

        with self.assertRaises(MutationPolicyConfigurationError):
            build_http_operation_catalog(app.routes)

    def test_new_non_safe_route_fails_closed_without_declaration(self) -> None:
        from emperor_workbench.api._mutations import (
            MutationPolicyConfigurationError,
            build_http_operation_catalog,
        )

        app = FastAPI()

        @app.post("/undeclared")
        async def undeclared() -> dict[str, bool]:
            return {"called": True}

        with self.assertRaises(MutationPolicyConfigurationError):
            build_http_operation_catalog(app.routes)

    def test_catalog_rejects_unknown_policy(self) -> None:
        from emperor_workbench.api._mutations import (
            MutationPolicyConfigurationError,
            build_http_operation_catalog,
            declare_http_operation,
        )

        app = FastAPI()

        @app.post("/unknown")
        @declare_http_operation("unknown")  # type: ignore[arg-type]
        async def unknown() -> dict[str, bool]:
            return {"called": True}

        with self.assertRaises(MutationPolicyConfigurationError):
            build_http_operation_catalog(app.routes)

    def test_catalog_rejects_duplicate_declarations(self) -> None:
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            MutationPolicyConfigurationError,
            build_http_operation_catalog,
            declare_http_operation,
        )

        app = FastAPI()

        @app.post("/duplicate")
        @declare_http_operation(HttpOperationPolicy.READ_ONLY)
        @declare_http_operation(HttpOperationPolicy.READ_ONLY)
        async def duplicate() -> dict[str, bool]:
            return {"called": True}

        with self.assertRaises(MutationPolicyConfigurationError):
            build_http_operation_catalog(app.routes)

    def test_catalog_rejects_conflicting_declarations(self) -> None:
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            MutationPolicyConfigurationError,
            build_http_operation_catalog,
            declare_http_operation,
        )

        app = FastAPI()

        @app.post("/conflicting")
        @declare_http_operation(HttpOperationPolicy.READ_ONLY)
        @declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
        async def conflicting() -> dict[str, bool]:
            return {"called": True}

        with self.assertRaises(MutationPolicyConfigurationError):
            build_http_operation_catalog(app.routes)

    def test_catalog_rejects_declared_but_unmounted_operation(self) -> None:
        from fastapi import APIRouter

        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            MutationPolicyConfigurationError,
            build_http_operation_catalog,
            declare_http_operation,
        )

        declared_router = APIRouter()

        @declared_router.post("/unmounted")
        @declare_http_operation(HttpOperationPolicy.READ_ONLY)
        async def unmounted() -> dict[str, bool]:
            return {"called": True}

        with self.assertRaises(MutationPolicyConfigurationError):
            build_http_operation_catalog(
                FastAPI().routes,
                declared_routes=declared_router.routes,
            )

    def test_read_only_post_needs_neither_mutation_proof_nor_settings(self) -> None:
        from emperor_workbench.api._container import WorkbenchContainerSlot
        from emperor_workbench.api._middleware import configure_middleware
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            build_http_operation_catalog,
            declare_http_operation,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        settings = WorkbenchApiSettings()
        app = FastAPI()

        @app.post("/query")
        @declare_http_operation(HttpOperationPolicy.READ_ONLY)
        async def query() -> dict[str, bool]:
            return {"readOnly": True}

        configure_middleware(
            app,
            settings,
            build_http_operation_catalog(app.routes),
            container_slot=WorkbenchContainerSlot(),
        )

        async def request() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://localhost",
            ) as client:
                return await client.post(
                    "/query",
                    headers={
                        "Origin": "https://hostile.example",
                        "Sec-Fetch-Site": "cross-site",
                    },
                )

        response = asyncio.run(request())

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"readOnly": True})

    def test_broad_mutation_and_log_import_enablement_are_independent(self) -> None:
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            operation_policy_enabled,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        cases = (
            (False, False, False, False),
            (False, True, False, True),
            (True, False, True, False),
            (True, True, True, True),
        )
        for broad_setting, import_setting, broad_enabled, import_enabled in cases:
            with self.subTest(
                broad_setting=broad_setting,
                import_setting=import_setting,
            ):
                settings = WorkbenchApiSettings(
                    allow_unsafe_local_mutations=broad_setting,
                    allow_log_imports=import_setting,
                )

                self.assertIs(
                    operation_policy_enabled(
                        HttpOperationPolicy.LOCAL_MUTATION,
                        settings,
                    ),
                    broad_enabled,
                )
                self.assertIs(
                    operation_policy_enabled(
                        HttpOperationPolicy.LOG_IMPORT,
                        settings,
                    ),
                    import_enabled,
                )

    def test_capabilities_report_the_policy_enablement_matrix(self) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

        async def capabilities(app: FastAPI) -> dict[str, object]:
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                ) as client:
                    response = await client.get("/capabilities")
                    self.assertEqual(response.status_code, 200, response.text)
                    return response.json()

        for broad_enabled in (False, True):
            for imports_enabled in (False, True):
                with self.subTest(
                    broad_enabled=broad_enabled,
                    imports_enabled=imports_enabled,
                ):
                    payload = asyncio.run(
                        capabilities(
                            create_app(
                                WorkbenchApiSettings(
                                    allow_unsafe_local_mutations=broad_enabled,
                                    allow_log_imports=imports_enabled,
                                    training_cancellation_mode="process-group",
                                )
                            )
                        )
                    )

                    self.assertIs(payload["trainingEnabled"], broad_enabled)
                    self.assertIs(payload["logDeletionEnabled"], broad_enabled)
                    self.assertIs(payload["configSnapshotsEnabled"], broad_enabled)
                    self.assertIs(payload["uploadsEnabled"], imports_enabled)

    def test_catalog_rejects_duplicate_method_and_path_operations(self) -> None:
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            MutationPolicyConfigurationError,
            build_http_operation_catalog,
            declare_http_operation,
        )

        app = FastAPI()

        @app.post("/duplicate-path")
        @declare_http_operation(HttpOperationPolicy.READ_ONLY)
        async def first() -> dict[str, int]:
            return {"handler": 1}

        @app.post("/duplicate-path")
        @declare_http_operation(HttpOperationPolicy.READ_ONLY)
        async def second() -> dict[str, int]:
            return {"handler": 2}

        with self.assertRaises(MutationPolicyConfigurationError):
            build_http_operation_catalog(app.routes)

    def test_encoded_parameterized_mutation_still_requires_proof(self) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.api._security import MUTATION_PROOF_REQUIRED_DETAIL
        from emperor_workbench.settings import WorkbenchApiSettings

        app = create_app(WorkbenchApiSettings(allow_unsafe_local_mutations=True))

        async def request() -> httpx.Response:
            async with lifespan_client(app) as client:
                return await client.delete("/config-snapshots/%73napshot")

        response = asyncio.run(request())

        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(
            response.json(),
            {"detail": MUTATION_PROOF_REQUIRED_DETAIL},
        )

    def test_trailing_slash_redirect_preserves_canonical_mutation_protection(
        self,
    ) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.api._security import MUTATION_PROOF_REQUIRED_DETAIL
        from emperor_workbench.settings import WorkbenchApiSettings

        app = create_app(WorkbenchApiSettings(allow_unsafe_local_mutations=True))

        async def request() -> tuple[httpx.Response, httpx.Response]:
            async with lifespan_client(
                app,
                follow_redirects=False,
            ) as client:
                redirected = await client.post(
                    "/config-snapshots/",
                    content=b"not-json",
                    headers={"content-type": "application/json"},
                )
                canonical = await client.post(
                    redirected.headers["location"],
                    content=b"not-json",
                    headers={"content-type": "application/json"},
                )
                return redirected, canonical

        redirected, canonical = asyncio.run(request())

        self.assertEqual(redirected.status_code, 307, redirected.text)
        self.assertEqual(
            httpx.URL(redirected.headers["location"]).path,
            "/config-snapshots",
        )
        self.assertEqual(canonical.status_code, 403, canonical.text)
        self.assertEqual(
            canonical.json(),
            {"detail": MUTATION_PROOF_REQUIRED_DETAIL},
        )

    def test_operational_policy_rejection_precedes_request_body_consumption(
        self,
    ) -> None:
        from emperor_workbench.api import create_app
        from emperor_workbench.api._security import (
            MUTATION_HEADER_NAME,
            MUTATION_HEADER_VALUE,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        app = create_app(WorkbenchApiSettings(allow_unsafe_local_mutations=False))

        async def request() -> httpx.Response:
            async with lifespan_client(app) as client:
                return await client.post(
                    "/config-snapshots",
                    content=b"not-json",
                    headers={
                        "content-type": "application/json",
                        MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                )

        response = asyncio.run(request())

        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(
            response.json(),
            {"detail": "Local mutation endpoints are disabled"},
        )

    def test_middleware_respects_first_matching_operation_order(self) -> None:
        from emperor_workbench.api._container import WorkbenchContainerSlot
        from emperor_workbench.api._middleware import configure_middleware
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            build_http_operation_catalog,
            declare_http_operation,
        )
        from emperor_workbench.settings import WorkbenchApiSettings

        settings = WorkbenchApiSettings(allow_unsafe_local_mutations=False)
        app = FastAPI()

        @app.post("/items/{name}")
        @declare_http_operation(HttpOperationPolicy.READ_ONLY)
        async def read_item(name: str) -> dict[str, str]:
            return {"handler": "read-only", "name": name}

        async def get_settings() -> WorkbenchApiSettings:
            return settings

        settings_dependency = Depends(get_settings)

        @app.post("/items/special")
        @declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
        async def mutate_special(settings=settings_dependency) -> dict[str, str]:
            return {"handler": "mutation"}

        configure_middleware(
            app,
            settings,
            build_http_operation_catalog(app.routes),
            container_slot=WorkbenchContainerSlot(),
        )

        async def request() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://localhost",
            ) as client:
                return await client.post("/items/special")

        response = asyncio.run(request())

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {"handler": "read-only", "name": "special"},
        )

    def test_catalog_rejects_opaque_non_safe_starlette_route(self) -> None:
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        from emperor_workbench.api._mutations import (
            MutationPolicyConfigurationError,
            build_http_operation_catalog,
        )

        app = FastAPI()

        async def raw_post(_request: Request) -> JSONResponse:
            return JSONResponse({"called": True})

        app.add_route("/raw", raw_post, methods=["POST"])

        with self.assertRaises(MutationPolicyConfigurationError):
            build_http_operation_catalog(app.routes)

    def test_first_matching_mutation_cannot_be_shadowed_by_later_read_only_route(
        self,
    ) -> None:
        from emperor_workbench.api._container import WorkbenchContainerSlot
        from emperor_workbench.api._middleware import configure_middleware
        from emperor_workbench.api._mutations import (
            HttpOperationPolicy,
            build_http_operation_catalog,
            declare_http_operation,
        )
        from emperor_workbench.api._security import MUTATION_PROOF_REQUIRED_DETAIL
        from emperor_workbench.settings import WorkbenchApiSettings

        settings = WorkbenchApiSettings(allow_unsafe_local_mutations=True)
        app = FastAPI()

        async def get_settings() -> WorkbenchApiSettings:
            return settings

        settings_dependency = Depends(get_settings)

        @app.post("/items/{name}")
        @declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
        async def mutate_item(
            name: str,
            settings=settings_dependency,
        ) -> dict[str, str]:
            return {"handler": "mutation", "name": name}

        @app.post("/items/special")
        @declare_http_operation(HttpOperationPolicy.READ_ONLY)
        async def read_special() -> dict[str, str]:
            return {"handler": "read-only"}

        configure_middleware(
            app,
            settings,
            build_http_operation_catalog(app.routes),
            container_slot=WorkbenchContainerSlot(),
        )

        async def request() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://localhost",
            ) as client:
                return await client.post("/items/special")

        response = asyncio.run(request())

        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(
            response.json(),
            {"detail": MUTATION_PROOF_REQUIRED_DETAIL},
        )


if __name__ == "__main__":
    unittest.main()
