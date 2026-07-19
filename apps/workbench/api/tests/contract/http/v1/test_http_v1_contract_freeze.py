from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts

CONTRACT_DIRECTORY = Path(__file__).resolve().parent
SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
OPENAPI_HTTP_METHODS = frozenset(
    {"delete", "get", "head", "options", "patch", "post", "put", "trace"}
)


def _read_json_fixture(name: str) -> Any:
    return json.loads((CONTRACT_DIRECTORY / name).read_text(encoding="utf-8"))


def _isolated_runtime_values(root: Path) -> dict[str, str]:
    return {
        "APPDATA": str(root / "appdata"),
        "HOME": str(root / "home"),
        "LOCALAPPDATA": str(root / "local-appdata"),
        "MPLCONFIGDIR": str(root / "matplotlib"),
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TMPDIR": str(root / "tmp"),
        "TORCH_HOME": str(root / "torch"),
        "USERPROFILE": str(root / "home"),
        "WORKBENCH_API_LOGS_ROOT": str(root / "logs"),
        "WORKBENCH_API_SNAPSHOTS_ROOT": str(root / "snapshots"),
        "WORKBENCH_API_STATE_ROOT": str(root / "state"),
        "XDG_CACHE_HOME": str(root / "xdg-cache"),
        "XDG_CONFIG_HOME": str(root / "xdg-config"),
        "XDG_DATA_HOME": str(root / "xdg-data"),
        "XDG_STATE_HOME": str(root / "xdg-state"),
    }


@contextmanager
def _isolated_runtime_environment(root: Path) -> Iterator[None]:
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("WORKBENCH_API_")
    }
    environment.update(_isolated_runtime_values(root))
    with patch.dict(os.environ, environment, clear=True):
        yield


@contextmanager
def _contract_app() -> Iterator[FastAPI]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with _isolated_runtime_environment(root):
            from emperor_workbench.api import create_app
            from emperor_workbench.settings import WorkbenchApiSettings

            yield create_app(
                WorkbenchApiSettings(
                    logs_root=str(root / "logs"),
                    snapshots_root=str(root / "snapshots"),
                    state_root=str(root / "state"),
                )
            )


def _response_model_name(route: Any) -> str:
    model = route.response_model
    name = getattr(model, "__name__", None)
    if not isinstance(name, str):
        raise AssertionError(f"{route.path} has no named response model")
    return name


def _route_manifest(app: FastAPI) -> list[dict[str, Any]]:
    openapi = app.openapi()
    operations: list[dict[str, Any]] = []
    for route in iter_route_contexts(app.routes):
        if not isinstance(route.original_route, APIRoute):
            continue
        for method in sorted(route.methods or ()):
            operation = openapi["paths"][route.path][method.lower()]
            policies = tuple(
                getattr(
                    route.endpoint,
                    "__workbench_http_operation_policies__",
                    (),
                )
            )
            if method in SAFE_HTTP_METHODS:
                mutation_policy = "safe"
                if policies:
                    raise AssertionError(
                        f"{method} {route.path} unexpectedly declares a policy"
                    )
            else:
                if len(policies) != 1:
                    raise AssertionError(
                        f"{method} {route.path} must declare exactly one policy"
                    )
                mutation_policy = policies[0].value

            security = operation.get("security", [])
            authentication = sorted(
                scheme for requirement in security for scheme in requirement
            )
            operations.append(
                {
                    "authentication": authentication,
                    "method": method,
                    "mutationPolicy": mutation_policy,
                    "operationId": operation["operationId"],
                    "path": route.path,
                    "responseModel": _response_model_name(route),
                }
            )
    return sorted(operations, key=lambda item: (item["path"], item["method"]))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_openapi_manifest(app: FastAPI) -> dict[str, Any]:
    document = app.openapi()
    operations = [
        {
            "method": method.upper(),
            "path": path,
            "sha256": _canonical_sha256(operation),
        }
        for path, path_item in sorted(document["paths"].items())
        for method, operation in sorted(path_item.items())
        if method in OPENAPI_HTTP_METHODS
    ]
    schemas = [
        {
            "name": name,
            "sha256": _canonical_sha256(schema),
        }
        for name, schema in sorted(
            document.get("components", {}).get("schemas", {}).items()
        )
    ]
    return {
        "canonicalization": (
            "UTF-8 JSON; object keys sorted; compact separators; array order preserved"
        ),
        "documentSha256": _canonical_sha256(document),
        "info": document["info"],
        "openapi": document["openapi"],
        "operationCount": len(operations),
        "operations": operations,
        "schemaCount": len(schemas),
        "schemas": schemas,
        "securitySchemes": document.get("components", {}).get(
            "securitySchemes",
            {},
        ),
    }


class HttpV1ContractFreezeTests(unittest.TestCase):
    def test_contract_runtime_environment_is_scoped_and_isolated(self) -> None:
        unsafe_environment = {
            "WORKBENCH_API_LOGS_ROOT": "/unsafe/logs",
            "WORKBENCH_API_SNAPSHOTS_ROOT": "/unsafe/snapshots",
            "WORKBENCH_API_STATE_ROOT": "/unsafe/state",
            "WORKBENCH_API_TOKEN": "unsafe-token",
        }
        with patch.dict(os.environ, unsafe_environment, clear=False):
            before = dict(os.environ)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with _isolated_runtime_environment(root):
                    expected = _isolated_runtime_values(root)
                    for key, value in expected.items():
                        with self.subTest(variable=key):
                            self.assertEqual(value, os.environ[key])
                    self.assertNotIn("WORKBENCH_API_TOKEN", os.environ)
            self.assertEqual(before, dict(os.environ))

    def test_normalized_openapi_matches_frozen_http_v1_contract(self) -> None:
        with _contract_app() as app:
            actual = _normalized_openapi_manifest(app)

        self.assertEqual(
            actual,
            _read_json_fixture("openapi.normalized.json"),
        )

    def test_route_manifest_matches_frozen_http_v1_contract(self) -> None:
        with _contract_app() as app:
            actual = _route_manifest(app)

        self.assertEqual(actual, _read_json_fixture("routes.json"))


if __name__ == "__main__":
    unittest.main()
