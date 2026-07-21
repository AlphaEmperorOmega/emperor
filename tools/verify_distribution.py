#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path, PurePosixPath
from sysconfig import get_path

PROJECT_FILES = (
    "CONTEXT.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "experiment.sh",
    "mise.toml",
    "pyproject.toml",
    "run_test.sh",
)
PROJECT_TREES = (
    "constraints",
    "docs",
    "src",
    "tests",
)
IGNORED_TREE_NAMES = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "*.egg-info",
)


class VerificationError(RuntimeError):
    """Raised when an artifact or installed-behavior contract is violated."""


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        output = "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        )
        raise VerificationError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{output}"
        )
    return completed.stdout.strip()


def _copy_project(repository: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for relative_name in PROJECT_FILES:
        shutil.copy2(repository / relative_name, destination / relative_name)
    for relative_name in PROJECT_TREES:
        shutil.copytree(
            repository / relative_name,
            destination / relative_name,
            ignore=shutil.ignore_patterns(*IGNORED_TREE_NAMES),
        )


def _copy_workbench(repository: Path, destination: Path) -> None:
    shutil.copytree(
        repository / "apps" / "workbench" / "api",
        destination,
        ignore=shutil.ignore_patterns(
            *IGNORED_TREE_NAMES,
            ".next",
            ".runtime",
            "node_modules",
        ),
    )


def _build_artifacts(source: Path, artifact_root: Path) -> tuple[Path, Path]:
    artifact_root.mkdir()
    build_code = """
from setuptools.build_meta import build_sdist, build_wheel

print(build_wheel('artifacts'))
print(build_sdist('artifacts'))
"""
    _run([sys.executable, "-P", "-c", build_code], cwd=source)
    wheels = sorted((source / "artifacts").glob("*.whl"))
    sdists = sorted((source / "artifacts").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise VerificationError(
            f"Expected one wheel and one sdist, found {wheels!r} and {sdists!r}."
        )
    wheel = artifact_root / wheels[0].name
    sdist = artifact_root / sdists[0].name
    shutil.copy2(wheels[0], wheel)
    shutil.copy2(sdists[0], sdist)
    return wheel, sdist


def _verify_wheel(wheel: Path) -> dict[str, int]:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]

    unexpected = sorted(
        name
        for name in names
        if not (
            name.startswith("emperor/")
            or name.startswith("models/")
            or name.startswith("model_runtime/")
            or (name.startswith("emperor-") and ".dist-info/" in name)
        )
    )
    if unexpected:
        raise VerificationError(f"Unexpected wheel members: {unexpected}")

    executable_tests = sorted(
        name
        for name in names
        if PurePosixPath(name).name.startswith("test") and name.endswith(".py")
    )
    if executable_tests:
        raise VerificationError(f"Tests leaked into wheel: {executable_tests}")

    counts = {
        "emperor": sum(name.startswith("emperor/") for name in names),
        "models": sum(name.startswith("models/") for name in names),
        "model_runtime": sum(name.startswith("model_runtime/") for name in names),
        "metadata": sum(".dist-info/" in name for name in names),
        "total": len(names),
    }
    if any(counts[package] == 0 for package in ("emperor", "models", "model_runtime")):
        raise VerificationError(f"Installable packages missing from wheel: {counts}")
    return counts


def _strip_sdist_root(name: str) -> str:
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else ""


def _verify_sdist(sdist: Path) -> dict[str, int]:
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = [
            _strip_sdist_root(member.name)
            for member in archive.getmembers()
            if member.isfile()
        ]

    required_prefixes = (
        "docs/",
        "src/emperor/",
        "src/models/",
        "src/model_runtime/",
        "tests/",
    )
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in names)
    ]
    if missing:
        raise VerificationError(f"Source distribution is missing: {missing}")
    required_files = {
        "constraints/python-3.13-linux-x86_64-cuda-legacy.txt",
    }
    missing_files = sorted(required_files.difference(names))
    if missing_files:
        raise VerificationError(
            f"Source distribution is missing required files: {missing_files}"
        )
    leaked_workbench = sorted(
        name
        for name in names
        if name.startswith("apps/workbench/") or name.startswith("workbench/")
    )
    if leaked_workbench:
        raise VerificationError(
            f"Workbench leaked into Emperor source distribution: {leaked_workbench}"
        )
    return {
        "constraints": sum(name.startswith("constraints/") for name in names),
        "docs": sum(name.startswith("docs/") for name in names),
        "emperor": sum(name.startswith("src/emperor/") for name in names),
        "models": sum(name.startswith("src/models/") for name in names),
        "model_runtime": sum(name.startswith("src/model_runtime/") for name in names),
        "tests": sum(name.startswith("tests/") for name in names),
        "total": len(names),
    }


def _verify_workbench_wheel(wheel: Path) -> dict[str, int]:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]

    unexpected = sorted(
        name
        for name in names
        if not (
            name.startswith("emperor_workbench/")
            or (name.startswith("emperor_workbench-") and ".dist-info/" in name)
        )
    )
    if unexpected:
        raise VerificationError(f"Unexpected Workbench wheel members: {unexpected}")
    leaked_tests = sorted(
        name for name in names if name.startswith("emperor_workbench/tests/")
    )
    if leaked_tests:
        raise VerificationError(f"Tests leaked into Workbench wheel: {leaked_tests}")
    required = {
        "emperor_workbench/__init__.py",
        "emperor_workbench/__main__.py",
        "emperor_workbench/api/__init__.py",
        "emperor_workbench/cli.py",
    }
    missing = sorted(required.difference(names))
    if missing:
        raise VerificationError(f"Workbench wheel is missing: {missing}")
    return {
        "emperor_workbench": sum(
            name.startswith("emperor_workbench/") for name in names
        ),
        "metadata": sum(".dist-info/" in name for name in names),
        "total": len(names),
    }


def _verify_workbench_sdist(sdist: Path) -> dict[str, int]:
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = {
            _strip_sdist_root(member.name)
            for member in archive.getmembers()
            if member.isfile()
        }
    required = {
        "src/emperor_workbench/__init__.py",
        "src/emperor_workbench/__main__.py",
        "src/emperor_workbench/api/__init__.py",
        "src/emperor_workbench/cli.py",
        "pyproject.toml",
    }
    missing = sorted(required.difference(names))
    if missing:
        raise VerificationError(f"Workbench sdist is missing: {missing}")
    leaked = sorted(name for name in names if name.startswith("web/"))
    if leaked:
        raise VerificationError(
            f"Non-runtime files leaked into Workbench sdist: {leaked}"
        )
    return {
        "emperor_workbench": sum(
            name.startswith("src/emperor_workbench/") for name in names
        ),
        "total": len(names),
    }


def _create_environment(path: Path) -> Path:
    venv.EnvBuilder(with_pip=True).create(path)
    python = path / "bin" / "python"
    environment_site_packages = Path(
        _run(
            [
                str(python),
                "-P",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            cwd=path,
        )
    )
    (environment_site_packages / "emperor-build-dependencies.pth").write_text(
        f"{Path(get_path('purelib')).resolve()}\n",
        encoding="utf-8",
    )
    return python


def _install_editable(python: Path, source: Path, outside: Path) -> None:
    _run(
        [
            str(python),
            "-P",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--editable",
            str(source),
        ],
        cwd=outside,
    )


def _install_wheel(python: Path, wheel: Path, outside: Path) -> None:
    _run(
        [str(python), "-P", "-m", "pip", "install", "--no-deps", str(wheel)],
        cwd=outside,
    )


def _isolated_environment(extra_python_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONSAFEPATH"] = "1"
    env["MPLCONFIGDIR"] = tempfile.gettempdir()
    if extra_python_path is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = str(extra_python_path)
    return env


def _installed_smoke(python: Path, outside: Path) -> dict[str, object]:
    smoke_code = """
import json
from importlib.metadata import version

import emperor
import model_runtime
import models
from model_runtime.inspection import InspectionRequest, inspect_model
from models.catalog import discover_model_packages, model_package
from models.catalog import discover_model_identity_payloads as legacy_identities

identities = [package.to_identity_payload() for package in discover_model_packages()]
package = model_package('linears/linear')
config = package.build_configurations()[0]
model = package.build_model(config)
inspection = inspect_model(
    package,
    InspectionRequest(preset='baseline', dataset='Mnist'),
)
print(json.dumps({
    'config_class': f'{type(config).__module__}.{type(config).__qualname__}',
    'emperor_path': emperor.__file__,
    'model_runtime_path': model_runtime.__file__,
    'identities': identities,
    'inspection_nodes': len(inspection.nodes),
    'legacy_matches': legacy_identities() == identities,
    'model_class': f'{type(model).__module__}.{type(model).__qualname__}',
    'models_path': models.__file__,
    'version': version('emperor'),
}, sort_keys=True))
"""
    output = _run(
        [str(python), "-P", "-c", smoke_code],
        cwd=outside,
        env=_isolated_environment(),
    )
    payload = json.loads(output.splitlines()[-1])
    cli_types = _run(
        [str(python), "-P", "-m", "models.catalog", "--list-types"],
        cwd=outside,
        env=_isolated_environment(),
    ).splitlines()
    payload["cli_types"] = cli_types
    project_cli_output = _run(
        [str(python.parent / "emperor"), "--list-model-types"],
        cwd=outside,
        env=_isolated_environment(),
    ).splitlines()
    payload["project_cli_types"] = [
        line.removeprefix("  --model-type ")
        for line in project_cli_output
        if line.startswith("  --model-type ")
    ]
    if payload["project_cli_types"] != cli_types:
        raise VerificationError(
            "The installed emperor command does not expose the catalog model types."
        )
    return payload


def _installed_contract(python: Path, outside: Path, test_root: Path) -> None:
    _run(
        [
            str(python),
            "-P",
            "-m",
            "unittest",
            "-f",
            "contract.test_model_package_interface",
            "contract.test_dataset_metadata",
            "unit.test_runs_execution.RunsExecutionTests.test_no_search_plan_executes_exact_run_and_writes_portable_artifacts",
        ],
        cwd=outside,
        env=_isolated_environment(test_root),
    )


def _workbench_smoke(
    python: Path,
    outside: Path,
    state_root: Path,
) -> dict[str, object]:
    smoke_code = """
import asyncio
import json
import os
from importlib.metadata import version

from fastapi.routing import APIRoute, iter_route_contexts
from emperor_workbench.api import create_app
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.settings import WorkbenchApiSettings
import emperor_workbench

settings = WorkbenchApiSettings(
    logs_root=os.environ['WORKBENCH_SMOKE_LOGS'],
    snapshots_root=os.environ['WORKBENCH_SMOKE_SNAPSHOTS'],
    state_root=os.environ['WORKBENCH_SMOKE_STATE'],
    training_cancellation_mode='process-group',
)
app = create_app(settings)
routes = sorted(
    [method, context.path]
    for context in iter_route_contexts(app.routes)
    if isinstance(context.original_route, APIRoute)
    for method in context.methods or ()
)

async def catalog_identities():
    async with app.router.lifespan_context(app):
        return ModelPackageCatalog(
            app.state.workbench_container.project_adapter
        ).identities()

catalog = asyncio.run(catalog_identities())
print(json.dumps({
    'catalog_count': len(catalog),
    'route_count': len(routes),
    'title': app.title,
    'version': version('emperor-workbench'),
    'emperor_workbench_path': emperor_workbench.__file__,
}, sort_keys=True))
"""
    env = _isolated_environment()
    env["EMPEROR_PROJECT_ADAPTER_COMMAND"] = f"{python} -P -m models.adapter_cli"
    env["WORKBENCH_SMOKE_LOGS"] = str(state_root / "logs")
    env["WORKBENCH_SMOKE_SNAPSHOTS"] = str(state_root / "snapshots")
    env["WORKBENCH_SMOKE_STATE"] = str(state_root / "state")
    output = _run([str(python), "-P", "-c", smoke_code], cwd=outside, env=env)
    return json.loads(output.splitlines()[-1])


def _require_workbench_path_under(payload: dict[str, object], root: Path) -> None:
    installed_path = Path(str(payload["emperor_workbench_path"])).resolve()
    if not installed_path.is_relative_to(root.resolve()):
        raise VerificationError(
            "emperor_workbench_path resolved outside the expected install: "
            f"{installed_path}"
        )


def _workbench_semantic_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in payload.items() if key != "emperor_workbench_path"
    }


def _semantic_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"emperor_path", "model_runtime_path", "models_path"}
    }


def _require_path_under(payload: dict[str, object], root: Path) -> None:
    for key in ("emperor_path", "model_runtime_path", "models_path"):
        installed_path = Path(str(payload[key])).resolve()
        if not installed_path.is_relative_to(root.resolve()):
            raise VerificationError(
                f"{key} resolved outside the expected install: {installed_path}"
            )


def verify(
    repository: Path,
    *,
    include_workbench_smoke: bool = True,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="emperor-distribution-") as temporary:
        root = Path(temporary)
        build_source = root / "build-source"
        editable_source = root / "editable-source"
        workbench_source = root / "workbench-source"
        outside = root / "outside"
        outside.mkdir()
        _copy_project(repository, build_source)
        _copy_project(repository, editable_source)
        if include_workbench_smoke:
            _copy_workbench(repository, workbench_source)

        wheel, sdist = _build_artifacts(build_source, root / "artifacts")
        wheel_manifest = _verify_wheel(wheel)
        sdist_manifest = _verify_sdist(sdist)
        workbench_wheel: Path | None = None
        workbench_wheel_manifest: dict[str, int] | None = None
        workbench_sdist_manifest: dict[str, int] | None = None
        if include_workbench_smoke:
            workbench_wheel, workbench_sdist = _build_artifacts(
                workbench_source,
                root / "workbench-artifacts",
            )
            workbench_wheel_manifest = _verify_workbench_wheel(workbench_wheel)
            workbench_sdist_manifest = _verify_workbench_sdist(workbench_sdist)

        editable_environment = root / "editable-environment"
        regular_environment = root / "regular-environment"
        editable_python = _create_environment(editable_environment)
        regular_python = _create_environment(regular_environment)
        _install_editable(editable_python, editable_source, outside)
        _install_wheel(regular_python, wheel, outside)
        if include_workbench_smoke:
            assert workbench_wheel is not None
            _install_editable(editable_python, workbench_source, outside)
            _install_wheel(regular_python, workbench_wheel, outside)

        editable_payload = _installed_smoke(editable_python, outside)
        regular_payload = _installed_smoke(regular_python, outside)
        _installed_contract(editable_python, outside, build_source / "tests")
        _installed_contract(regular_python, outside, build_source / "tests")
        _require_path_under(editable_payload, editable_source)
        _require_path_under(regular_payload, regular_environment)
        if _semantic_payload(editable_payload) != _semantic_payload(regular_payload):
            raise VerificationError(
                "Editable and regular installs differ:\n"
                f"editable={editable_payload}\nregular={regular_payload}"
            )

        result = {
            "install_contract": _semantic_payload(editable_payload),
            "installed_contract_tests": 3,
            "sdist_manifest": sdist_manifest,
            "wheel_manifest": wheel_manifest,
        }
        if include_workbench_smoke:
            editable_workbench = _workbench_smoke(
                editable_python,
                outside,
                root / "editable-workbench-state",
            )
            regular_workbench = _workbench_smoke(
                regular_python,
                outside,
                root / "regular-workbench-state",
            )
            _require_workbench_path_under(editable_workbench, workbench_source)
            _require_workbench_path_under(regular_workbench, regular_environment)
            if _workbench_semantic_payload(
                editable_workbench
            ) != _workbench_semantic_payload(regular_workbench):
                raise VerificationError(
                    "Workbench startup differs between installs:\n"
                    f"editable={editable_workbench}\nregular={regular_workbench}"
                )
            result["workbench_sdist_manifest"] = workbench_sdist_manifest
            result["workbench_startup"] = _workbench_semantic_payload(
                editable_workbench
            )
            result["workbench_wheel_manifest"] = workbench_wheel_manifest
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and verify Emperor distributions without downloading dependencies."
        )
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--skip-workbench-smoke",
        action="store_true",
        help="Verify only the root distribution and installed consumer contract.",
    )
    args = parser.parse_args()
    result = verify(
        args.repository.resolve(),
        include_workbench_smoke=not args.skip_workbench_smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
