#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import configparser
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from email.parser import BytesParser
from email.policy import default
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
ROOT_RUNTIME_DEPENDENCIES = frozenset(
    {
        "datasets",
        "filelock",
        "gymnasium",
        "ipython",
        "lightning",
        "matplotlib",
        "numpy",
        "pillow",
        "tensorboard",
        "tokenizers",
        "torch",
        "torchmetrics",
        "torchtext",
        "torchvision",
    }
)
ROOT_DEV_DEPENDENCIES = frozenset(
    {
        "build",
        "coverage",
        "mutmut",
        "psutil",
        "ruff",
    }
)
ROOT_RUNTIME_REQUIREMENTS = frozenset(
    {
        "Pillow",
        "datasets",
        "filelock>=3.18,<4",
        "gymnasium",
        "ipython",
        "lightning",
        "matplotlib",
        "numpy",
        "tensorboard",
        "tokenizers",
        "torch",
        "torchmetrics",
        "torchtext",
        "torchvision",
    }
)
ROOT_DEV_REQUIREMENTS = frozenset(
    {
        "build>=1.2,<2",
        "coverage[toml]>=7.15,<8",
        "mutmut>=3.6,<4",
        "psutil>=7,<8",
        "ruff>=0.8,<0.14",
    }
)
WORKBENCH_RUNTIME_DEPENDENCIES = frozenset(
    {
        "emperor",
        "fastapi",
        "pydantic",
        "pydantic-settings",
        "pywin32",
        "starlette",
        "tensorboard",
        "torch",
        "typing-extensions",
        "uvicorn",
    }
)
WORKBENCH_DEV_DEPENDENCIES = frozenset(
    {
        "httpx",
        "psutil",
        "pytest",
        "ruff",
    }
)
WORKBENCH_RUNTIME_REQUIREMENTS = frozenset(
    {
        "emperor>=0.1,<0.2",
        "fastapi>=0.139,<0.140",
        "pydantic-settings",
        "pydantic>=2.7,<3",
        "pywin32>=311; sys_platform == 'win32'",
        "starlette>=1.3,<1.4",
        "tensorboard",
        "torch",
        "typing-extensions>=4.8,<5",
        "uvicorn[standard]>=0.51,<0.52",
    }
)
WORKBENCH_DEV_REQUIREMENTS = frozenset(
    {
        "httpx>=0.28,<0.29",
        "psutil>=7,<8",
        "pytest>=9,<10",
        "ruff>=0.8,<0.14",
    }
)
ROOT_IMPORT_DEPENDENCIES = {
    "IPython": "ipython",
    "PIL": "pillow",
    "datasets": "datasets",
    "filelock": "filelock",
    "gymnasium": "gymnasium",
    "lightning": "lightning",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "torchmetrics": "torchmetrics",
    "torchtext": "torchtext",
    "torchvision": "torchvision",
}
WORKBENCH_IMPORT_DEPENDENCIES = {
    "emperor": "emperor",
    "fastapi": "fastapi",
    "model_runtime": "emperor",
    "models": "emperor",
    "ntsecuritycon": "pywin32",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "pywintypes": "pywin32",
    "starlette": "starlette",
    "tensorboard": "tensorboard",
    "torch": "torch",
    "typing_extensions": "typing-extensions",
    "uvicorn": "uvicorn",
    "win32api": "pywin32",
    "win32con": "pywin32",
    "win32event": "pywin32",
    "win32file": "pywin32",
    "win32job": "pywin32",
    "win32process": "pywin32",
    "win32security": "pywin32",
}
_DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_EXTRA_MARKER = re.compile(r"\bextra\s*==\s*(['\"])(?P<extra>[^'\"]+)\1")
FIRST_PARTY_IMPORT_NAMES = (
    "emperor",
    "emperor_workbench",
    "model_runtime",
    "models",
)
WORKBENCH_WORKER_MODULES = (
    "emperor_workbench.inspection.worker",
    "emperor_workbench.training_jobs.worker",
    "emperor_workbench.training_jobs.cgroup_worker",
)
WORKBENCH_REQUIRED_WORKER_MEMBERS = frozenset(
    f"{module_name.replace('.', '/')}.py" for module_name in WORKBENCH_WORKER_MODULES
)
WORKBENCH_REQUIRED_WHEEL_MEMBERS = frozenset(
    {
        "emperor_workbench/__init__.py",
        "emperor_workbench/__main__.py",
        "emperor_workbench/api/__init__.py",
        "emperor_workbench/cli.py",
        "emperor_workbench/inspection/__init__.py",
        "emperor_workbench/training_jobs/__init__.py",
        *WORKBENCH_REQUIRED_WORKER_MEMBERS,
    }
)
WORKBENCH_REQUIRED_SDIST_MEMBERS = frozenset(
    {
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "tests/__init__.py",
        "tests/e2e/browser_performance_server.py",
        "tests/e2e/contract_server.py",
        "tests/fixtures/inspection_contract_v1.json",
        "tests/support/http.py",
        "tests/support/inspection.py",
        "tests/support/model_packages.py",
        "tests/support/training_jobs.py",
        *(f"src/{member}" for member in WORKBENCH_REQUIRED_WHEEL_MEMBERS),
    }
)


class VerificationError(RuntimeError):
    """Raised when an artifact or installed-behavior contract is violated."""


def _canonical_dependency_name(requirement: str) -> str:
    match = _DEPENDENCY_NAME.match(requirement)
    if match is None:
        raise VerificationError(f"Invalid Requires-Dist value: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _normalize_marker(marker: str) -> str:
    return re.sub(r"\s+", " ", marker.replace("'", '"')).strip()


def _requirement_contract(
    requirement: str,
    *,
    strip_extra_marker: bool,
) -> str:
    requirement_body, separator, marker = requirement.partition(";")
    name_match = _DEPENDENCY_NAME.match(requirement_body)
    if name_match is None:
        raise VerificationError(f"Invalid dependency requirement: {requirement!r}")
    name = _canonical_dependency_name(requirement_body)
    remainder = requirement_body[name_match.end() :].strip()

    extras: tuple[str, ...] = ()
    if remainder.startswith("["):
        extras_end = remainder.find("]")
        if extras_end == -1:
            raise VerificationError(f"Invalid dependency extras: {requirement!r}")
        extras = tuple(
            sorted(
                extra.strip().lower()
                for extra in remainder[1:extras_end].split(",")
                if extra.strip()
            )
        )
        remainder = remainder[extras_end + 1 :].strip()

    specifiers = tuple(
        sorted(specifier.strip() for specifier in remainder.split(",") if specifier)
    )
    normalized_marker = _normalize_marker(marker) if separator else ""
    if strip_extra_marker and normalized_marker:
        extra_match = _EXTRA_MARKER.search(normalized_marker)
        if extra_match is None:
            raise VerificationError(
                f"Development dependency is missing its extra marker: {requirement!r}"
            )
        normalized_marker = (
            normalized_marker[: extra_match.start()]
            + normalized_marker[extra_match.end() :]
        ).strip()
        normalized_marker = re.sub(
            r"^(?:and|or)\s+|\s+(?:and|or)$",
            "",
            normalized_marker,
        ).strip()

    return "|".join(
        (
            name,
            f"extras={','.join(extras)}",
            f"specifiers={','.join(specifiers)}",
            f"marker={normalized_marker}",
        )
    )


def _wheel_dependency_groups(
    archive: zipfile.ZipFile,
    *,
    distribution_name: str,
) -> tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    frozenset[str],
]:
    metadata_members = [
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_members) != 1:
        raise VerificationError(
            f"Expected one wheel METADATA file, found {metadata_members!r}."
        )
    metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_members[0]))
    actual_name = re.sub(
        r"[-_.]+",
        "-",
        str(metadata.get("Name", "")).strip(),
    ).lower()
    expected_name = re.sub(r"[-_.]+", "-", distribution_name).lower()
    if actual_name != expected_name:
        raise VerificationError(
            "Wheel METADATA distribution name differs from the expected "
            f"distribution: expected={expected_name!r}, actual={actual_name!r}."
        )

    runtime_dependencies: set[str] = set()
    dev_dependencies: set[str] = set()
    runtime_requirements: set[str] = set()
    dev_requirements: set[str] = set()
    for requirement in metadata.get_all("Requires-Dist", []):
        dependency_name = _canonical_dependency_name(requirement)
        extra_match = _EXTRA_MARKER.search(requirement)
        if extra_match is None:
            runtime_dependencies.add(dependency_name)
            runtime_requirements.add(requirement)
            continue
        extra = extra_match.group("extra")
        if extra != "dev":
            raise VerificationError(
                f"Unexpected dependency extra {extra!r}: {requirement!r}"
            )
        dev_dependencies.add(dependency_name)
        dev_requirements.add(requirement)
    return (
        frozenset(runtime_dependencies),
        frozenset(dev_dependencies),
        frozenset(runtime_requirements),
        frozenset(dev_requirements),
    )


def _wheel_imported_dependencies(
    archive: zipfile.ZipFile,
    *,
    package_prefixes: tuple[str, ...],
    first_party_imports: frozenset[str],
    import_dependencies: dict[str, str],
) -> frozenset[str]:
    imported_dependencies: set[str] = set()
    unmapped_imports: set[str] = set()
    for member_name in archive.namelist():
        if not member_name.endswith(".py") or not member_name.startswith(
            package_prefixes
        ):
            continue
        source = archive.read(member_name).decode("utf-8")
        tree = ast.parse(source, member_name)
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
            ):
                imported_modules.add(node.module.partition(".")[0])
        for module_name in imported_modules:
            if (
                module_name in sys.stdlib_module_names
                or module_name in first_party_imports
            ):
                continue
            dependency_name = import_dependencies.get(module_name)
            if dependency_name is None:
                unmapped_imports.add(module_name)
            else:
                imported_dependencies.add(dependency_name)
    if unmapped_imports:
        raise VerificationError(
            "Wheel source imports external modules without a declared "
            f"distribution mapping: {sorted(unmapped_imports)}"
        )
    return frozenset(imported_dependencies)


def _verify_direct_dependencies(
    archive: zipfile.ZipFile,
    *,
    distribution_name: str,
    expected_runtime: frozenset[str],
    expected_dev: frozenset[str],
    expected_runtime_requirements: frozenset[str],
    expected_dev_requirements: frozenset[str],
    package_prefixes: tuple[str, ...],
    first_party_imports: frozenset[str],
    import_dependencies: dict[str, str],
) -> dict[str, list[str]]:
    (
        runtime_dependencies,
        dev_dependencies,
        runtime_requirements,
        dev_requirements,
    ) = _wheel_dependency_groups(archive, distribution_name=distribution_name)
    if runtime_dependencies != expected_runtime:
        raise VerificationError(
            f"{distribution_name} runtime dependencies differ from the direct "
            "dependency contract: "
            f"missing={sorted(expected_runtime - runtime_dependencies)}, "
            f"unexpected={sorted(runtime_dependencies - expected_runtime)}."
        )
    if dev_dependencies != expected_dev:
        raise VerificationError(
            f"{distribution_name} dev dependencies differ from the direct "
            "dependency contract: "
            f"missing={sorted(expected_dev - dev_dependencies)}, "
            f"unexpected={sorted(dev_dependencies - expected_dev)}."
        )
    actual_runtime_contracts = {
        _requirement_contract(requirement, strip_extra_marker=False)
        for requirement in runtime_requirements
    }
    expected_runtime_contracts = {
        _requirement_contract(requirement, strip_extra_marker=False)
        for requirement in expected_runtime_requirements
    }
    actual_dev_contracts = {
        _requirement_contract(requirement, strip_extra_marker=True)
        for requirement in dev_requirements
    }
    expected_dev_contracts = {
        _requirement_contract(requirement, strip_extra_marker=False)
        for requirement in expected_dev_requirements
    }
    if actual_runtime_contracts != expected_runtime_contracts:
        missing_runtime_contracts = sorted(
            expected_runtime_contracts - actual_runtime_contracts
        )
        unexpected_runtime_contracts = sorted(
            actual_runtime_contracts - expected_runtime_contracts
        )
        raise VerificationError(
            f"{distribution_name} runtime requirement metadata differs from "
            "the version/extras/marker contract: "
            f"missing={missing_runtime_contracts}, "
            f"unexpected={unexpected_runtime_contracts}."
        )
    if actual_dev_contracts != expected_dev_contracts:
        raise VerificationError(
            f"{distribution_name} dev requirement metadata differs from the "
            "version/extras contract: "
            f"missing={sorted(expected_dev_contracts - actual_dev_contracts)}, "
            f"unexpected={sorted(actual_dev_contracts - expected_dev_contracts)}."
        )

    imported_dependencies = _wheel_imported_dependencies(
        archive,
        package_prefixes=package_prefixes,
        first_party_imports=first_party_imports,
        import_dependencies=import_dependencies,
    )
    missing_import_dependencies = imported_dependencies - runtime_dependencies
    if missing_import_dependencies:
        raise VerificationError(
            f"{distribution_name} source imports undeclared direct dependencies: "
            f"{sorted(missing_import_dependencies)}"
        )
    return {
        "runtime_dependencies": sorted(runtime_dependencies),
        "dev_dependencies": sorted(dev_dependencies),
        "imported_dependencies": sorted(imported_dependencies),
    }


def _wheel_console_scripts(archive: zipfile.ZipFile) -> dict[str, str]:
    members = [
        name
        for name in archive.namelist()
        if name.endswith(".dist-info/entry_points.txt")
    ]
    if len(members) != 1:
        raise VerificationError(
            f"Expected one wheel entry_points.txt file, found {members!r}."
        )
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(archive.read(members[0]).decode("utf-8"))
    if not parser.has_section("console_scripts"):
        raise VerificationError("Wheel has no console_scripts entry-point group.")
    return dict(parser.items("console_scripts"))


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


def _verify_wheel(wheel: Path) -> dict[str, int | list[str]]:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        dependency_manifest = _verify_direct_dependencies(
            archive,
            distribution_name="emperor",
            expected_runtime=ROOT_RUNTIME_DEPENDENCIES,
            expected_dev=ROOT_DEV_DEPENDENCIES,
            expected_runtime_requirements=ROOT_RUNTIME_REQUIREMENTS,
            expected_dev_requirements=ROOT_DEV_REQUIREMENTS,
            package_prefixes=("emperor/", "model_runtime/", "models/"),
            first_party_imports=frozenset(
                {
                    "emperor",
                    "model_runtime",
                    "models",
                }
            ),
            import_dependencies=ROOT_IMPORT_DEPENDENCIES,
        )

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
    return {**counts, **dependency_manifest}


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


def _verify_workbench_wheel(wheel: Path) -> dict[str, int | list[str]]:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        console_scripts = _wheel_console_scripts(archive)
        dependency_manifest = _verify_direct_dependencies(
            archive,
            distribution_name="emperor-workbench",
            expected_runtime=WORKBENCH_RUNTIME_DEPENDENCIES,
            expected_dev=WORKBENCH_DEV_DEPENDENCIES,
            expected_runtime_requirements=WORKBENCH_RUNTIME_REQUIREMENTS,
            expected_dev_requirements=WORKBENCH_DEV_REQUIREMENTS,
            package_prefixes=("emperor_workbench/",),
            first_party_imports=frozenset({"emperor_workbench"}),
            import_dependencies=WORKBENCH_IMPORT_DEPENDENCIES,
        )
    expected_console_scripts = {
        "emperor-workbench": "emperor_workbench.cli:main",
    }
    if console_scripts != expected_console_scripts:
        raise VerificationError(
            "Workbench console scripts differ from the canonical entry-point "
            f"contract: expected={expected_console_scripts}, "
            f"actual={console_scripts}."
        )

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
    executable_tests = sorted(
        name
        for name in names
        if PurePosixPath(name).name.startswith("test") and name.endswith(".py")
    )
    if executable_tests:
        raise VerificationError(
            f"Tests leaked into Workbench wheel: {executable_tests}"
        )
    missing = sorted(WORKBENCH_REQUIRED_WHEEL_MEMBERS.difference(names))
    if missing:
        raise VerificationError(f"Workbench wheel is missing: {missing}")
    forbidden = {"emperor_workbench/launch.py"}.intersection(names)
    if forbidden:
        raise VerificationError(
            f"Legacy Workbench launcher leaked into wheel: {sorted(forbidden)}"
        )
    counts = {
        "emperor_workbench": sum(
            name.startswith("emperor_workbench/") for name in names
        ),
        "metadata": sum(".dist-info/" in name for name in names),
        "total": len(names),
    }
    return {
        **counts,
        **dependency_manifest,
        "console_scripts": sorted(
            f"{name}={target}" for name, target in console_scripts.items()
        ),
        "required_members": sorted(WORKBENCH_REQUIRED_WHEEL_MEMBERS),
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
