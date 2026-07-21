from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

MANIFEST_RELATIVE_PATH = Path("tests/architecture/emperor_test_manifest.toml")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_family(project_root: Path, family_name: str) -> dict[str, Any]:
    with (project_root / MANIFEST_RELATIVE_PATH).open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    try:
        return manifest["families"][family_name]
    except KeyError as error:
        available = ", ".join(sorted(manifest["families"]))
        raise ValueError(
            f"Unknown Emperor test family {family_name!r}. Available: {available}"
        ) from error


def resolve_test_paths(
    project_root: Path,
    patterns: list[str],
) -> tuple[Path, ...]:
    resolved = {
        path.resolve()
        for pattern in patterns
        for path in project_root.glob(pattern)
        if path.is_file()
    }
    return tuple(sorted(resolved))


def test_module_names(
    project_root: Path,
    test_paths: tuple[Path, ...],
) -> tuple[str, ...]:
    return tuple(
        ".".join(path.relative_to(project_root).with_suffix("").parts)
        for path in test_paths
    )


def coverage_include_argument(module_paths: list[str]) -> str:
    return "--include=" + ",".join(module_paths)


def _python_environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [
        str(project_root / "src"),
        str(project_root / "tests"),
        str(project_root),
    ]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_paths.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["PYTHONSAFEPATH"] = "1"
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    environment.setdefault("IPYTHONDIR", "/tmp/ipython")
    return environment


def _family_test_paths(
    project_root: Path,
    family: dict[str, Any],
) -> tuple[Path, ...]:
    patterns = [
        *family["focused_tests"],
        *family.get("integration_tests", []),
    ]
    paths = resolve_test_paths(project_root, patterns)
    if not paths:
        raise ValueError("The family has no resolvable tests.")
    return paths


def _family_test_modules(
    project_root: Path,
    family: dict[str, Any],
) -> tuple[str, ...]:
    return test_module_names(project_root, _family_test_paths(project_root, family))


def run_tests(project_root: Path, family: dict[str, Any]) -> int:
    command = [
        sys.executable,
        "-m",
        "unittest",
        *_family_test_modules(project_root, family),
    ]
    return subprocess.run(
        command,
        cwd=project_root,
        env=_python_environment(project_root),
        check=False,
    ).returncode


def run_coverage(project_root: Path, family: dict[str, Any]) -> int:
    environment = _python_environment(project_root)
    erase = subprocess.run(
        [sys.executable, "-m", "coverage", "erase"],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if erase.returncode != 0:
        return erase.returncode

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--source=emperor",
            "-m",
            "unittest",
            *_family_test_modules(project_root, family),
        ],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if run.returncode != 0:
        return run.returncode

    combine = subprocess.run(
        [sys.executable, "-m", "coverage", "combine"],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if combine.returncode != 0:
        return combine.returncode

    report = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "report",
            coverage_include_argument(family["modules"]),
        ],
        cwd=project_root,
        env=environment,
        check=False,
    )
    return report.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run manifest-driven Emperor family quality gates."
    )
    parser.add_argument("mode", choices=("tests", "coverage"))
    parser.add_argument("family")
    arguments = parser.parse_args(argv)

    try:
        family = load_family(PROJECT_ROOT, arguments.family)
        if arguments.mode == "tests":
            return run_tests(PROJECT_ROOT, family)
        return run_coverage(PROJECT_ROOT, family)
    except ValueError as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
