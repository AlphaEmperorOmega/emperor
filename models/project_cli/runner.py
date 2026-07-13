"""Run the repository test Interface from the project-owned CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def run_tests(argv: Sequence[str], *, repository_root: Path | None = None) -> int:
    root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
    test_root = root / "tests"
    try:
        distribution("emperor")
    except PackageNotFoundError:
        print(
            "Error: install Emperor in the active environment before running tests.",
            file=sys.stderr,
        )
        print("Run: python -m pip install --no-deps -e .", file=sys.stderr)
        return 1

    environment = {
        **os.environ,
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": str(test_root),
    }
    arguments = list(argv)
    if arguments:
        file_name = arguments[0].removesuffix(".py")
        test_path = test_root / "unit" / f"test_{file_name}.py"
        if not test_path.is_file():
            print(f"Error: {test_path} not found", file=sys.stderr)
            return 1
        target = f"unit.test_{file_name}"
        if len(arguments) >= 2:
            target += f".{arguments[1]}"
        if len(arguments) >= 3:
            target += f".{arguments[2]}"
        command = [sys.executable, "-P", "-m", "unittest", "-f", target]
    else:
        command = [
            sys.executable,
            "-P",
            "-m",
            "unittest",
            "discover",
            "-f",
            "-s",
            str(test_root),
            "-t",
            str(test_root),
        ]
    return subprocess.run(  # noqa: S603 - fixed interpreter and unittest module
        command,
        cwd=root,
        env=environment,
        check=False,
    ).returncode


__all__ = ["run_tests"]
