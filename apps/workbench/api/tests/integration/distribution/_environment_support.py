from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

from tests.support import REPOSITORY_ROOT

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "emperor-matplotlib"),
)

PROJECT_ROOT = REPOSITORY_ROOT

REQUIRED_BACKEND_TEST_MODULES = (
    ("torch", "torch"),
    ("fastapi", "fastapi"),
    ("tensorboard", "tensorboard"),
    ("pydantic", "pydantic"),
    ("pydantic-settings", "pydantic_settings"),
    ("httpx", "httpx"),
    ("lightning", "lightning.pytorch"),
    ("ruff", "ruff"),
    ("filelock", "filelock"),
    ("psutil", "psutil"),
)


def _load_emperor_dev():
    name = "_emperor_dev_environment_tests"
    spec = importlib.util.spec_from_file_location(
        name,
        PROJECT_ROOT / "tools" / "emperor_dev.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the portable launcher for testing.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


emperor_dev = _load_emperor_dev()

__all__ = [
    "PROJECT_ROOT",
    "REQUIRED_BACKEND_TEST_MODULES",
    "emperor_dev",
]
