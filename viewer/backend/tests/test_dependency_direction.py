from __future__ import annotations

import ast
import unittest
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_PACKAGE_DIRS = ("emperor", "models")
CORE_PACKAGE_ROOTS = frozenset(CORE_PACKAGE_DIRS)
VIEWER_BACKEND_DIR = REPO_ROOT / "viewer" / "backend"


@dataclass(frozen=True)
class AbsoluteImport:
    path: Path
    line_number: int
    module: str

    @property
    def root(self) -> str:
        return self.module.split(".", maxsplit=1)[0]

    def format_for_failure(self) -> str:
        relative_path = self.path.relative_to(REPO_ROOT)
        return f"{relative_path}:{self.line_number}: {self.module}"


def _python_files(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def _absolute_imports(path: Path) -> Iterator[AbsoluteImport]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        relative_path = path.relative_to(REPO_ROOT)
        raise AssertionError(f"Could not parse {relative_path}: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield AbsoluteImport(path, node.lineno, alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield AbsoluteImport(path, node.lineno, node.module)


def _absolute_imports_under(roots: Iterable[Path]) -> Iterator[AbsoluteImport]:
    for path in _python_files(roots):
        yield from _absolute_imports(path)


class DependencyDirectionTests(unittest.TestCase):
    def test_core_packages_do_not_import_viewer(self) -> None:
        core_roots = [REPO_ROOT / package_dir for package_dir in CORE_PACKAGE_DIRS]
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under(core_roots)
            if source_import.root == "viewer"
        ]

        self.assertEqual([], violations)

    def test_viewer_backend_may_import_core_packages(self) -> None:
        imported_core_roots = {
            source_import.root
            for source_import in _absolute_imports_under([VIEWER_BACKEND_DIR])
            if source_import.root in CORE_PACKAGE_ROOTS
        }

        self.assertEqual(CORE_PACKAGE_ROOTS, imported_core_roots)
