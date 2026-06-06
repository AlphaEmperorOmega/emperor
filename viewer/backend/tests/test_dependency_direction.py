from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROTECTED_PACKAGES = ("models", "emperor")
FORBIDDEN_IMPORT_ROOT = "viewer"


def repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if all((parent / package).is_dir() for package in PROTECTED_PACKAGES):
            return parent
    raise RuntimeError("Could not locate repository root.")


def python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def is_forbidden_import(module_name: str) -> bool:
    return (
        module_name == FORBIDDEN_IMPORT_ROOT
        or module_name.startswith(f"{FORBIDDEN_IMPORT_ROOT}.")
    )


def import_violations(path: Path, repo_root: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    relative_path = path.relative_to(repo_root).as_posix()
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_forbidden_import(alias.name):
                    violations.append(
                        f"{relative_path}:{node.lineno} imports {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            if is_forbidden_import(node.module):
                violations.append(
                    f"{relative_path}:{node.lineno} imports {node.module}"
                )

    return violations


class DependencyDirectionTests(unittest.TestCase):
    def test_models_and_emperor_do_not_import_viewer(self) -> None:
        repo_root = repository_root()
        violations = []

        for package in PROTECTED_PACKAGES:
            for path in python_files(repo_root / package):
                violations.extend(import_violations(path, repo_root))

        if violations:
            self.fail(
                "models/ and emperor/ must not import viewer/:\n"
                + "\n".join(violations)
            )


if __name__ == "__main__":
    unittest.main()
