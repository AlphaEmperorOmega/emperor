from __future__ import annotations

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMPEROR_ROOT = PROJECT_ROOT / "src" / "emperor"
VALIDATOR_MODULE_FILENAMES = {
    "_validation.py",
    "_validator.py",
    "validation.py",
    "validator.py",
}


# This is a migration ledger, not a permanent exception list. Remove exactly one
# entry when the corresponding validator is migrated. The final state is empty.
LEGACY_HARDCODED_DISPATCH: set[str] = set()

LEGACY_STATIC_VALIDATE_ENTRYPOINTS: set[str] = set()

LEGACY_DIRECT_VALIDATOR_CALLS: set[str] = set()


def _validator_source_files() -> list[Path]:
    return sorted(
        path
        for path in EMPEROR_ROOT.rglob("*.py")
        if path.name in VALIDATOR_MODULE_FILENAMES
    )


def _production_source_files() -> list[Path]:
    return sorted(
        path
        for path in EMPEROR_ROOT.rglob("*.py")
        if path.name not in VALIDATOR_MODULE_FILENAMES
    )


def _qualified_class_name(path: Path, class_name: str) -> str:
    return f"{path.relative_to(PROJECT_ROOT).as_posix()}:{class_name}"


def _is_validator_class(node: ast.ClassDef) -> bool:
    return "Validator" in node.name or node.name.endswith("ValidationMixin")


def _decorator_names(method: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {
        decorator.id
        for decorator in method.decorator_list
        if isinstance(decorator, ast.Name)
    }


class TestValidatorConventions(unittest.TestCase):
    def test_hardcoded_validator_dispatch_matches_migration_ledger(self):
        actual = set()
        for path in _validator_source_files():
            tree = ast.parse(path.read_text())
            for validator_class in (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and _is_validator_class(node)
            ):
                for call in (
                    node
                    for node in ast.walk(validator_class)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                ):
                    owner = call.func.value
                    if isinstance(owner, ast.Name) and (
                        "Validator" in owner.id or owner.id.endswith("ValidationMixin")
                    ):
                        actual.add(_qualified_class_name(path, validator_class.name))

        self.assertEqual(actual, LEGACY_HARDCODED_DISPATCH)

    def test_static_validate_entrypoints_match_migration_ledger(self):
        actual = set()
        for path in _validator_source_files():
            tree = ast.parse(path.read_text())
            for validator_class in (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and _is_validator_class(node)
            ):
                for method in validator_class.body:
                    if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if method.name != "validate":
                        continue
                    if "classmethod" not in _decorator_names(method):
                        actual.add(_qualified_class_name(path, validator_class.name))

        self.assertEqual(actual, LEGACY_STATIC_VALIDATE_ENTRYPOINTS)

    def test_direct_production_calls_match_migration_ledger(self):
        actual = set()
        for path in _production_source_files():
            tree = ast.parse(path.read_text())
            for call in (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                owner = call.func.value
                if isinstance(owner, ast.Name) and "Validator" in owner.id:
                    actual.add(owner.id)

        self.assertEqual(actual, LEGACY_DIRECT_VALIDATOR_CALLS)

    def test_validators_remain_stateless(self):
        violations = []
        for path in _validator_source_files():
            tree = ast.parse(path.read_text())
            for validator_class in (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and _is_validator_class(node)
            ):
                for method in validator_class.body:
                    if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    decorators = _decorator_names(method)
                    if method.name == "__init__" or not decorators.intersection(
                        {"classmethod", "staticmethod"}
                    ):
                        violations.append(
                            f"{_qualified_class_name(path, validator_class.name)}."
                            f"{method.name}"
                        )

        self.assertEqual(violations, [])

    def test_validators_are_never_instantiated(self):
        violations = []
        for path in EMPEROR_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                if isinstance(call.func, ast.Name) and "Validator" in call.func.id:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT).as_posix()}:{call.lineno}:"
                        f"{call.func.id}"
                    )

        self.assertEqual(violations, [])
