from __future__ import annotations

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path

from tests.architecture._support import (
    API_ROOT,
    PACKAGE_NAME,
    load_manifest,
    manifest_records,
    owner_for,
    private_imports,
    source_module,
    source_paths,
)


@dataclass(frozen=True, order=True)
class PrivateTestImportViolation:
    source: str
    imported: str
    test_owner: str
    imported_owner: str


def _is_private_workbench_module(module_name: str) -> bool:
    return module_name.startswith(f"{PACKAGE_NAME}.") and any(
        part.startswith("_") for part in module_name.split(".")[1:]
    )


def _constant_strings(
    node: ast.expr,
    constants: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.Name):
        return constants.get(node.id, ())
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        values: list[str] = []
        for element in node.elts:
            element_values = _constant_strings(element, constants)
            if not element_values:
                return ()
            values.extend(element_values)
        return tuple(values)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "list", "set", "tuple"}
        and len(node.args) == 1
        and not node.keywords
    ):
        return _constant_strings(node.args[0], constants)
    return ()


def _module_constants(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    constants: dict[str, tuple[str, ...]] = {}
    for statement in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target = statement.targets[0]
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            target = statement.target
            value = statement.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        constant_values = _constant_strings(value, constants)
        if constant_values:
            constants[target.id] = constant_values
    return constants


def _is_module_not_found_error(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "ModuleNotFoundError") or (
        isinstance(node, ast.Attribute) and node.attr == "ModuleNotFoundError"
    )


def _is_negative_import_audit(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call) or not node.args:
        return False
    function = node.func
    function_name = (
        function.id
        if isinstance(function, ast.Name)
        else function.attr
        if isinstance(function, ast.Attribute)
        else None
    )
    return function_name in {"assertRaises", "assertRaisesRegex", "raises"} and (
        _is_module_not_found_error(node.args[0])
    )


class _PrivateTestImportVisitor(ast.NodeVisitor):
    def __init__(self, constants: dict[str, tuple[str, ...]]) -> None:
        self.constants = constants
        self.imported: set[str] = set()
        self.loop_constants: dict[str, tuple[str, ...]] = {}
        self.negative_import_audit_depth = 0

    def visit_Import(self, node: ast.Import) -> None:
        self.imported.update(
            alias.name
            for alias in node.names
            if _is_private_workbench_module(alias.name)
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module
        if module_name is None or not module_name.startswith(f"{PACKAGE_NAME}."):
            return
        if _is_private_workbench_module(module_name):
            self.imported.add(module_name)
        self.imported.update(
            f"{module_name}.{alias.name}"
            for alias in node.names
            if alias.name.startswith("_")
        )

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        is_patch = (isinstance(function, ast.Name) and function.id == "patch") or (
            isinstance(function, ast.Attribute) and function.attr == "patch"
        )
        if is_patch and node.args:
            target_argument = node.args[0]
            if (
                isinstance(target_argument, ast.Constant)
                and isinstance(target_argument.value, str)
                and _is_private_workbench_module(target_argument.value)
            ):
                self.imported.add(target_argument.value)

        is_dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (isinstance(function, ast.Attribute) and function.attr == "import_module")
        if is_dynamic_import and node.args and self.negative_import_audit_depth == 0:
            module_argument = node.args[0]
            candidates = (
                self.loop_constants.get(module_argument.id, ())
                if isinstance(module_argument, ast.Name)
                else _constant_strings(module_argument, self.constants)
            )
            self.imported.update(
                candidate
                for candidate in candidates
                if _is_private_workbench_module(candidate)
            )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for(node)

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        previous: tuple[str, ...] | None = None
        has_previous = False
        if isinstance(node.target, ast.Name):
            has_previous = node.target.id in self.loop_constants
            previous = self.loop_constants.get(node.target.id)
            values = _constant_strings(node.iter, self.constants)
            if values:
                self.loop_constants[node.target.id] = values
        self.visit(node.target)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        if isinstance(node.target, ast.Name):
            if has_previous:
                assert previous is not None
                self.loop_constants[node.target.id] = previous
            else:
                self.loop_constants.pop(node.target.id, None)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        is_negative_audit = any(
            _is_negative_import_audit(item.context_expr) for item in node.items
        )
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
        if is_negative_audit:
            self.negative_import_audit_depth += 1
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            if is_negative_audit:
                self.negative_import_audit_depth -= 1


def _private_test_imports_from_source(
    source: str,
    *,
    filename: str = "<unknown>",
) -> tuple[str, ...]:
    tree = ast.parse(source, filename=filename)
    visitor = _PrivateTestImportVisitor(_module_constants(tree))
    visitor.visit(tree)
    return tuple(sorted(visitor.imported))


def _private_test_imports(source_path: Path) -> tuple[str, ...]:
    return _private_test_imports_from_source(
        source_path.read_text(encoding="utf-8"),
        filename=source_path.as_posix(),
    )


def _test_owner(source_path: Path) -> str:
    relative = source_path.relative_to(API_ROOT / "tests")
    if len(relative.parts) >= 2 and relative.parts[0] == "unit":
        return relative.parts[1]
    return "<public-only>"


class PrivateTestImportDiscoveryTests(unittest.TestCase):
    def test_detects_private_imports_from_a_constant_collection_loop(self) -> None:
        source = """
import importlib

SMOKE_MODULES = (
    "emperor_workbench.filesystem",
    "emperor_workbench.training_jobs._status",
    "emperor_workbench.training_jobs._store",
)

for module_name in SMOKE_MODULES:
    importlib.import_module(module_name)
"""

        self.assertEqual(
            _private_test_imports_from_source(source),
            (
                "emperor_workbench.training_jobs._status",
                "emperor_workbench.training_jobs._store",
            ),
        )

    def test_ignores_private_targets_in_negative_path_audits(self) -> None:
        source = """
import importlib
import unittest

REMOVED_MODULES = (
    "emperor_workbench.schemas._inspection",
    "emperor_workbench.training_jobs._legacy",
)

class RemovedPathTests(unittest.TestCase):
    def test_removed_paths_stay_absent(self) -> None:
        for module_name in REMOVED_MODULES:
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(module_name)
"""

        self.assertEqual(_private_test_imports_from_source(source), ())


class WorkbenchPrivateOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()

    def test_every_implementation_module_has_one_owner(self) -> None:
        unowned = [
            source_module(source_path)
            for source_path in source_paths()
            if source_module(source_path) != PACKAGE_NAME
            and owner_for(source_module(source_path), self.manifest) is None
        ]
        self.assertEqual([], unowned)

    def test_cross_owner_private_import_ledger_is_exact(self) -> None:
        actual = tuple(
            violation.as_dict() for violation in private_imports(self.manifest)
        )
        expected = manifest_records(
            self.manifest,
            "legacy_cross_owner_private_imports",
        )
        self.assertEqual(expected, actual)

    def test_private_test_imports_are_limited_to_matching_unit_owner(
        self,
    ) -> None:
        test_root = API_ROOT / "tests"
        violations: list[PrivateTestImportViolation] = []
        for source_path in sorted(test_root.rglob("*.py")):
            if "__pycache__" in source_path.parts:
                continue
            test_owner = _test_owner(source_path)
            for imported_module in _private_test_imports(source_path):
                imported_owner = owner_for(imported_module, self.manifest)
                if imported_owner is None or imported_owner == test_owner:
                    continue
                violations.append(
                    PrivateTestImportViolation(
                        source=source_path.relative_to(test_root).as_posix(),
                        imported=imported_module,
                        test_owner=test_owner,
                        imported_owner=imported_owner,
                    )
                )

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
