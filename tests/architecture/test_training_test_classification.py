from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = PROJECT_ROOT / "tests"


@dataclass(frozen=True)
class _FunctionRecord:
    qualified_name: str
    class_name: str | None
    node: ast.FunctionDef | ast.AsyncFunctionDef


def _is_training_operation(
    node: ast.Call,
    *,
    include_run_execution: bool,
) -> bool:
    if (
        include_run_execution
        and isinstance(node.func, ast.Name)
        and node.func.id == "execute_runs"
    ):
        return True
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr == "fit":
        return True
    if node.func.attr != "step":
        return False
    return "optimizer" in ast.unparse(node.func.value).lower()


def _function_records(
    tree: ast.Module,
) -> tuple[dict[str, _FunctionRecord], set[str]]:
    records: dict[str, _FunctionRecord] = {}
    class_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            records[node.name] = _FunctionRecord(node.name, None, node)
            continue
        if not isinstance(node, ast.ClassDef):
            continue
        class_names.add(node.name)
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified_name = f"{node.name}.{member.name}"
                records[qualified_name] = _FunctionRecord(
                    qualified_name,
                    node.name,
                    member,
                )
    return records, class_names


def _local_references(
    record: _FunctionRecord,
    records: dict[str, _FunctionRecord],
    functions_by_class: dict[str, set[str]],
) -> set[str]:
    references: set[str] = set()
    for node in ast.walk(record.node):
        if isinstance(node, ast.Name):
            if node.id in records:
                references.add(node.id)
            references.update(functions_by_class.get(node.id, ()))
            continue
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"self", "cls"}
            and record.class_name is not None
        ):
            target = f"{record.class_name}.{node.attr}"
            if target in records:
                references.add(target)
    return references


def _training_tests(
    tree: ast.Module,
    *,
    include_run_execution: bool,
) -> tuple[list[_FunctionRecord], list[_FunctionRecord]]:
    records, class_names = _function_records(tree)
    functions_by_class = {
        class_name: {
            name for name, record in records.items() if record.class_name == class_name
        }
        for class_name in class_names
    }
    references = {
        name: _local_references(record, records, functions_by_class)
        for name, record in records.items()
    }
    training_functions = {
        name
        for name, record in records.items()
        if any(
            _is_training_operation(
                node,
                include_run_execution=include_run_execution,
            )
            for node in ast.walk(record.node)
            if isinstance(node, ast.Call)
        )
    }

    changed = True
    while changed:
        changed = False
        for name, targets in references.items():
            if name not in training_functions and targets & training_functions:
                training_functions.add(name)
                changed = True

    training_tests = [
        record
        for name, record in records.items()
        if name in training_functions and record.node.name.startswith("test")
    ]
    referenced_functions = set().union(*references.values(), set())
    unresolved_helpers = [
        record
        for name, record in records.items()
        if name in training_functions
        and not record.node.name.startswith("test")
        and name not in referenced_functions
    ]
    return training_tests, unresolved_helpers


def _has_training_marker(record: _FunctionRecord) -> bool:
    return any(
        ast.unparse(decorator).removesuffix("()") == "pytest.mark.training"
        for decorator in record.node.decorator_list
    )


def test_training_marker_is_registered() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    markers = project["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.partition(":")[0].strip() == "training" for marker in markers)


def test_training_operations_are_statically_classified() -> None:
    missing_markers: list[str] = []
    unresolved_helpers: list[str] = []
    for path in sorted(TEST_ROOT.rglob("*.py")):
        relative_path = path.relative_to(PROJECT_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        training_tests, unresolved = _training_tests(
            tree,
            include_run_execution=relative_path.parts[:2] == ("tests", "integration"),
        )
        missing_markers.extend(
            f"{relative_path}:{record.node.lineno}: {record.qualified_name}"
            for record in training_tests
            if not _has_training_marker(record)
        )
        unresolved_helpers.extend(
            f"{relative_path}:{record.node.lineno}: {record.qualified_name}"
            for record in unresolved
        )

    assert not unresolved_helpers, (
        "Training helpers are not statically reachable from a test:\n"
        + "\n".join(unresolved_helpers)
    )
    assert not missing_markers, (
        "Tests that fit models or step optimizers require @pytest.mark.training:\n"
        + "\n".join(missing_markers)
    )
