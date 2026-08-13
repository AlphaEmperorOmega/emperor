from __future__ import annotations

import ast
import inspect
import unittest
from collections import defaultdict
from pathlib import Path
from types import NoneType
from typing import Any, get_args, get_origin, get_type_hints

import models
from models.catalog import discover_model_packages

_MODELS_ROOT = Path(models.__file__).parent


def _constant_assignments(module: ast.Module) -> dict[str, list[int]]:
    assignments: defaultdict[str, list[int]] = defaultdict(list)
    for statement in module.body:
        if isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
        elif isinstance(statement, ast.Assign):
            targets = tuple(statement.targets)
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                assignments[target.id].append(statement.lineno)
    return dict(assignments)


def _declared_class_bases(annotation: object) -> tuple[object, ...]:
    if get_origin(annotation) is type:
        return get_args(annotation)
    return tuple(
        base
        for option in get_args(annotation)
        for base in _declared_class_bases(option)
    )


def _duplicate_public_config_declarations() -> list[str]:
    duplicates: list[str] = []
    for config_path in sorted(_MODELS_ROOT.glob("**/config.py")):
        module = ast.parse(config_path.read_text(), filename=str(config_path))
        for name, lines in _constant_assignments(module).items():
            if len(lines) > 1:
                relative_path = config_path.relative_to(_MODELS_ROOT)
                duplicates.append(f"{relative_path}:{name} at lines {lines}")

    return duplicates


def _runtime_default_declaration_errors() -> tuple[list[str], list[str], list[str]]:
    missing: list[str] = []
    incompatible_none: list[str] = []
    unbounded_classes: list[str] = []
    for package in discover_model_packages():
        spec = package.runtime_defaults_spec
        annotations = get_type_hints(spec._config_module)
        for key in spec.supported_keys:
            value = spec.current_value(key)
            if value is not None and not inspect.isclass(value):
                continue
            annotation = annotations.get(key)
            if annotation is None:
                missing.append(f"{package.catalog_key}:{key}")
                continue
            if value is None and annotation not in (Any, object):
                if annotation is not NoneType and NoneType not in get_args(annotation):
                    incompatible_none.append(f"{package.catalog_key}:{key}")
            if inspect.isclass(value):
                options = _declared_class_bases(annotation)
                if not options or not any(
                    inspect.isclass(base) and issubclass(value, base)
                    for base in options
                ):
                    unbounded_classes.append(f"{package.catalog_key}:{key}")

    return missing, incompatible_none, unbounded_classes


class ModelPackageConfigDeclarationTests(unittest.TestCase):
    def test_model_package_configs_declare_each_public_constant_once(self) -> None:
        duplicates = _duplicate_public_config_declarations()

        self.assertEqual(
            duplicates,
            [],
            "Duplicate public config declarations:\n" + "\n".join(duplicates),
        )

    def test_nullable_and_class_defaults_have_complete_declarations(self) -> None:
        missing, incompatible_none, unbounded_classes = (
            _runtime_default_declaration_errors()
        )

        self.assertEqual(
            missing,
            [],
            "Runtime Defaults missing type declarations:\n" + "\n".join(missing),
        )
        self.assertEqual(
            incompatible_none,
            [],
            "Runtime Defaults declared non-nullable with None defaults:\n"
            + "\n".join(incompatible_none),
        )
        self.assertEqual(
            unbounded_classes,
            [],
            "Runtime Default class declarations missing an accepted base:\n"
            + "\n".join(unbounded_classes),
        )


if __name__ == "__main__":
    unittest.main()
