from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import models

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


def test_model_package_configs_declare_each_public_constant_once() -> None:
    duplicates: list[str] = []
    for config_path in sorted(_MODELS_ROOT.glob("**/config.py")):
        module = ast.parse(config_path.read_text(), filename=str(config_path))
        for name, lines in _constant_assignments(module).items():
            if len(lines) > 1:
                relative_path = config_path.relative_to(_MODELS_ROOT)
                duplicates.append(f"{relative_path}:{name} at lines {lines}")

    assert not duplicates, "Duplicate public config declarations:\n" + "\n".join(
        duplicates
    )
