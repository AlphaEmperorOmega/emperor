from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from importlib.util import resolve_name
from pathlib import Path

from models.catalog import MODEL_CATALOG

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_OPERATIONAL_MODEL_MODULES = frozenset(
    {
        "models.catalog",
        "models.config_ast_listing",
        "models.config_overrides",
        "models.config_value_parser",
        "models.dataset_naming",
        "models.experiment_cli_parser",
        "models.experiment_mode",
        "models.log_migration",
        "models.model_metadata",
        "models.package_cli",
        "models.parser",
        "models.training_test_utils",
    }
)

_REMOVED_MODEL_MODULES = (
    "models.model_inspector",
    "models.adaptive_parameter_config_factory",
    "models.trainer_config",
    "models.transformer._builder_options",
    "models.linears._builder_adapter",
    "models.linears._builder_options",
    "models.linears._controller_stack",
    "models.linears._gate_config_factory",
    "models.linears._halting_config_factory",
    "models.linears._memory_config_factory",
    "models.linears._recurrent_config_factory",
    "models.linears.linear._boundary_model_config_factory",
    "models.linears.linear_adaptive._adaptive_generator_stack_config_factory",
    "models.linears.linear_adaptive._boundary_model_config_factory",
    "models.linears.linear_adaptive._builder_options",
    "models.experts._builder_adapter",
    "models.experts._builder_options",
    "models.experts._controller_stack",
    "models.experts._gate_config_factory",
    "models.experts._halting_config_factory",
    "models.experts._memory_config_factory",
    "models.experts._recurrent_config_factory",
    "models.experts.linear._boundary_config_factory",
    "models.experts.linear._control_config_factory",
    "models.bert._base_config_builder",
    "models.bert._builder_adapter",
    "models.bert._builder_options",
    "models.bert._config_defaults",
    "models.bert._controller_stack",
    "models.bert._gate_config_factory",
    "models.bert._halting_config_factory",
    "models.bert._memory_config_factory",
    "models.bert._recurrent_config_factory",
    "models.bert._tokenwise",
    "models.bert.linear_adaptive.runtime_defaults",
    "models.bert.linear_adaptive._adaptive_builder_options",
    "models.bert.linear_adaptive._builder_options",
    "models.bert.linear_adaptive._controller_stack",
    "models.bert.linear_adaptive._linear_builder_options",
    "models.bert.linear_adaptive._transformer_builder_options",
    "models.bert.expert_linear._base_config_builder",
    "models.bert.expert_linear.runtime_defaults",
    "models.bert.expert_linear._control_support",
    "models.bert.expert_linear._expert_adapter_support",
    "models.bert.expert_linear._tokenwise",
    "models.vit._boundary_model_config_factory",
    "models.vit._builder_adapter",
    "models.vit._builder_support",
    "models.vit._config_defaults",
    "models.vit._control_factory_dependencies",
    "models.vit._core_config_factory",
    "models.vit._expert_config_factory",
    "models.vit._gate_config_factory",
    "models.vit._halting_config_factory",
    "models.vit._linear_layer_config_factory",
    "models.vit._memory_config_factory",
    "models.vit._patch_config_factory",
    "models.vit._positional_embedding_config_factory",
    "models.vit._recurrent_config_factory",
    "models.vit.linear._builder_options",
    "models.vit.linear._controller_stack",
    "models.vit.linear._transformer_options",
    "models.vit.linear_adaptive._adaptive_builder_options",
    "models.vit.linear_adaptive._builder_options",
    "models.vit.linear_adaptive._controller_stack",
    "models.vit.linear_adaptive._transformer_options",
    "models.vit.expert_linear._builder_options",
    "models.vit.expert_linear._controller_stack",
    "models.vit.expert_linear._expert_builder_options",
    "models.vit.expert_linear._transformer_options",
    "models.vit.expert_linear_adaptive._adaptive_builder_options",
    "models.vit.expert_linear_adaptive._builder_options",
    "models.vit.expert_linear_adaptive._controller_stack",
    "models.vit.expert_linear_adaptive._expert_builder_options",
    "models.vit.expert_linear_adaptive._transformer_options",
    "models.neuron._builder_options",
    "models.neuron._config_builder",
    "models.neuron._control_config_factory",
    "models.neuron._controller_stack",
    "models.neuron._controller_stack_config_factory",
    "models.neuron._presets",
    "models.neuron._source_adapter",
    "models.neuron._test_cases",
    "models.neuron.experiment_config",
    "models.parametric._shared_stack_factory",
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    current_module = _module_name(path)
    package = (
        current_module
        if path.name == "__init__.py"
        else current_module.rsplit(".", 1)[0]
    )
    imported: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend((node.lineno, alias.name) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            relative = "." * node.level + (node.module or "")
            imported.append((node.lineno, resolve_name(relative, package)))
        elif node.module:
            imported.append((node.lineno, node.module))
    return imported


def _is_allowed(package: str, imported: str) -> bool:
    if imported == package or imported.startswith(package + "."):
        return True
    if imported == "emperor" or imported.startswith("emperor."):
        return True
    return any(
        imported == allowed or imported.startswith(allowed + ".")
        for allowed in _OPERATIONAL_MODEL_MODULES
    )


class TestModelPackageIsolation(unittest.TestCase):
    def test_catalog_packages_have_no_construction_imports_outside_themselves(self):
        violations = []
        for entry in MODEL_CATALOG.values():
            package = entry.module_path
            package_root = PROJECT_ROOT.joinpath(*package.split("."))
            for path in sorted(package_root.rglob("*.py")):
                for line, imported in _imported_modules(path):
                    if not imported.startswith("models."):
                        continue
                    if _is_allowed(package, imported):
                        continue
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{line}: {imported}"
                    )
        self.assertEqual(violations, [], "\n" + "\n".join(violations))

    def test_importing_one_catalog_package_does_not_load_another(self):
        packages = sorted({entry.module_path for entry in MODEL_CATALOG.values()})
        script = """
import importlib
import json
import sys

package = sys.argv[1]
catalog_packages = json.loads(sys.argv[2])
importlib.import_module(package)
loaded = sorted(
    other
    for other in catalog_packages
    if other != package
    and any(name == other or name.startswith(other + '.') for name in sys.modules)
)
if loaded:
    raise SystemExit(f'{package} loaded sibling Model Packages: {loaded}')
"""
        environment = {
            **os.environ,
            "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "matplotlib"),
        }
        for package in packages:
            with self.subTest(package=package):
                result = subprocess.run(
                    [sys.executable, "-c", script, package, json.dumps(packages)],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    result.stdout + result.stderr,
                )

    def test_obsolete_model_modules_are_removed(self):
        remaining = []
        for module in _REMOVED_MODEL_MODULES:
            path = PROJECT_ROOT.joinpath(*module.split(".")).with_suffix(".py")
            if path.exists():
                remaining.append(module)
        self.assertEqual(remaining, [])

        script = """
import importlib
import json
import sys

unexpected = []
for module in json.loads(sys.argv[1]):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            unexpected.append(f'{module} failed through missing {exc.name}')
    except Exception as exc:
        unexpected.append(f'{module} raised {type(exc).__name__}: {exc}')
    else:
        unexpected.append(f'{module} remains importable')
if unexpected:
    raise SystemExit('\\n'.join(unexpected))
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                json.dumps(_REMOVED_MODEL_MODULES),
            ],
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "matplotlib"),
            },
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
