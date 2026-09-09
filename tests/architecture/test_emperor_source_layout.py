import ast
import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
EMPEROR_SOURCE = SOURCE_ROOT / "emperor"
LAYERS_SOURCE = EMPEROR_SOURCE / "layers"

_LAYER_CONFIG_CONSTRUCTOR_OWNERS = {
    ("_composition/gate/core.py", "LayerGate"),
    ("_composition/recurrent/base.py", "RecurrentCompositionAbstract"),
    (
        "_composition/recurrent/runtime/iteration_schedule.py",
        "RecurrentIterationSchedule",
    ),
    ("_composition/recurrent/variants/standard.py", "RecurrentLayer"),
    ("_composition/recurrent/variants/inner_thinking.py", "InnerThinkingRecurrent"),
    (
        "_composition/recurrent/variants/tiny_recursive_model.py",
        "TinyRecursiveModelRecurrent",
    ),
    (
        "_composition/recurrent/variants/hierarchical_reasoning_model.py",
        "HierarchicalReasoningModelRecurrent",
    ),
    ("_composition/residual/base.py", "ResidualConnectionAbstract"),
    ("_composition/residual/pairwise.py", "WeightedPairwiseResidualAbstract"),
    ("_composition/residual/variants/attention/core.py", "AttentionResidual"),
    ("_layer/core.py", "Layer"),
    ("_layer/pipeline/halting.py", "LayerHaltingDelegate"),
    ("_layer/pipeline/memory.py", "LayerMemoryDelegate"),
    ("_layer/pipeline/normalization.py", "LayerNormalizationDelegate"),
    ("_layer/pipeline/postprocessing.py", "LayerPostprocessingDelegate"),
    ("_layer/pipeline/residual.py", "LayerResidualDelegate"),
    ("_stack/builder.py", "LayerStackBuilder"),
    ("_stack/core.py", "LayerStack"),
    ("_stack/shared_controllers.py", "LayerStackSharedControllers"),
    ("_stack/topology.py", "LayerStackTopology"),
}

_LAYER_DELEGATE_INITIALIZER_OWNERS = {
    ("_composition/recurrent/base.py", "RecurrentCompositionAbstract"),
    ("_layer/core.py", "Layer"),
    ("_stack/core.py", "LayerStack"),
}


def module_name(path: Path) -> str:
    relative_path = path.relative_to(SOURCE_ROOT)
    module_parts = list(relative_path.parts)
    if module_parts[-1] == "__init__.py":
        module_parts = module_parts[:-1]
    else:
        module_parts[-1] = path.stem
    return ".".join(module_parts)


def emperor_modules() -> set[str]:
    return {module_name(path) for path in EMPEROR_SOURCE.rglob("*.py")}


def parsed_source_files():
    for path in SOURCE_ROOT.rglob("*.py"):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class EmperorSourceLayoutTests(unittest.TestCase):
    def test_layer_source_contains_no_config_initializer_helpers(self):
        helper_references = []
        for path in sorted(LAYERS_SOURCE.rglob("*.py")):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if re.search(r"initialize_?from_?config", line, re.IGNORECASE):
                    helper_references.append(
                        (path.relative_to(LAYERS_SOURCE).as_posix(), line_number, line)
                    )
        self.assertEqual(helper_references, [])

    def test_layer_config_consumers_initialize_in_constructor(self):
        owners: dict[tuple[str, str], ast.ClassDef] = {}
        for path in sorted(LAYERS_SOURCE.rglob("*.py")):
            relative_path = path.relative_to(LAYERS_SOURCE).as_posix()
            syntax_tree = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
            for class_definition in (
                node for node in ast.walk(syntax_tree) if isinstance(node, ast.ClassDef)
            ):
                owners[(relative_path, class_definition.name)] = class_definition

        self.assertTrue(_LAYER_CONFIG_CONSTRUCTOR_OWNERS.issubset(owners))
        for owner, class_definition in sorted(owners.items()):
            with self.subTest(owner=owner):
                methods = {
                    node.name: node
                    for node in class_definition.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                config_reads_by_method = {
                    method.name: [
                        node
                        for node in ast.walk(method)
                        if isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "self"
                        and node.value.attr == "cfg"
                    ]
                    for method in methods.values()
                }
                config_reads_outside_constructor = [
                    (method_name, node.lineno, ast.unparse(node))
                    for method_name, config_reads in config_reads_by_method.items()
                    if method_name != "__init__"
                    for node in config_reads
                ]
                self.assertEqual(config_reads_outside_constructor, [])
                config_reads = config_reads_by_method.get("__init__", [])
                if owner in _LAYER_CONFIG_CONSTRUCTOR_OWNERS:
                    self.assertTrue(config_reads)
                if not config_reads:
                    continue

                initializer = methods["__init__"]
                first_config_read_line = min(node.lineno for node in config_reads)
                last_config_read_line = max(node.end_lineno for node in config_reads)
                constructor_calls = [
                    node for node in ast.walk(initializer) if isinstance(node, ast.Call)
                ]
                validator_calls = [
                    node
                    for node in constructor_calls
                    if isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "self"
                    and node.func.value.attr == "VALIDATOR"
                    and node.func.attr.startswith("validate")
                ]
                for validator_call in validator_calls:
                    self.assertLess(validator_call.end_lineno, first_config_read_line)

                super_initializer_calls = [
                    node
                    for node in constructor_calls
                    if isinstance(node.func, ast.Attribute)
                    and node.func.attr == "__init__"
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == "super"
                ]
                for super_initializer_call in super_initializer_calls:
                    self.assertLess(
                        super_initializer_call.end_lineno, first_config_read_line
                    )

                if owner not in _LAYER_DELEGATE_INITIALIZER_OWNERS:
                    continue
                self.assertIn("__initialize_delegates", methods)
                delegate_initializer_calls = [
                    node
                    for node in constructor_calls
                    if isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr == "__initialize_delegates"
                ]
                self.assertEqual(len(delegate_initializer_calls), 1)
                self.assertLess(
                    last_config_read_line, delegate_initializer_calls[0].lineno
                )

    def test_emperor_source_contains_no_symlink_bridge(self):
        symlinks = [
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in EMPEROR_SOURCE.rglob("*")
            if path.is_symlink()
        ]

        self.assertEqual(symlinks, [])

    def test_production_imports_reference_physical_emperor_modules(self):
        available_modules = emperor_modules()
        missing_modules = []

        for path, syntax_tree in parsed_source_files():
            for node in ast.walk(syntax_tree):
                referenced_modules = []
                if isinstance(node, ast.Import):
                    referenced_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    referenced_modules.append(node.module)

                for referenced_module in referenced_modules:
                    if not (
                        referenced_module == "emperor"
                        or referenced_module.startswith("emperor.")
                    ):
                        continue
                    if referenced_module not in available_modules:
                        missing_modules.append(
                            (
                                path.relative_to(REPOSITORY_ROOT).as_posix(),
                                node.lineno,
                                referenced_module,
                            )
                        )

        self.assertEqual(missing_modules, [])

    def test_emperor_runtime_contains_no_import_redirection_hook(self):
        forbidden_tokens = (
            "sys.path",
            "sys.modules",
            "sys.meta_path",
            "spec_from_file_location",
            "SourceFileLoader",
            "MetaPathFinder",
            "PathFinder",
            "__path__",
        )
        matches = []

        for path in EMPEROR_SOURCE.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                if token in source:
                    matches.append(
                        (path.relative_to(REPOSITORY_ROOT).as_posix(), token)
                    )

        self.assertEqual(matches, [])

    def test_source_strings_do_not_preserve_retired_emperor_paths(self):
        available_modules = emperor_modules()
        retired_references = []
        module_reference = re.compile(r"^emperor(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
        legacy_path_reference = re.compile(r"(?<!src/)emperor/(?:$|[A-Za-z_])")

        for path, syntax_tree in parsed_source_files():
            for node in ast.walk(syntax_tree):
                if not isinstance(node, ast.Constant) or not isinstance(
                    node.value, str
                ):
                    continue

                value = node.value
                if legacy_path_reference.search(value):
                    retired_references.append(
                        (
                            path.relative_to(REPOSITORY_ROOT).as_posix(),
                            node.lineno,
                            value,
                        )
                    )
                    continue

                match = module_reference.match(value)
                if match is None:
                    continue
                module_parts = match.group(0).split(".")
                current_module_parts = [module_parts[0]]
                for part in module_parts[1:]:
                    if part[0].isupper():
                        break
                    current_module_parts.append(part)
                referenced_module = ".".join(current_module_parts)
                if referenced_module not in available_modules:
                    retired_references.append(
                        (
                            path.relative_to(REPOSITORY_ROOT).as_posix(),
                            node.lineno,
                            value,
                        )
                    )

        self.assertEqual(retired_references, [])

    def test_residual_composition_has_no_retired_dispatch_or_config_shape(self):
        retired_names = {
            "AttentionResidualOption",
            "PAIRWISE_RESIDUAL_TYPES",
            "RESIDUAL_OPTION_TYPES",
            "ResidualConnection",
            "ResidualConnectionOptions",
            "_PairwiseResidualParameters",
        }
        retired_residual_config_fields = {
            "attention_config",
            "model_config",
            "option",
        }
        violations = []

        for path, syntax_tree in parsed_source_files():
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            for node in ast.walk(syntax_tree):
                if isinstance(node, ast.Name) and node.id in retired_names:
                    violations.append((relative_path, node.lineno, node.id))
                    continue
                if isinstance(node, ast.Attribute):
                    if node.attr in retired_names or node.attr == "attention_residual":
                        violations.append((relative_path, node.lineno, node.attr))
                    continue
                if not isinstance(node, ast.Call):
                    continue
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else None
                )
                if function_name != "ResidualConfig":
                    continue
                for keyword in node.keywords:
                    if keyword.arg in retired_residual_config_fields:
                        violations.append(
                            (
                                relative_path,
                                node.lineno,
                                f"ResidualConfig({keyword.arg}=...)",
                            )
                        )

        self.assertEqual(violations, [])

    def test_residual_composition_does_not_own_recurrent_integration(self):
        residual_root = SOURCE_ROOT / "emperor" / "layers" / "_composition" / "residual"
        recurrent_lifecycle_names = {
            "advance_state",
            "fork_at_gradient_boundary",
            "fork_with_detached_generated_sources",
        }
        violations = []

        for path in residual_root.rglob("*.py"):
            syntax_tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            for node in ast.walk(syntax_tree):
                referenced_modules = []
                if isinstance(node, ast.Import):
                    referenced_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    referenced_modules.append(node.module)
                for referenced_module in referenced_modules:
                    if ".recurrent" in referenced_module:
                        violations.append(
                            (relative_path, node.lineno, referenced_module)
                        )

                identifier = (
                    node.name
                    if isinstance(
                        node,
                        (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                    else node.id
                    if isinstance(node, ast.Name)
                    else node.attr
                    if isinstance(node, ast.Attribute)
                    else None
                )
                if identifier is not None and (
                    "recurrent" in identifier.lower()
                    or identifier in recurrent_lifecycle_names
                ):
                    violations.append((relative_path, node.lineno, identifier))

        self.assertFalse((residual_root / "recurrent.py").exists())
        self.assertEqual(violations, [])

    def test_recurrent_composition_has_no_standalone_reasoning_boundary(self):
        retired_modules = (
            "emperor.layers._recurrent",
            "emperor.layers._validation.recurrent",
            "emperor.layers._composition.recurrent.variants.hrm",
            "emperor.layers._composition.recurrent.variants.trm",
            "emperor.reasoning",
            "models.reasoning",
        )
        retired_names = {
            "HRMRecurrent",
            "HRMRecurrentConfig",
            "ReasoningProcess",
            "STRUCTURED_REASONING",
            "TRMRecurrent",
            "TRMRecurrentConfig",
        }
        recurrent_root = (
            SOURCE_ROOT / "emperor" / "layers" / "_composition" / "recurrent"
        )
        violations = []

        for path, syntax_tree in parsed_source_files():
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            for node in ast.walk(syntax_tree):
                referenced_modules = []
                if isinstance(node, ast.Import):
                    referenced_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    referenced_modules.append(node.module)
                for referenced_module in referenced_modules:
                    if referenced_module.startswith(retired_modules):
                        violations.append(
                            (relative_path, node.lineno, referenced_module)
                        )
                    if path.is_relative_to(recurrent_root) and (
                        referenced_module == "models"
                        or referenced_module.startswith("models.")
                    ):
                        violations.append(
                            (relative_path, node.lineno, referenced_module)
                        )
                if isinstance(node, ast.Name) and node.id in retired_names:
                    violations.append((relative_path, node.lineno, node.id))
                elif isinstance(node, ast.Attribute) and node.attr in retired_names:
                    violations.append((relative_path, node.lineno, node.attr))

        self.assertFalse((SOURCE_ROOT / "emperor" / "reasoning").exists())
        self.assertFalse((SOURCE_ROOT / "models" / "reasoning").exists())
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
