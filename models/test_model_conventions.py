import ast
import os
import unittest
from importlib import import_module
from pathlib import Path

from emperor.experiments.base import PresetDefinition

os.environ.setdefault("MPLCONFIGDIR", "/tmp")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"


class TestModelConventions(unittest.TestCase):
    def test_non_init_modules_are_not_pure_reexports(self):
        def is_docstring(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )

        def is_reexport_statement(node: ast.AST) -> bool:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return True
            if isinstance(node, ast.Assign):
                return all(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
            if isinstance(node, ast.AnnAssign):
                return (
                    isinstance(node.target, ast.Name)
                    and node.target.id == "__all__"
                )
            return False

        pure_reexport_modules = []
        for path in sorted(MODELS_ROOT.glob("**/*.py")):
            if path.name == "__init__.py":
                continue
            source = path.read_text()
            tree = ast.parse(source)
            body = [node for node in tree.body if not is_docstring(node)]
            if body and all(is_reexport_statement(node) for node in body):
                pure_reexport_modules.append(path.relative_to(PROJECT_ROOT))

        self.assertEqual(pure_reexport_modules, [])

    def test_model_constructors_use_linear_config_naming(self):
        for path in sorted(MODELS_ROOT.glob("**/model.py")):
            source = path.read_text()
            tree = ast.parse(source)
            model_class = next(
                (
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "Model"
                ),
                None,
            )
            if model_class is None:
                continue

            init = next(
                (
                    node
                    for node in model_class.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                ),
                None,
            )
            self.assertIsNotNone(init, path)
            positional_args = init.args.args

            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertGreaterEqual(len(positional_args), 2)
                self.assertEqual(positional_args[0].arg, "self")
                self.assertEqual(positional_args[1].arg, "config")
                self.assertEqual(
                    ast.unparse(positional_args[1].annotation), "'ModelConfig'"
                )
                self.assertIn("super().__init__(config)", source)
                self.assertIn("self.experiment_config", source)
                self.assertNotIn("model_cfg", source)
                self.assertNotIn("exp_cfg", source)
                self.assertNotIn("main_cfg", source)
                self.assertNotIn('cfg: "ModelConfig"', source)

    def test_presets_keep_values_and_descriptions_together(self):
        for path in sorted(MODELS_ROOT.glob("**/presets.py")):
            module_name = ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
            module = import_module(module_name)
            preset_enum = module.ExperimentPreset
            presets = module.ExperimentPresets()
            preset_members = list(preset_enum)
            preset_definitions = getattr(module, "_PRESET_DEFINITIONS")

            with self.subTest(module=module_name):
                self.assertEqual(
                    [preset.value for preset in preset_members],
                    list(range(1, len(preset_members) + 1)),
                )
                self.assertEqual(set(preset_definitions), set(preset_members))
                for public_map_name in (
                    "PRESET_DEFINITIONS",
                    "PRESET_OVERRIDES",
                    "PRESET_DESCRIPTIONS",
                    "PRESET_LOCKS",
                ):
                    self.assertNotIn(public_map_name, module.ExperimentPresets.__dict__)

                for preset in preset_members:
                    definition = presets.definition_for_preset(preset)
                    self.assertEqual(definition, preset_definitions[preset])
                    self.assertIsInstance(definition, PresetDefinition)
                    self.assertIsInstance(definition.preset_values, dict)
                    self.assertIsInstance(definition.description, str)
                    self.assertTrue(definition.description.strip())
                    self.assertEqual(
                        presets.overrides_for_preset(preset),
                        definition.preset_values,
                    )
                    self.assertEqual(
                        presets.description_for_preset(preset),
                        definition.description,
                    )
                    self.assertIsInstance(presets.locks_for_preset(preset), dict)

    def test_presets_use_data_first_instance_interface(self):
        for path in sorted(MODELS_ROOT.glob("**/presets.py")):
            source = path.read_text()
            tree = ast.parse(source)

            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertNotIn("TypedDict", source)
                self.assertNotIn("_preset_overrides", source)
                self.assertNotIn("_preset_descriptions", source)
                self.assertNotIn("_preset_locks", source)

                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and node.name.startswith(
                        "_Preset"
                    ):
                        self.fail(f"descriptor class remains: {node.name}")
                    if (
                        isinstance(node, ast.ClassDef)
                        and node.name == "ExperimentPresets"
                    ):
                        for item in node.body:
                            if isinstance(item, (ast.Assign, ast.AnnAssign)):
                                targets = (
                                    item.targets
                                    if isinstance(item, ast.Assign)
                                    else [item.target]
                                )
                                names = [
                                    target.id
                                    for target in targets
                                    if isinstance(target, ast.Name)
                                ]
                                self.assertFalse(
                                    any(name.startswith("PRESET_") for name in names),
                                    names,
                                )
                            if not isinstance(item, ast.FunctionDef):
                                continue
                            is_static = any(
                                isinstance(decorator, ast.Name)
                                and decorator.id == "staticmethod"
                                for decorator in item.decorator_list
                            )
                            self.assertFalse(
                                is_static and item.name.endswith("_preset"),
                                item.name,
                            )


if __name__ == "__main__":
    unittest.main()
