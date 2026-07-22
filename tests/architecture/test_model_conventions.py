import ast
import os
import unittest
from importlib import import_module
from pathlib import Path

from emperor.config import ModelConfig
from emperor.datasets.text.language_modeling import PennTreebank, WikiText2
from emperor.experiments import ExperimentTask
from model_runtime.packages import PresetDefinition
from models.catalog import MODEL_CATALOG

os.environ.setdefault("MPLCONFIGDIR", "/tmp")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
MODELS_ROOT = SOURCE_ROOT / "models"
MODEL_PACKAGE_TESTS_ROOT = PROJECT_ROOT / "tests" / "model_packages"

_STANDARD_PUBLIC_EXPORTS = ("MODEL_PACKAGE",)

_EXPECTED_CONFIG_BUILDERS = {
    "models.bert.linear": "BertLinearConfigBuilder",
    "models.bert.linear_adaptive": "BertLinearAdaptiveConfigBuilder",
    "models.bert.expert_linear": "BertExpertLinearConfigBuilder",
    "models.bert.expert_linear_adaptive": "BertExpertLinearAdaptiveConfigBuilder",
    "models.gpt.linear": "GptLinearConfigBuilder",
    "models.gpt.linear_adaptive": "GptLinearAdaptiveConfigBuilder",
    "models.gpt.expert_linear": "GptExpertLinearConfigBuilder",
    "models.gpt.expert_linear_adaptive": "GptExpertLinearAdaptiveConfigBuilder",
    "models.vit.linear": "VitLinearConfigBuilder",
    "models.vit.linear_adaptive": "VitLinearAdaptiveConfigBuilder",
    "models.vit.expert_linear": "VitExpertLinearConfigBuilder",
    "models.vit.expert_linear_adaptive": "VitExpertLinearAdaptiveConfigBuilder",
    "models.transformer.linear": "TransformerLinearConfigBuilder",
    "models.transformer.linear_adaptive": ("TransformerLinearAdaptiveConfigBuilder"),
    "models.transformer.expert_linear": "TransformerExpertLinearConfigBuilder",
    "models.transformer.expert_linear_adaptive": (
        "TransformerExpertLinearAdaptiveConfigBuilder"
    ),
    "models.linears.linear": "LinearConfigBuilder",
    "models.linears.linear_adaptive": "LinearAdaptiveConfigBuilder",
    "models.experts.linear": "LinearConfigBuilder",
    "models.experts.linear_adaptive": "LinearAdaptiveConfigBuilder",
    "models.parametric.parametric_vector": "ParametricVectorConfigBuilder",
    "models.parametric.parametric_matrix": "ParametricMatrixConfigBuilder",
    "models.parametric.parametric_generator": "ParametricGeneratorConfigBuilder",
    "models.neuron.linear": "NeuronLinearConfigBuilder",
    "models.neuron.linear_adaptive": "NeuronLinearAdaptiveConfigBuilder",
    "models.neuron.expert_linear": "NeuronExpertLinearConfigBuilder",
    "models.neuron.expert_linear_adaptive": "NeuronExpertLinearAdaptiveConfigBuilder",
}


def _package_module(package) -> str:
    return f"models.{package.identity.model_type}.{package.identity.model}"


class TestModelConventions(unittest.TestCase):
    def test_gpt_presets_search_axes_and_monitors_match_applicable_bert_metadata(self):
        for backend in (
            "linear",
            "linear_adaptive",
            "expert_linear",
            "expert_linear_adaptive",
        ):
            bert_presets_module = import_module(f"models.bert.{backend}.presets")
            gpt_presets_module = import_module(f"models.gpt.{backend}.presets")
            bert_monitors_module = import_module(
                f"models.bert.{backend}.monitor_options"
            )
            gpt_monitors_module = import_module(f"models.gpt.{backend}.monitor_options")
            gpt_datasets_module = import_module(f"models.gpt.{backend}.dataset_options")
            bert_search_module = import_module(f"models.bert.{backend}.search_space")
            gpt_search_module = import_module(f"models.gpt.{backend}.search_space")

            with self.subTest(backend=backend):
                self.assertEqual(
                    [preset.name for preset in gpt_presets_module.ExperimentPreset],
                    [
                        preset.name
                        for preset in bert_presets_module.ExperimentPreset
                        if preset.name != "CAUSAL"
                    ],
                )
                self.assertEqual(
                    {
                        preset.name: definition.preset_values
                        for preset, definition in vars(bert_presets_module)[
                            "_PRESET_DEFINITIONS"
                        ].items()
                        if preset.name != "CAUSAL"
                    },
                    {
                        preset.name: definition.preset_values
                        for preset, definition in vars(gpt_presets_module)[
                            "_PRESET_DEFINITIONS"
                        ].items()
                    },
                )
                self.assertEqual(
                    [option.name for option in gpt_monitors_module.MONITOR_OPTIONS],
                    [option.name for option in bert_monitors_module.MONITOR_OPTIONS],
                )
                self.assertIs(
                    gpt_datasets_module.DEFAULT_EXPERIMENT_TASK,
                    ExperimentTask.CAUSAL_LANGUAGE_MODELING,
                )
                self.assertEqual(
                    gpt_datasets_module.DATASET_OPTIONS_BY_TASK,
                    {
                        ExperimentTask.CAUSAL_LANGUAGE_MODELING: [
                            WikiText2,
                            PennTreebank,
                        ]
                    },
                )
                self.assertEqual(
                    {
                        key: value
                        for key, value in vars(gpt_search_module).items()
                        if key.startswith("SEARCH_SPACE_") and isinstance(value, list)
                    },
                    {
                        key: value
                        for key, value in vars(bert_search_module).items()
                        if key.startswith("SEARCH_SPACE_") and isinstance(value, list)
                    },
                )

    def test_catalog_package_public_exports_are_stable(self):
        for entry in MODEL_CATALOG.values():
            module_name = _package_module(entry)
            package = import_module(module_name)

            with self.subTest(package=entry.catalog_key):
                self.assertEqual(tuple(package.__all__), _STANDARD_PUBLIC_EXPORTS)
                for name in _STANDARD_PUBLIC_EXPORTS:
                    self.assertTrue(hasattr(package, name), name)
                self.assertIs(package.MODEL_PACKAGE, entry)

    def test_catalog_packages_keep_direct_builder_imports(self):
        self.assertEqual(
            set(_EXPECTED_CONFIG_BUILDERS),
            {_package_module(entry) for entry in MODEL_CATALOG.values()},
        )
        for module_path, builder_name in _EXPECTED_CONFIG_BUILDERS.items():
            builder_module = import_module(f"{module_path}.config_builder")

            with self.subTest(package=module_path):
                builder = getattr(builder_module, builder_name)
                self.assertEqual(builder.__module__, builder_module.__name__)

    def test_catalog_entrypoints_are_identity_adapters_for_shared_runner(self):
        for entry in MODEL_CATALOG.values():
            package_module = _package_module(entry)
            package_root = SOURCE_ROOT.joinpath(*package_module.split("."))
            source = (package_root / "__main__.py").read_text()

            with self.subTest(package=entry.catalog_key):
                self.assertIn("run_model_package_cli", source)
                self.assertIn(
                    f'MODEL_PACKAGE_KEY = "{entry.catalog_key}"',
                    source,
                )
                self.assertNotIn(".train_model(", source)

    def test_runtime_option_types_belong_to_their_runtime_options_modules(self):
        for entry in MODEL_CATALOG.values():
            package = _package_module(entry)
            module_name = f"{package}.runtime_options"
            module = import_module(module_name)
            path = SOURCE_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
            tree = ast.parse(path.read_text(), filename=str(path))
            defined_names = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            ]

            for name in defined_names:
                with self.subTest(module=module_name, name=name):
                    self.assertEqual(getattr(module, name).__module__, module_name)

    def test_all_catalog_presets_build_model_configs(self):
        for entry in MODEL_CATALOG.values():
            for preset in entry.preset_type:
                with self.subTest(package=entry.catalog_key, preset=preset.name):
                    config = entry.build_configuration(preset)
                    self.assertIsInstance(config, ModelConfig)

    def test_catalog_packages_have_one_external_test_module(self):
        for entry in MODEL_CATALOG.values():
            package_module = _package_module(entry)
            package_root = SOURCE_ROOT.joinpath(*package_module.split("."))
            relative_package = package_module.removeprefix("models.").split(".")
            external_test_root = MODEL_PACKAGE_TESTS_ROOT.joinpath(*relative_package)

            with self.subTest(package=entry.catalog_key):
                self.assertEqual(
                    sorted(path.name for path in package_root.glob("test*.py")),
                    [],
                )
                self.assertEqual(
                    sorted(path.name for path in external_test_root.glob("test*.py")),
                    ["test_model.py"],
                )

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
                return isinstance(node.target, ast.Name) and node.target.id == "__all__"
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
                self.assertIsNotNone(positional_args[1].annotation)
                self.assertEqual(
                    ast.unparse(positional_args[1].annotation).strip("'\""),
                    "ModelConfig",
                )
                self.assertIn("super().__init__(config)", source)
                self.assertIn("self.experiment_config", source)
                self.assertNotIn("model_cfg", source)
                self.assertNotIn("exp_cfg", source)
                self.assertNotIn("main_cfg", source)
                self.assertNotIn('cfg: "ModelConfig"', source)

    def test_presets_keep_values_and_descriptions_together(self):
        for path in sorted(MODELS_ROOT.glob("**/presets.py")):
            module_name = ".".join(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
            module = import_module(module_name)
            preset_enum = module.ExperimentPreset
            presets = module.ExperimentPresets()
            preset_members = list(preset_enum)
            preset_definitions = module._PRESET_DEFINITIONS

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
