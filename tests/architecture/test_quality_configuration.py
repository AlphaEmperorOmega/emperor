from __future__ import annotations

import ast
import configparser
import json
import os
import tomllib
import unittest
from pathlib import Path

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get(
        "MODEL_RUNTIME_QUALITY_SOURCE_ROOT",
        str(DEFAULT_PROJECT_ROOT),
    )
).resolve()
PYRIGHT_BASELINE = {
    "src/models/catalog.py",
    "src/model_runtime",
    "src/models/linears/linear/_residual.py",
    "src/models/linears/linear/runtime_defaults.py",
    "src/models/experts/linear_adaptive/_config_implementation.py",
    "src/models/experts/linear_adaptive/_residual.py",
    "src/models/experts/linear_adaptive/runtime_defaults.py",
    "src/models/neuron/expert_linear_adaptive/_hidden/_config_implementation.py",
    "src/models/neuron/expert_linear_adaptive/_hidden/runtime_defaults.py",
    "src/models/neuron/expert_linear_adaptive/_residual.py",
}
REQUIRED_GENERATED_EXCLUDES = {
    "**/__pycache__",
    ".next",
    ".runtime",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "node_modules",
    "torchenv",
    "apps",
}
REQUIRED_RUNTIME_MUTATION_SELECTORS = {
    "*planning.x_plan_runs__mutmut_4",
    "*planning.x_accept_run_plan__mutmut_19",
    "*_search_parsing.x__requires_custom_value_authorization__mutmut_7",
    "*preflight.xǁ_InspectionPreflightǁ_parameter_estimate__mutmut_70",
    "*preflight.xǁ_InspectionPreflightǁ_maximum_parameter_estimate__mutmut_6",
    "*execution.x__monitor_callbacks__mutmut_1",
    "*execution.xǁ_RunExecutorǁ_callback_groups__mutmut_13",
    "*execution.x__validate_training_run_handoff__mutmut_16",
    "*experiment.xǁExperimentBaseǁ_emit_training_error_preserving_primary__mutmut_5",
    "*_wire_shared.x__require_json_container_depth__mutmut_1",
    "*_graph_accounting.xǁ_ParameterAccumulatorǁ_reserve_memberships__mutmut_4",
    "*_graph_accounting.xǁ_ParameterAccumulatorǁ_reserve_registration__mutmut_4",
    "*model_graph.x_inspect_model_graph__mutmut_1",
    "*_tensor_walk.xǁ_TensorWalkerǁ_reserve_visit__mutmut_1",
    "*shape_trace.xǁ_ModuleCallRecorderǁinstall__mutmut_1",
    "*shape_trace.x__variable_tracer__mutmut_1",
    "*_shape_runtime.xǁShapeTraceRuntimeǁrestore__mutmut_1",
    "*_shape_runtime.xǁShapeTraceRuntimeǁrestore__mutmut_2",
    "*capture_limits.x__scalar_output_bytes__mutmut_17",
    "*inspection_limits.x__validate_positive_integer_limits__mutmut_1",
    "*inspection_limits.x__frozen_limits__mutmut_31",
    "*artifacts.x__validate_path_segment__mutmut_1",
    "*artifacts.x__model_id__mutmut_6",
    "*artifacts.x__resolved_contained_path__mutmut_3",
    "*artifacts.x__best_results_path__mutmut_6",
    "*artifacts.x__result_path__mutmut_12",
    "*artifacts.x__best_results_lock_path__mutmut_7",
}


class QualityConfigurationTests(unittest.TestCase):
    def test_runtime_value_validation_uses_python_311_typing_surface(self) -> None:
        for relative_path in (
            "src/model_runtime/packages/runtime_values.py",
            "tests/contract/test_runtime_default_value_validation.py",
        ):
            tree = ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
            typing_imports = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "typing"
                for alias in node.names
            }
            with self.subTest(path=relative_path):
                self.assertTrue(
                    typing_imports.isdisjoint({"TypeAliasType", "is_protocol"})
                )

    def test_strict_type_baseline_cannot_silently_shrink(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "pyrightconfig.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config["typeCheckingMode"], "strict")
        self.assertEqual(config["pythonVersion"], "3.13")
        self.assertTrue(PYRIGHT_BASELINE.issubset(config["include"]))
        self.assertTrue(REQUIRED_GENERATED_EXCLUDES.issubset(config["exclude"]))
        self.assertNotIn("ignore", config)
        self.assertNotIn("diagnosticSeverityOverrides", config)
        self.assertEqual(config["venvPath"], ".")
        self.assertEqual(config["venv"], "torchenv")

        runtime_option_modules = {
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / "src" / "models").rglob("runtime_options.py")
        }
        self.assertTrue(runtime_option_modules)
        self.assertTrue(runtime_option_modules.issubset(config["include"]))

        included_sources: set[Path] = set()
        for relative_path in config["include"]:
            included_path = PROJECT_ROOT / relative_path
            paths = (
                sorted(included_path.rglob("*.py"))
                if included_path.is_dir()
                else [included_path]
            )
            for path in paths:
                included_sources.add(path.resolve())
                source = path.read_text(encoding="utf-8")
                with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                    self.assertNotIn("type: ignore", source)
                    self.assertNotIn("pyright: ignore", source)

        runtime_sources = {
            path.resolve()
            for path in (PROJECT_ROOT / "src/model_runtime").rglob("*.py")
        }
        self.assertTrue(runtime_sources.issubset(included_sources))

    def test_model_runtime_has_independent_quality_gates(self) -> None:
        with (
            (PROJECT_ROOT / "tests/architecture/model_runtime_quality.toml").open(
                "rb"
            ) as manifest_file,
            (PROJECT_ROOT / "mise.toml").open("rb") as mise_file,
        ):
            manifest = tomllib.load(manifest_file)
            mise = tomllib.load(mise_file)

        self.assertEqual(manifest["schema_version"], 1)
        structure = manifest["structure"]
        self.assertEqual(structure["source_root"], "src/model_runtime")
        self.assertEqual(
            structure["ruff_rules"],
            ["C901", "PLR0911", "PLR0912", "PLR0913", "PLR0915"],
        )
        self.assertEqual(
            {
                key: structure[key]
                for key in (
                    "maximum_complexity",
                    "maximum_branches",
                    "maximum_returns",
                    "maximum_statements",
                    "maximum_arguments",
                    "maximum_function_lines",
                    "module_review_threshold",
                )
            },
            {
                "maximum_complexity": 10,
                "maximum_branches": 12,
                "maximum_returns": 6,
                "maximum_statements": 50,
                "maximum_arguments": 5,
                "maximum_function_lines": 60,
                "module_review_threshold": 600,
            },
        )
        self.assertEqual(
            {
                (item["path"], item["code"], item["function"])
                for item in structure["allowed_findings"]
            },
            {
                (
                    "src/model_runtime/runs/execution.py",
                    "PLR0913",
                    "execute_runs",
                ),
                (
                    "src/model_runtime/runs/experiment.py",
                    "PLR0913",
                    "ExperimentBase.execute_training_run",
                ),
            },
        )
        expected_reviewed_modules = {
            "src/model_runtime/inspection/shape_trace.py": 717,
            "src/model_runtime/runs/experiment.py": 602,
        }
        self.assertEqual(
            {
                item["path"]: item["maximum_lines"]
                for item in structure["reviewed_modules"]
            },
            expected_reviewed_modules,
        )
        for item in [
            *structure["allowed_findings"],
            *structure["reviewed_modules"],
        ]:
            with self.subTest(structural_exception=item["path"]):
                self.assertTrue(item["rationale"].strip())
                self.assertTrue((PROJECT_ROOT / item["path"]).is_file())
        for relative_path, maximum_lines in expected_reviewed_modules.items():
            line_count = len(
                (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
            )
            with self.subTest(reviewed_module=relative_path):
                self.assertGreater(line_count, structure["module_review_threshold"])
                self.assertLessEqual(line_count, maximum_lines)

        coverage = manifest["coverage"]
        self.assertEqual(coverage["source"], "model_runtime")
        self.assertGreaterEqual(coverage["minimum_statement_percent"], 88.0)
        self.assertGreaterEqual(coverage["minimum_branch_percent"], 75.0)
        self.assertIn(
            "tests.architecture.test_quality_configuration",
            coverage["tests"],
        )
        for test_module in coverage["tests"]:
            test_path = PROJECT_ROOT / Path(*test_module.split(".")).with_suffix(".py")
            with self.subTest(coverage_test=test_module):
                self.assertTrue(test_path.is_file())

        mutation = manifest["mutation"]
        self.assertEqual(mutation["source_root"], "src/model_runtime/")
        self.assertEqual(
            set(mutation["critical_sources"]),
            {
                "src/model_runtime/runs/planning.py",
                "src/model_runtime/runs/_search_parsing.py",
                "src/model_runtime/inspection/preflight.py",
                "src/model_runtime/runs/execution.py",
                "src/model_runtime/runs/experiment.py",
                "src/model_runtime/cli/_wire_shared.py",
                "src/model_runtime/inspection/_graph_accounting.py",
                "src/model_runtime/inspection/model_graph.py",
                "src/model_runtime/inspection/_shape_runtime.py",
                "src/model_runtime/inspection/_tensor_walk.py",
                "src/model_runtime/inspection/shape_trace.py",
                "src/model_runtime/inspection/capture_limits.py",
                "src/model_runtime/packages/inspection_limits.py",
                "src/model_runtime/runs/artifacts.py",
            },
        )
        self.assertEqual(
            set(mutation["selectors"]),
            REQUIRED_RUNTIME_MUTATION_SELECTORS,
        )
        self.assertIn(
            "tests/architecture/test_quality_configuration.py",
            mutation["tests"],
        )
        for relative_path in [*mutation["critical_sources"], *mutation["tests"]]:
            with self.subTest(mutation_path=relative_path):
                self.assertTrue((PROJECT_ROOT / relative_path).is_file())

        expected_tasks = {
            "test:model-runtime": (
                "python tools/emperor_dev.py python -- "
                "tools/model_runtime_quality.py tests"
            ),
            "test:model-runtime-coverage": (
                "python tools/emperor_dev.py python -- "
                "tools/model_runtime_quality.py coverage"
            ),
            "test:model-runtime-mutation": (
                "python tools/emperor_dev.py python -- "
                "tools/model_runtime_quality.py mutation"
            ),
            "test:model-runtime-structure": (
                "python tools/emperor_dev.py python -- "
                "tools/model_runtime_quality.py structure"
            ),
            "test:model-runtime-types": "npm run typecheck:model-runtime",
        }
        for task_name, command in expected_tasks.items():
            with self.subTest(task=task_name):
                self.assertTrue(mise["tasks"][task_name]["raw"])
                self.assertEqual(mise["tasks"][task_name]["run"], command)
        self.assertTrue((PROJECT_ROOT / "tools/model_runtime_quality.py").is_file())
        self.assertTrue(
            (PROJECT_ROOT / "tests/architecture/model_runtime_coveragerc").is_file()
        )
        coverage_config = configparser.ConfigParser()
        coverage_config.read(
            PROJECT_ROOT / "tests/architecture/model_runtime_coveragerc",
            encoding="utf-8",
        )
        self.assertTrue(coverage_config.getboolean("run", "branch"))
        self.assertEqual(coverage_config.get("run", "source"), "model_runtime")
        self.assertFalse(coverage_config.has_option("run", "omit"))
        self.assertFalse(coverage_config.has_option("report", "omit"))
        self.assertEqual(
            {
                line.strip()
                for line in coverage_config.get(
                    "report",
                    "exclude_lines",
                ).splitlines()
                if line.strip()
            },
            {"if TYPE_CHECKING:", "pragma: no cover"},
        )

    def test_model_runtime_uses_locked_canonical_type_checker(self) -> None:
        package = json.loads(
            (PROJECT_ROOT / "package.json").read_text(encoding="utf-8")
        )
        package_lock = json.loads(
            (PROJECT_ROOT / "package-lock.json").read_text(encoding="utf-8")
        )

        self.assertEqual(package["devDependencies"], {"pyright": "1.1.411"})
        self.assertEqual(
            package["scripts"]["typecheck:model-runtime"],
            "pyright --project pyrightconfig.json src/model_runtime",
        )
        self.assertEqual(
            package_lock["packages"]["node_modules/pyright"]["version"],
            "1.1.411",
        )

    def test_model_package_strict_type_scope_cannot_silently_shrink(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "pyright-model-packages.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            set(config),
            {
                "include",
                "pythonPlatform",
                "pythonVersion",
                "extraPaths",
                "venvPath",
                "venv",
                "typeCheckingMode",
            },
        )
        self.assertEqual(
            config["include"],
            [
                "src/models/catalog.py",
                "src/models/**/runtime_options.py",
            ],
        )
        self.assertEqual(config["typeCheckingMode"], "strict")
        self.assertEqual(config["pythonPlatform"], "Linux")
        self.assertEqual(config["pythonVersion"], "3.13")
        self.assertEqual(config["extraPaths"], ["src"])
        self.assertEqual(config["venvPath"], ".")
        self.assertEqual(config["venv"], "torchenv")

        runtime_option_modules = {
            path.resolve()
            for path in (PROJECT_ROOT / "src/models").glob("**/runtime_options.py")
        }
        certified_sources = {
            (PROJECT_ROOT / "src/models/catalog.py").resolve(),
            *runtime_option_modules,
        }
        self.assertGreaterEqual(len(certified_sources), 36)
        self.assertEqual(
            runtime_option_modules,
            {
                path.resolve()
                for path in (PROJECT_ROOT / "src/models").rglob("runtime_options.py")
            },
        )
        for path in sorted(certified_sources):
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertTrue(path.is_relative_to(PROJECT_ROOT))
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("type: ignore", source)
                self.assertNotIn("pyright: ignore", source)
                self.assertNotIn("# pyright:", source)

    def test_model_packages_use_locked_canonical_type_checker(self) -> None:
        package = json.loads(
            (PROJECT_ROOT / "package.json").read_text(encoding="utf-8")
        )
        package_lock = json.loads(
            (PROJECT_ROOT / "package-lock.json").read_text(encoding="utf-8")
        )
        with (PROJECT_ROOT / "mise.toml").open("rb") as mise_file:
            mise = tomllib.load(mise_file)

        command = package["scripts"]["typecheck:model-packages"]
        self.assertEqual(
            command,
            "pyright --project pyright-model-packages.json",
        )
        self.assertNotIn(" src/", command)
        task = mise["tasks"]["test:model-packages-types"]
        self.assertTrue(task["raw"])
        self.assertEqual(task["run"], "npm run typecheck:model-packages")
        self.assertEqual(package["devDependencies"]["pyright"], "1.1.411")
        self.assertEqual(
            package_lock["packages"]["node_modules/pyright"]["version"],
            "1.1.411",
        )

    def test_repository_versions_match_contracts(self) -> None:
        with (PROJECT_ROOT / "mise.toml").open("rb") as mise_file:
            mise = tomllib.load(mise_file)
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
            project = tomllib.load(project_file)

        self.assertEqual(mise["tools"], {"python": "3.13", "node": "24"})
        self.assertEqual(
            (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip(),
            "3.13",
        )
        self.assertEqual(project["project"]["requires-python"], ">=3.11,<3.14")


if __name__ == "__main__":
    unittest.main()
