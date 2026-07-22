from __future__ import annotations

import ast
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
EMPEROR_ROOT = SOURCE_ROOT / "emperor"
MODEL_RUNTIME_ROOT = SOURCE_ROOT / "model_runtime"
INSPECTION_ROOT = MODEL_RUNTIME_ROOT / "inspection"
RUNS_ROOT = MODEL_RUNTIME_ROOT / "runs"
PROJECT_CLI_ROOT = SOURCE_ROOT / "models" / "project_cli"
PUBLIC_RUNTIME_PACKAGES = ("packages", "inspection", "runs", "cli")


def _imports_under(root: Path) -> list[tuple[Path, str]]:
    imports: list[tuple[Path, str]] = []
    for source_path in sorted(root.rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            source_path.as_posix(),
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((source_path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((source_path, node.module))
    return imports


class ModelRuntimeBoundaryTests(unittest.TestCase):
    def test_run_artifacts_own_lifecycle_without_experiment_forwarders(self) -> None:
        self.assertFalse((RUNS_ROOT / "locking.py").exists())
        experiment_path = RUNS_ROOT / "experiment.py"
        tree = ast.parse(
            experiment_path.read_text(encoding="utf-8"),
            experiment_path.as_posix(),
        )
        experiment = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ExperimentBase"
        )
        methods = {
            node.name for node in experiment.body if isinstance(node, ast.FunctionDef)
        }
        artifact_forwarders = {
            "load_best_results",
            "_load_best_results",
            "_write_training_result",
            "_update_best_results",
            "_result_ranking_score",
            "_best_results_path",
            "_build_log_path",
            "_artifact_store",
        }

        self.assertTrue(methods.isdisjoint(artifact_forwarders))

    def test_inspection_modules_do_not_import_each_others_private_names(
        self,
    ) -> None:
        violations: list[tuple[str, str, str]] = []
        for source_path in sorted(INSPECTION_ROOT.glob("*.py")):
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                source_path.as_posix(),
            )
            for node in ast.walk(tree):
                if (
                    not isinstance(node, ast.ImportFrom)
                    or node.module is None
                    or not node.module.startswith("model_runtime.inspection.")
                ):
                    continue
                violations.extend(
                    (
                        source_path.relative_to(PROJECT_ROOT).as_posix(),
                        node.module,
                        alias.name,
                    )
                    for alias in node.names
                    if alias.name.startswith("_")
                )

        self.assertEqual(violations, [])

    def test_inspection_root_does_not_export_graph_implementation_helpers(
        self,
    ) -> None:
        source_path = INSPECTION_ROOT / "__init__.py"
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            source_path.as_posix(),
        )
        all_assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        )
        exported = set(ast.literal_eval(all_assignment.value))
        graph_implementation_names = {
            "ARCHITECTURE_ROLE",
            "INTERNAL_ROLE",
            "ROOT_NODE_ID",
            "ROOT_NODE_PATH",
            "RUNTIME_ROLE",
            "graph_role",
            "inspect_model_graph",
            "module_details",
            "parameter_count",
            "parameter_size_bytes",
        }

        self.assertTrue(exported.isdisjoint(graph_implementation_names))

    def test_project_cli_is_owned_outside_emperor(self) -> None:
        self.assertFalse((EMPEROR_ROOT / "cli").exists())
        self.assertFalse((EMPEROR_ROOT / "__main__.py").exists())
        self.assertTrue((PROJECT_CLI_ROOT / "__main__.py").is_file())

        with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
            project = tomllib.load(project_file)["project"]

        self.assertEqual(
            project["scripts"]["emperor"],
            "models.project_cli:main",
        )
        self.assertNotIn("emperor.project_adapter", project.get("entry-points", {}))

    def test_runtime_public_package_shell_exists(self) -> None:
        self.assertTrue((MODEL_RUNTIME_ROOT / "__init__.py").is_file())
        for package in PUBLIC_RUNTIME_PACKAGES:
            with self.subTest(package=package):
                self.assertTrue(
                    (MODEL_RUNTIME_ROOT / package / "__init__.py").is_file()
                )

    def test_emperor_has_no_outward_project_imports(self) -> None:
        forbidden = [
            (path, module)
            for path, module in _imports_under(EMPEROR_ROOT)
            if module == "model_runtime"
            or module.startswith("model_runtime.")
            or module == "models"
            or module.startswith("models.")
            or module == "workbench"
            or module.startswith("workbench.")
            or module == "emperor_workbench"
            or module.startswith("emperor_workbench.")
        ]

        self.assertEqual(forbidden, [])

    def test_generic_runtime_has_no_project_or_workbench_imports(self) -> None:
        forbidden = [
            (path, module)
            for path, module in _imports_under(MODEL_RUNTIME_ROOT)
            if module == "models"
            or module.startswith("models.")
            or module == "workbench"
            or module.startswith("workbench.")
            or module == "emperor_workbench"
            or module.startswith("emperor_workbench.")
        ]

        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
