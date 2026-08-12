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
CLI_ROOT = MODEL_RUNTIME_ROOT / "cli"
WORKBENCH_SOURCE_ROOT = PROJECT_ROOT / "apps" / "workbench" / "api" / "src"
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
    def test_wire_facade_exposes_only_record_specific_codecs(self) -> None:
        for module_name in (
            "_wire_inspection.py",
            "_wire_packages.py",
            "_wire_runs.py",
            "_wire_search.py",
            "_wire_shared.py",
        ):
            with self.subTest(module=module_name):
                self.assertTrue((CLI_ROOT / module_name).is_file())

        facade_path = CLI_ROOT / "wire.py"
        facade = ast.parse(
            facade_path.read_text(encoding="utf-8"),
            facade_path.as_posix(),
        )
        facade_functions = {
            node.name for node in facade.body if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual(facade_functions, set())

        for public_path in (facade_path, CLI_ROOT / "__init__.py"):
            public_tree = ast.parse(
                public_path.read_text(encoding="utf-8"),
                public_path.as_posix(),
            )
            exported = next(
                set(ast.literal_eval(node.value))
                for node in public_tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
            )
            self.assertNotIn("to_wire", exported)

        unrestricted_projector_imports: list[str] = []
        for root in (SOURCE_ROOT / "models", WORKBENCH_SOURCE_ROOT):
            for source_path in sorted(root.rglob("*.py")):
                tree = ast.parse(
                    source_path.read_text(encoding="utf-8"),
                    source_path.as_posix(),
                )
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.ImportFrom)
                        and node.module
                        in {"model_runtime.cli", "model_runtime.cli.wire"}
                        and any(alias.name == "to_wire" for alias in node.names)
                    ):
                        unrestricted_projector_imports.append(
                            source_path.relative_to(PROJECT_ROOT).as_posix()
                        )

        self.assertEqual(unrestricted_projector_imports, [])

    def test_wire_decoders_own_complete_transport_acceptance(self) -> None:
        graph_source = (CLI_ROOT / "_wire_graph.py").read_text(encoding="utf-8")
        facade_source = (CLI_ROOT / "wire.py").read_text(encoding="utf-8")
        public_source = (CLI_ROOT / "__init__.py").read_text(encoding="utf-8")
        adapter_source = (SOURCE_ROOT / "models" / "adapter_cli.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("InspectionCapture", graph_source)
        self.assertNotIn("InspectionError", graph_source)
        self.assertNotIn("require_transportable_run_budget", facade_source)
        self.assertNotIn("require_transportable_run_budget", public_source)
        self.assertNotIn("require_transportable_run_budget", adapter_source)

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

    def test_model_package_interface_hides_metadata_source_modules(self) -> None:
        source_names = {
            "dataset_options",
            "monitor_options_source",
            "runtime_defaults",
            "search_space",
        }
        exposed: dict[str, set[str]] = {}
        for path, class_name in (
            (MODEL_RUNTIME_ROOT / "packages" / "definition.py", "ModelPackage"),
            (MODEL_RUNTIME_ROOT / "packages" / "metadata.py", "ModelMetadata"),
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"), path.as_posix())
            module_class = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            public_names = {
                name
                for node in module_class.body
                for name in (
                    node.name
                    if isinstance(node, ast.FunctionDef)
                    else (
                        node.target.id
                        if isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Name)
                        else ""
                    ),
                )
                if name and not name.startswith("_")
            }
            leaked = public_names & source_names
            if leaked:
                exposed[class_name] = leaked

        self.assertEqual(exposed, {})

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
