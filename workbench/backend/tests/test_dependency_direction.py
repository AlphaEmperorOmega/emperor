from __future__ import annotations

import ast
import unittest
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from models.catalog import MODEL_CATALOG

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_PACKAGE_DIRS = ("emperor", "models")
CORE_PACKAGE_ROOTS = frozenset(CORE_PACKAGE_DIRS)
WORKBENCH_BACKEND_DIR = REPO_ROOT / "workbench" / "backend"
WORKBENCH_TESTS_DIR = WORKBENCH_BACKEND_DIR / "tests"
WORKBENCH_TEST_HELPERS = WORKBENCH_TESTS_DIR / "helpers.py"
TRAINING_JOB_SERVICE_TEST = WORKBENCH_TESTS_DIR / "test_training_job_service.py"
WORKBENCH_INSPECTION_ADAPTER = WORKBENCH_BACKEND_DIR / "inspection_adapter.py"
WORKBENCH_HISTORICAL_INSPECTION = WORKBENCH_BACKEND_DIR / "historical_inspection.py"
WORKBENCH_INSPECTION_SERVICE = WORKBENCH_BACKEND_DIR / "services" / "inspection.py"
WORKBENCH_RUN_PLAN_ADAPTER = (
    WORKBENCH_BACKEND_DIR / "training_jobs" / "run_plan_adapter.py"
)
WORKBENCH_TRAINING_JOB_CONTRACTS = (
    WORKBENCH_BACKEND_DIR / "training_jobs" / "contracts.py"
)
RUN_PLAN_ADAPTATION_IMPORTS = frozenset(
    {
        "PlanningBudget",
        "RunRequest",
        "SearchAxisSelection",
        "SearchSpec",
        "SubmittedRun",
        "accept_run_plan",
        "plan_runs",
    }
)
WORKBENCH_INSPECTION_IMPLEMENTATION_MODULES = frozenset(
    {
        "workbench.backend.inspection_serialization",
    }
)
WORKBENCH_INSPECTION_ADAPTER_PATHS = frozenset(
    {
        WORKBENCH_INSPECTION_ADAPTER,
        WORKBENCH_HISTORICAL_INSPECTION,
        WORKBENCH_INSPECTION_SERVICE,
        WORKBENCH_BACKEND_DIR / "api" / "v1" / "routers" / "inspection.py",
    }
)
LEGACY_WORKBENCH_MODELS_IMPORTS = frozenset(
    {(WORKBENCH_BACKEND_DIR / "cli.py", "models.parser")}
)
REMOVED_SHALLOW_INSPECTION_PATHS = (
    WORKBENCH_BACKEND_DIR / "services" / "models.py",
    WORKBENCH_BACKEND_DIR / "inspector" / "config_classes.py",
    WORKBENCH_BACKEND_DIR / "inspector" / "field_descriptions.py",
    WORKBENCH_BACKEND_DIR / "inspector" / "overrides.py",
    WORKBENCH_BACKEND_DIR / "inspector" / "values.py",
)
REMOVED_RUN_IMPLEMENTATION_PATHS = (
    REPO_ROOT / "emperor" / "experiments" / "progress.py",
    WORKBENCH_BACKEND_DIR / "inspector" / "search.py",
    WORKBENCH_BACKEND_DIR / "training_events.py",
)
REMOVED_TRAINING_JOB_IMPLEMENTATION_PATHS = (
    WORKBENCH_BACKEND_DIR / "job_store.py",
    WORKBENCH_BACKEND_DIR / "runtime" / "__init__.py",
    WORKBENCH_BACKEND_DIR / "runtime" / "job_status.py",
    WORKBENCH_BACKEND_DIR / "services" / "training.py",
    WORKBENCH_BACKEND_DIR / "training_cgroups.py",
    WORKBENCH_BACKEND_DIR / "training_contracts.py",
    WORKBENCH_BACKEND_DIR / "training_job_lifecycle.py",
    WORKBENCH_BACKEND_DIR / "training_job_projector.py",
    WORKBENCH_BACKEND_DIR / "training_jobs.py",
    WORKBENCH_BACKEND_DIR / "training_limits.py",
    WORKBENCH_BACKEND_DIR / "training_live_projection.py",
    WORKBENCH_BACKEND_DIR / "training_monitor_locator.py",
    WORKBENCH_BACKEND_DIR / "training_progress_store.py",
    WORKBENCH_BACKEND_DIR / "training_request_commands.py",
    WORKBENCH_BACKEND_DIR / "training_run_plans.py",
    WORKBENCH_BACKEND_DIR / "training_run_progress.py",
    WORKBENCH_BACKEND_DIR / "training_worker_launcher.py",
    WORKBENCH_BACKEND_DIR / "training_jobs" / "plans.py",
    WORKBENCH_BACKEND_DIR / "training_jobs" / "run_progress.py",
    WORKBENCH_BACKEND_DIR / "training_jobs" / "serialization.py",
)
REMOVED_RUN_HISTORY_IMPLEMENTATION_PATHS = (
    WORKBENCH_BACKEND_DIR / "log_experiment_mutations.py",
    WORKBENCH_BACKEND_DIR / "log_run_artifacts.py",
    WORKBENCH_BACKEND_DIR / "log_run_deletion.py",
    WORKBENCH_BACKEND_DIR / "log_run_models.py",
    WORKBENCH_BACKEND_DIR / "log_run_names.py",
    WORKBENCH_BACKEND_DIR / "log_run_query.py",
    WORKBENCH_BACKEND_DIR / "log_run_scanner.py",
    WORKBENCH_BACKEND_DIR / "log_runs.py",
    WORKBENCH_BACKEND_DIR / "monitor_data.py",
    WORKBENCH_BACKEND_DIR / "tensorboard_reader.py",
    WORKBENCH_BACKEND_DIR / "services" / "log_import.py",
    WORKBENCH_BACKEND_DIR / "services" / "logs.py",
)
RUN_HISTORY_DIR = WORKBENCH_BACKEND_DIR / "run_history"
SHARED_TENSORBOARD_DIR = WORKBENCH_BACKEND_DIR / "tensorboard"
EMPEROR_DIR = REPO_ROOT / "emperor"
MODELS_DIR = REPO_ROOT / "models"
CATALOG_PACKAGE_ROOTS = tuple(
    (REPO_ROOT.joinpath(*entry.module_path.split(".")), entry.module_path)
    for entry in MODEL_CATALOG.values()
)


@dataclass(frozen=True)
class AbsoluteImport:
    path: Path
    line_number: int
    module: str

    @property
    def root(self) -> str:
        return self.module.split(".", maxsplit=1)[0]

    def format_for_failure(self) -> str:
        relative_path = self.path.relative_to(REPO_ROOT)
        return f"{relative_path}:{self.line_number}: {self.module}"


@dataclass(frozen=True)
class ConfigImport:
    path: Path
    line_number: int
    imported_config_module: str

    def format_for_failure(self) -> str:
        relative_path = self.path.relative_to(REPO_ROOT)
        return f"{relative_path}:{self.line_number}: {self.imported_config_module}"


def _python_files(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def _absolute_imports(path: Path) -> Iterator[AbsoluteImport]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        relative_path = path.relative_to(REPO_ROOT)
        raise AssertionError(f"Could not parse {relative_path}: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield AbsoluteImport(path, node.lineno, alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield AbsoluteImport(path, node.lineno, node.module)


def _absolute_imports_under(roots: Iterable[Path]) -> Iterator[AbsoluteImport]:
    for path in _python_files(roots):
        yield from _absolute_imports(path)


def _models_package_for_path(path: Path) -> str:
    for package_root, module_path in CATALOG_PACKAGE_ROOTS:
        if path.is_relative_to(package_root):
            return module_path
    relative_path = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(relative_path.parts[:-1])


def _config_imports(path: Path) -> Iterator[ConfigImport]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        relative_path = path.relative_to(REPO_ROOT)
        raise AssertionError(f"Could not parse {relative_path}: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("models.") and alias.name.endswith(".config"):
                    yield ConfigImport(path, node.lineno, alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith("models.") and node.module.endswith(".config"):
                yield ConfigImport(path, node.lineno, node.module)
            elif node.module.startswith("models."):
                for alias in node.names:
                    if alias.name == "config":
                        yield ConfigImport(path, node.lineno, f"{node.module}.config")


def _config_imports_under(roots: Iterable[Path]) -> Iterator[ConfigImport]:
    for path in _python_files(roots):
        yield from _config_imports(path)


class DependencyDirectionTests(unittest.TestCase):
    def test_shallow_inspection_implementations_remain_removed(self) -> None:
        remaining = [
            str(path.relative_to(REPO_ROOT))
            for path in REMOVED_SHALLOW_INSPECTION_PATHS
            if path.exists()
        ]

        self.assertEqual([], remaining)

    def test_obsolete_run_implementations_remain_removed(self) -> None:
        remaining = [
            str(path.relative_to(REPO_ROOT))
            for path in REMOVED_RUN_IMPLEMENTATION_PATHS
            if path.exists()
        ]

        self.assertEqual([], remaining)

    def test_obsolete_training_job_implementations_remain_removed(self) -> None:
        remaining = [
            str(path.relative_to(REPO_ROOT))
            for path in REMOVED_TRAINING_JOB_IMPLEMENTATION_PATHS
            if path.exists()
        ]

        self.assertEqual([], remaining)

    def test_obsolete_run_history_implementations_remain_removed(self) -> None:
        remaining = [
            str(path.relative_to(REPO_ROOT))
            for path in REMOVED_RUN_HISTORY_IMPLEMENTATION_PATHS
            if path.exists()
        ]

        self.assertEqual([], remaining)

    def test_run_history_does_not_import_http_or_training_implementations(
        self,
    ) -> None:
        forbidden_prefixes = (
            "workbench.backend.api",
            "workbench.backend.training_jobs",
        )
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under([RUN_HISTORY_DIR])
            if source_import.module.startswith(forbidden_prefixes)
        ]

        self.assertEqual([], violations)

    def test_shared_tensorboard_does_not_import_capability_implementations(
        self,
    ) -> None:
        forbidden_prefixes = (
            "workbench.backend.run_history",
            "workbench.backend.training_jobs",
        )
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under([SHARED_TENSORBOARD_DIR])
            if source_import.module.startswith(forbidden_prefixes)
        ]

        self.assertEqual([], violations)

    def test_emperor_does_not_import_model_packages_by_legacy_namespace(
        self,
    ) -> None:
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under([EMPEROR_DIR])
            if source_import.root == "models"
        ]

        self.assertEqual([], violations)

    def test_core_packages_do_not_import_workbench(self) -> None:
        core_roots = [REPO_ROOT / package_dir for package_dir in CORE_PACKAGE_DIRS]
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under(core_roots)
            if source_import.root == "workbench"
        ]

        self.assertEqual([], violations)

    def test_workbench_backend_may_import_core_packages(self) -> None:
        imported_core_roots = {
            source_import.root
            for source_import in _absolute_imports_under([WORKBENCH_BACKEND_DIR])
            if source_import.root in CORE_PACKAGE_ROOTS
        }

        self.assertEqual(CORE_PACKAGE_ROOTS, imported_core_roots)

    def test_workbench_imports_only_allowlisted_public_emperor_interfaces(
        self,
    ) -> None:
        allowed = {
            "emperor.inspection",
            "emperor.model_packages",
            "emperor.runs",
        }
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under([WORKBENCH_BACKEND_DIR])
            if not source_import.path.is_relative_to(WORKBENCH_TESTS_DIR)
            and source_import.root == "emperor"
            and source_import.module not in allowed
        ]

        self.assertEqual([], violations)

    def test_workbench_uses_public_model_package_interface_except_cli(self) -> None:
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under([WORKBENCH_BACKEND_DIR])
            if not source_import.path.is_relative_to(WORKBENCH_TESTS_DIR)
            and source_import.root == "models"
            and (source_import.path, source_import.module)
            not in LEGACY_WORKBENCH_MODELS_IMPORTS
        ]

        self.assertEqual([], violations)

    def test_workbench_inspection_adaptation_has_one_implementation(self) -> None:
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under([WORKBENCH_BACKEND_DIR])
            if not source_import.path.is_relative_to(WORKBENCH_TESTS_DIR)
            and source_import.path not in WORKBENCH_INSPECTION_ADAPTER_PATHS
            and source_import.module in WORKBENCH_INSPECTION_IMPLEMENTATION_MODULES
        ]

        self.assertEqual([], violations)

    def test_historical_inspection_policy_has_one_deep_interface(self) -> None:
        service_source = WORKBENCH_INSPECTION_SERVICE.read_text(encoding="utf-8")
        historical_source = WORKBENCH_HISTORICAL_INSPECTION.read_text(encoding="utf-8")
        graph_test_source = (WORKBENCH_TESTS_DIR / "test_inspector_graph.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("WorkbenchHistoricalInspection", service_source)
        self.assertNotIn("load_checkpoint_graph_shapes", service_source)
        self.assertNotIn("def _checkpoint_overrides", service_source)
        self.assertIn("def _checkpoint_overrides", historical_source)
        self.assertNotIn("_checkpoint_overrides", graph_test_source)
        self.assertEqual(
            MODEL_CATALOG["linears/linear"].checkpoint_metadata_module,
            "models.linears.linear.checkpoint_metadata",
        )

    def test_domain_implementations_do_not_import_http_error_types(self) -> None:
        implementation_roots = [
            WORKBENCH_BACKEND_DIR / "training_jobs",
            WORKBENCH_BACKEND_DIR / "run_history",
            WORKBENCH_BACKEND_DIR / "log_experiments",
            WORKBENCH_BACKEND_DIR / "config_snapshots.py",
            WORKBENCH_BACKEND_DIR / "historical_inspection.py",
            WORKBENCH_BACKEND_DIR / "inspection_adapter.py",
            WORKBENCH_BACKEND_DIR / "inspection_errors.py",
            WORKBENCH_BACKEND_DIR / "inspection_worker.py",
            WORKBENCH_BACKEND_DIR / "model_identity.py",
            WORKBENCH_BACKEND_DIR / "services" / "inspection.py",
        ]
        forbidden_modules = {
            "workbench.backend.core.errors",
            "workbench.backend.inspector.errors",
        }
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under(implementation_roots)
            if source_import.module in forbidden_modules
        ]

        self.assertEqual([], violations)

    def test_workbench_run_plan_adaptation_has_one_implementation(self) -> None:
        violations: list[str] = []
        for path in _python_files([WORKBENCH_BACKEND_DIR]):
            if path == WORKBENCH_RUN_PLAN_ADAPTER or path.is_relative_to(
                WORKBENCH_TESTS_DIR
            ):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level != 0 or node.module != "emperor.runs":
                    continue
                duplicated = sorted(
                    alias.name
                    for alias in node.names
                    if alias.name in RUN_PLAN_ADAPTATION_IMPORTS
                )
                if duplicated:
                    relative_path = path.relative_to(REPO_ROOT)
                    violations.append(
                        f"{relative_path}:{node.lineno}: {', '.join(duplicated)}"
                    )

        self.assertEqual([], violations)

    def test_training_job_contracts_do_not_import_implementation_modules(self) -> None:
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports(WORKBENCH_TRAINING_JOB_CONTRACTS)
            if source_import.module.startswith("workbench.backend.training_jobs.")
        ]

        self.assertEqual([], violations)

    def test_production_uses_one_config_snapshot_interface(self) -> None:
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under([WORKBENCH_BACKEND_DIR])
            if not source_import.path.is_relative_to(WORKBENCH_TESTS_DIR)
            and source_import.module == "workbench.backend.services.config_snapshots"
        ]

        self.assertEqual([], violations)

    def test_worker_plan_tests_target_the_shared_adapter_interface(self) -> None:
        worker_test = WORKBENCH_TESTS_DIR / "test_training_worker.py"
        source = worker_test.read_text(encoding="utf-8")

        self.assertIn("run_plan_adapter.accept_worker_run_plan", source)
        self.assertNotIn("training_worker._accepted_plan", source)

    def test_training_job_workflow_harness_uses_public_interface(self) -> None:
        helper_tree = ast.parse(
            WORKBENCH_TEST_HELPERS.read_text(encoding="utf-8"),
            filename=str(WORKBENCH_TEST_HELPERS),
        )
        private_runtime_subclasses = [
            node.name
            for node in ast.walk(helper_tree)
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Name) and base.id == "_TrainingJobRuntime"
                for base in node.bases
            )
        ]
        private_runtime_assignments = [
            target.lineno
            for node in ast.walk(helper_tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Attribute) and target.attr == "_runtime"
        ]
        public_workflow_private_imports = [
            source_import.format_for_failure()
            for source_import in _absolute_imports(TRAINING_JOB_SERVICE_TEST)
            if source_import.module == "workbench.backend.training_jobs.runtime"
        ]

        self.assertEqual([], private_runtime_subclasses)
        self.assertEqual([], private_runtime_assignments)
        self.assertEqual([], public_workflow_private_imports)

    def test_model_packages_do_not_import_other_model_configs(self) -> None:
        violations = []
        for source_import in _config_imports_under([MODELS_DIR]):
            current_package = _models_package_for_path(source_import.path)
            allowed_config_module = f"{current_package}.config"
            if source_import.imported_config_module != allowed_config_module:
                violations.append(source_import.format_for_failure())

        self.assertEqual([], violations)
