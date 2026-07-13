from __future__ import annotations

import ast
import unittest
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from models.catalog import MODEL_CATALOG

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_PACKAGE_DIRS = ("emperor", "models")
WORKBENCH_BACKEND_DIR = REPO_ROOT / "workbench" / "backend"
WORKBENCH_TESTS_DIR = WORKBENCH_BACKEND_DIR / "tests"
WORKBENCH_TEST_HELPERS = WORKBENCH_TESTS_DIR / "helpers.py"
TRAINING_JOB_SERVICE_TEST = WORKBENCH_TESTS_DIR / "test_training_job_service.py"
WORKBENCH_INSPECTION_ADAPTER = WORKBENCH_BACKEND_DIR / "inspection_adapter.py"
WORKBENCH_PROJECT_ADAPTER = WORKBENCH_BACKEND_DIR / "project_adapter.py"
WORKBENCH_HISTORICAL_INSPECTION_DIR = (
    WORKBENCH_BACKEND_DIR / "historical_inspection"
)
WORKBENCH_HISTORICAL_INSPECTION = (
    WORKBENCH_HISTORICAL_INSPECTION_DIR / "_inspection.py"
)
OBSOLETE_WORKBENCH_INSPECTOR_ADAPTERS = tuple(
    WORKBENCH_BACKEND_DIR / "inspector" / filename
    for filename in (
        "__init__.py",
        "checkpoint_shapes.py",
        "discovery.py",
        "errors.py",
        "graph.py",
        "schema.py",
        "service.py",
    )
)
WORKBENCH_INSPECTION_SERVICE = WORKBENCH_BACKEND_DIR / "services" / "inspection.py"
WORKBENCH_RUN_PLAN_ADAPTER = (
    WORKBENCH_BACKEND_DIR / "training_jobs" / "run_plan_adapter.py"
)
WORKBENCH_TRAINING_JOB_CONTRACTS = (
    WORKBENCH_BACKEND_DIR / "training_jobs" / "contracts.py"
)
TYPED_RUN_PLAN_RUNTIME_MODULES = (
    WORKBENCH_BACKEND_DIR / "training_jobs" / "runtime.py",
    WORKBENCH_BACKEND_DIR / "training_jobs" / "store.py",
    WORKBENCH_BACKEND_DIR / "training_jobs" / "projection.py",
    WORKBENCH_BACKEND_DIR / "training_jobs" / "lifecycle.py",
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
PUBLIC_MODEL_RUNTIME_MODULES = frozenset(
    {
        "model_runtime",
        "model_runtime.cli",
        "model_runtime.inspection",
        "model_runtime.packages",
        "model_runtime.runs",
    }
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
    WORKBENCH_BACKEND_DIR / "run_history" / "checkpoint_ranking.py",
)
RUN_HISTORY_DIR = WORKBENCH_BACKEND_DIR / "run_history"
RUN_HISTORY_RECORDS = RUN_HISTORY_DIR / "records.py"
LOGS_HTTP_MAPPING = WORKBENCH_BACKEND_DIR / "api" / "v1" / "logs_mapping.py"
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


def _mentions_run_plan(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name) and "run_plan" in child.id
        for child in ast.walk(node)
    )


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

    def test_run_history_records_are_transport_neutral(self) -> None:
        source = RUN_HISTORY_RECORDS.read_text(encoding="utf-8")
        self.assertNotIn("def to_response", source)
        self.assertNotIn("model_identity_payload_from_id", source)

        tree = ast.parse(source, filename=str(RUN_HISTORY_RECORDS))
        transport_named_fields = [
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and any(character.isupper() for character in node.target.id)
        ]
        self.assertEqual([], transport_named_fields)

        mapping_source = LOGS_HTTP_MAPPING.read_text(encoding="utf-8")
        self.assertIn("model_identity_payload_from_id", mapping_source)
        self.assertIn("LOG_METADATA_RESPONSE_LIMIT", mapping_source)
        self.assertIn('"relativePath"', mapping_source)

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

    def test_workbench_production_does_not_import_project_packages(self) -> None:
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under([WORKBENCH_BACKEND_DIR])
            if not source_import.path.is_relative_to(WORKBENCH_TESTS_DIR)
            and source_import.root in {"emperor", "models"}
        ]

        self.assertEqual([], violations)

    def test_workbench_imports_only_public_model_runtime_interfaces(self) -> None:
        violations = [
            source_import.format_for_failure()
            for source_import in _absolute_imports_under([WORKBENCH_BACKEND_DIR])
            if not source_import.path.is_relative_to(WORKBENCH_TESTS_DIR)
            and source_import.root == "model_runtime"
            and source_import.module not in PUBLIC_MODEL_RUNTIME_MODULES
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

    def test_historical_inspection_owns_checkpoint_policy_and_interpretation(
        self,
    ) -> None:
        ranking_module = (
            WORKBENCH_HISTORICAL_INSPECTION_DIR / "_checkpoint_ranking.py"
        )
        shape_module = WORKBENCH_HISTORICAL_INSPECTION_DIR / "_checkpoint_shapes.py"
        self.assertTrue(ranking_module.is_file())
        self.assertTrue(shape_module.is_file())

        run_history_policy = [
            str(path.relative_to(REPO_ROOT))
            for path in _python_files([RUN_HISTORY_DIR])
            if "rank_historical_checkpoints" in path.read_text(encoding="utf-8")
            or "load_checkpoint_graph_shapes" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual([], run_history_policy)

        self.assertTrue(
            all(not path.exists() for path in OBSOLETE_WORKBENCH_INSPECTOR_ADAPTERS)
        )

    def test_domain_implementations_do_not_import_http_error_types(self) -> None:
        implementation_roots = [
            WORKBENCH_BACKEND_DIR / "training_jobs",
            WORKBENCH_BACKEND_DIR / "run_history",
            WORKBENCH_BACKEND_DIR / "log_experiments",
            WORKBENCH_BACKEND_DIR / "config_snapshots.py",
            WORKBENCH_HISTORICAL_INSPECTION_DIR,
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
            if path in {
                WORKBENCH_PROJECT_ADAPTER,
                WORKBENCH_RUN_PLAN_ADAPTER,
            } or path.is_relative_to(WORKBENCH_TESTS_DIR):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level != 0 or node.module != "model_runtime.runs":
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

    def test_training_job_runtime_modules_keep_run_plans_typed(self) -> None:
        contracts_source = WORKBENCH_TRAINING_JOB_CONTRACTS.read_text(
            encoding="utf-8"
        )
        self.assertNotIn("TrainingRunPlanDocument", contracts_source)

        violations: list[str] = []
        for path in TYPED_RUN_PLAN_RUNTIME_MODULES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                serialized_field: str | None = None
                if (
                    isinstance(node, ast.Subscript)
                    and _mentions_run_plan(node.value)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                ):
                    serialized_field = node.slice.value
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and _mentions_run_plan(node.func.value)
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    serialized_field = node.args[0].value
                if serialized_field is not None:
                    relative_path = path.relative_to(REPO_ROOT)
                    violations.append(
                        f"{relative_path}:{node.lineno}: {serialized_field}"
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
