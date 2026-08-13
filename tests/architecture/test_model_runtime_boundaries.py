from __future__ import annotations

import ast
import json
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
WORKBENCH_WEB_ROOT = PROJECT_ROOT / "apps" / "workbench" / "web"
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

    def test_cli_wire_policies_have_one_private_owner(self) -> None:
        module_names = (
            "_wire_graph.py",
            "_wire_packages.py",
            "_wire_runs.py",
            "_wire_search.py",
            "_wire_shared.py",
        )
        trees = {
            name: ast.parse(
                (CLI_ROOT / name).read_text(encoding="utf-8"),
                (CLI_ROOT / name).as_posix(),
            )
            for name in module_names
        }

        def function_owners(function_name: str) -> list[str]:
            return [
                name
                for name, tree in trees.items()
                if any(
                    isinstance(node, ast.FunctionDef) and node.name == function_name
                    for node in tree.body
                )
            ]

        self.assertEqual(
            function_owners("require_sequence_limit"),
            ["_wire_shared.py"],
        )
        self.assertEqual(
            function_owners("experiment_task_from_wire"),
            ["_wire_packages.py"],
        )

        for consumer in ("_wire_graph.py", "_wire_runs.py", "_wire_search.py"):
            with self.subTest(sequence_limit_consumer=consumer):
                self.assertTrue(
                    any(
                        isinstance(node, ast.ImportFrom)
                        and node.module == "model_runtime.cli._wire_shared"
                        and any(
                            alias.name == "require_sequence_limit"
                            for alias in node.names
                        )
                        for node in trees[consumer].body
                    )
                )

        shared_wire_list = next(
            node
            for node in trees["_wire_shared.py"].body
            if isinstance(node, ast.FunctionDef) and node.name == "wire_list"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "require_sequence_limit"
                for node in ast.walk(shared_wire_list)
            )
        )

        runs_tree = trees["_wire_runs.py"]
        self.assertTrue(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "model_runtime.cli._wire_packages"
                and any(
                    alias.name == "experiment_task_from_wire" for alias in node.names
                )
                for node in runs_tree.body
            )
        )
        self.assertFalse(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "emperor.experiments"
                for node in runs_tree.body
            )
        )

        for public_module in ("wire.py", "__init__.py"):
            source = (CLI_ROOT / public_module).read_text(encoding="utf-8")
            with self.subTest(public_module=public_module):
                self.assertNotIn("require_sequence_limit", source)
                self.assertNotIn("experiment_task_from_wire", source)

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

    def test_inspection_runtime_defaults_translation_has_one_owner(self) -> None:
        runtime_defaults_source = (INSPECTION_ROOT / "runtime_defaults.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def apply_runtime_defaults(", runtime_defaults_source)
        self.assertIn(
            "def raise_runtime_defaults_inspection_error(",
            runtime_defaults_source,
        )

        duplicated_translation = (
            "raise InspectionError(str(exc)) from (exc.__cause__ or exc)"
        )
        for source_path in sorted(INSPECTION_ROOT.glob("*.py")):
            if source_path.name == "runtime_defaults.py":
                continue
            with self.subTest(module=source_path.name):
                self.assertNotIn(
                    duplicated_translation,
                    source_path.read_text(encoding="utf-8"),
                )

        facade_source = (INSPECTION_ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("apply_runtime_defaults", facade_source)
        self.assertNotIn("raise_runtime_defaults_inspection_error", facade_source)

    def test_runs_consume_runtime_defaults_through_model_package_interface(
        self,
    ) -> None:
        violations = [
            (
                source_path.relative_to(PROJECT_ROOT).as_posix(),
                imported_module,
            )
            for source_path, imported_module in _imports_under(RUNS_ROOT)
            if imported_module == "model_runtime.inspection"
            or imported_module.startswith("model_runtime.inspection.")
        ]

        self.assertEqual(violations, [])

    def test_runs_own_a_typed_run_to_experiment_handoff(self) -> None:
        handoff_path = RUNS_ROOT / "_handoff.py"
        self.assertTrue(handoff_path.is_file())
        tree = ast.parse(
            handoff_path.read_text(encoding="utf-8"),
            handoff_path.as_posix(),
        )
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }

        request = classes["TrainingRunRequest"]
        request_decorators = {
            ast.unparse(decorator) for decorator in request.decorator_list
        }
        self.assertIn("dataclass(frozen=True, slots=True)", request_decorators)
        execution_request = classes["TrainingExecutionRequest"]
        execution_request_decorators = {
            ast.unparse(decorator) for decorator in execution_request.decorator_list
        }
        self.assertIn(
            "dataclass(frozen=True, slots=True)",
            execution_request_decorators,
        )
        experiment_port = classes["RunExperiment"]
        self.assertEqual(
            {
                node.name
                for node in experiment_port.body
                if isinstance(node, ast.FunctionDef)
            },
            {"execute_training", "materialize_training_runs"},
        )
        self.assertIn("Protocol", {ast.unparse(base) for base in experiment_port.bases})
        materialize_operation = next(
            node
            for node in experiment_port.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "materialize_training_runs"
        )
        materialize_contract = ast.get_docstring(materialize_operation) or ""
        self.assertIn("same order", materialize_contract)
        for identity_field in (
            "run id",
            "run index",
            "run total",
            "preset identity",
            "Dataset identity",
        ):
            with self.subTest(identity_field=identity_field):
                self.assertIn(identity_field, materialize_contract)

        execution_source = (RUNS_ROOT / "execution.py").read_text(encoding="utf-8")
        experiment_source = (RUNS_ROOT / "experiment.py").read_text(encoding="utf-8")
        package_source = (MODEL_RUNTIME_ROOT / "packages" / "definition.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('run["config_overrides"]', execution_source)
        self.assertNotIn('run["preset"]', experiment_source)
        self.assertNotIn('run["dataset_type"]', experiment_source)
        self.assertIn(") -> RunExperiment:", package_source)

    def test_runs_immutable_values_have_one_private_policy_owner(self) -> None:
        value_policy_path = RUNS_ROOT / "_value_policy.py"
        self.assertTrue(value_policy_path.is_file())

        consumers = {
            name: (RUNS_ROOT / name).read_text(encoding="utf-8")
            for name in ("records.py", "_handoff.py", "_progress_events.py")
        }
        for name, source in consumers.items():
            with self.subTest(consumer=name):
                self.assertIn(
                    "from model_runtime.runs._value_policy import",
                    source,
                )

        self.assertNotIn("def _freeze_value", consumers["records.py"])
        self.assertNotIn(
            "MappingProxyType(dict(",
            consumers["_handoff.py"],
        )
        runs_facade = (RUNS_ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("_value_policy", runs_facade)
        self.assertNotIn("deep_freeze", runs_facade)
        self.assertNotIn("deep_thaw", runs_facade)

    def test_inspection_snapshots_normalize_every_collection_owner(self) -> None:
        records_path = INSPECTION_ROOT / "records.py"
        records_source = records_path.read_text(encoding="utf-8")
        records_tree = ast.parse(records_source, records_path.as_posix())
        classes = {
            node.name: node
            for node in records_tree.body
            if isinstance(node, ast.ClassDef)
        }
        collection_owners = (
            "InspectionRequest",
            "ParsedOverrides",
            "ConfigurationFieldCondition",
            "ConfigurationField",
            "ConfigurationSchema",
            "SearchAxis",
            "SearchSpace",
            "GraphConfigurationField",
            "GraphConfiguration",
            "GraphNode",
            "ModelGraph",
            "InspectionResult",
        )
        for class_name in collection_owners:
            with self.subTest(class_name=class_name):
                self.assertIn(
                    "__post_init__",
                    {
                        node.name
                        for node in classes[class_name].body
                        if isinstance(node, ast.FunctionDef)
                    },
                )

        freeze_owners: list[str] = []
        for source_path in sorted(INSPECTION_ROOT.glob("*.py")):
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                source_path.as_posix(),
            )
            if any(
                isinstance(node, ast.FunctionDef) and node.name == "freeze_value"
                for node in tree.body
            ):
                freeze_owners.append(source_path.name)
        self.assertEqual(freeze_owners, ["records.py"])
        self.assertIn("(set, frozenset)", records_source)

        inspection_facade = (INSPECTION_ROOT / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"freeze_value"', inspection_facade)

    def test_parsed_override_admission_has_one_non_facade_owner(self) -> None:
        records_source = (INSPECTION_ROOT / "records.py").read_text(encoding="utf-8")
        overrides_source = (INSPECTION_ROOT / "overrides.py").read_text(
            encoding="utf-8"
        )
        materialization_source = (INSPECTION_ROOT / "materialization.py").read_text(
            encoding="utf-8"
        )
        inspection_facade = (INSPECTION_ROOT / "__init__.py").read_text(
            encoding="utf-8"
        )
        local_cli_source = (SOURCE_ROOT / "models" / "inspection_cli.py").read_text(
            encoding="utf-8"
        )

        for helper in (
            "parsed_overrides_for_identity",
            "parsed_overrides_identity",
        ):
            with self.subTest(helper=helper):
                self.assertIn(f"def {helper}(", records_source)
                self.assertNotIn(f'"{helper}"', inspection_facade)
        self.assertIn("def validate_typed_overrides(", overrides_source)
        self.assertIn("def validated_overrides_for_materialization(", overrides_source)
        self.assertIn(
            "return validate_typed_overrides(package, overrides.values, preset=preset)",
            overrides_source,
        )
        self.assertIn(
            "validated_overrides_for_materialization(",
            materialization_source,
        )
        self.assertNotIn(
            "if isinstance(request.overrides, ParsedOverrides):",
            materialization_source,
        )
        self.assertIn("validate_typed_overrides(", local_cli_source)
        self.assertNotIn(
            "ParsedOverrides(selection.config_overrides)", local_cli_source
        )
        self.assertNotIn('"validate_typed_overrides"', inspection_facade)
        self.assertNotIn('"validated_overrides_for_materialization"', inspection_facade)

    def test_runs_own_one_checkpoint_continuation_lifecycle(self) -> None:
        checkpoint_path = RUNS_ROOT / "checkpoints.py"
        checkpoint_tree = ast.parse(
            checkpoint_path.read_text(encoding="utf-8"),
            checkpoint_path.as_posix(),
        )
        checkpoint_classes = {
            node.name: node
            for node in checkpoint_tree.body
            if isinstance(node, ast.ClassDef)
        }
        lifecycle = checkpoint_classes["CheckpointContinuationLifecycle"]
        lifecycle_methods = {
            node.name
            for node in lifecycle.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertTrue({"admit", "bind_training_runs"} <= lifecycle_methods)

        execution_source = (RUNS_ROOT / "execution.py").read_text(encoding="utf-8")
        leaked_implementation = {
            "_LoadedCheckpointContinuation",
            "load_checkpoint_continuation",
            "resumed_from_payload",
            "validate_model_state",
            "validate_target_epochs",
            "_ContinuationExecution",
        }
        self.assertTrue(
            all(name not in execution_source for name in leaked_implementation)
        )

    def test_run_projection_uses_each_presets_search_provenance(self) -> None:
        records_source = (RUNS_ROOT / "records.py").read_text(encoding="utf-8")
        planning_source = (RUNS_ROOT / "planning.py").read_text(encoding="utf-8")
        service_source = (
            WORKBENCH_SOURCE_ROOT.parent
            / "src"
            / "emperor_workbench"
            / "run_plans"
            / "_service.py"
        ).read_text(encoding="utf-8")
        worker_source = (
            WORKBENCH_SOURCE_ROOT.parent
            / "src"
            / "emperor_workbench"
            / "run_plans"
            / "_worker_acceptance.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class PresetSearch", records_source)
        self.assertIn("def search_for_preset", records_source)
        self.assertNotIn("def normalized_search", planning_source)
        self.assertNotIn("search=semantic_plan.search,", service_source)
        self.assertNotIn("search=semantic_plan.search,", worker_source)

    def test_package_experiments_do_not_repeat_obsolete_construction_hooks(
        self,
    ) -> None:
        obsolete_hooks = {
            "_dataset_options",
            "_experiment_preset_enum",
            "_model_type",
            "_preset_generator_instance",
        }
        violations: list[tuple[str, str]] = []
        for source_path in sorted((SOURCE_ROOT / "models").glob("*/*/presets.py")):
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                source_path.as_posix(),
            )
            experiment = next(
                (
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "Experiment"
                ),
                None,
            )
            if experiment is None:
                continue
            violations.extend(
                (
                    source_path.relative_to(PROJECT_ROOT).as_posix(),
                    node.name,
                )
                for node in experiment.body
                if isinstance(node, ast.FunctionDef) and node.name in obsolete_hooks
            )

        self.assertEqual(violations, [])

    def test_run_progress_producers_use_the_typed_event_vocabulary(self) -> None:
        vocabulary_path = RUNS_ROOT / "_progress_events.py"
        self.assertTrue(vocabulary_path.is_file())
        raw_event_mappings: list[tuple[str, int]] = []
        for source_path in (
            RUNS_ROOT / "experiment.py",
            RUNS_ROOT / "_lightning_progress.py",
        ):
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                source_path.as_posix(),
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                if any(
                    isinstance(key, ast.Constant) and key.value == "type"
                    for key in node.keys
                ):
                    raw_event_mappings.append(
                        (
                            source_path.relative_to(PROJECT_ROOT).as_posix(),
                            node.lineno,
                        )
                    )

        self.assertEqual(raw_event_mappings, [])

    def test_workbench_manifest_matches_the_runtime_progress_vocabulary(self) -> None:
        from model_runtime.runs.progress import (
            MODEL_RUNTIME_PROGRESS_CONTEXT_FIELDS,
            MODEL_RUNTIME_PROGRESS_EVENT_FIELDS,
            MODEL_RUNTIME_PROGRESS_OPTIONAL_FIELDS,
        )

        manifest_path = (
            WORKBENCH_WEB_ROOT
            / "src"
            / "lib"
            / "api"
            / "model-runtime-progress-events.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "contextFields": list(MODEL_RUNTIME_PROGRESS_CONTEXT_FIELDS),
            "events": {
                event_type: {
                    "fields": list(fields),
                    "optionalFields": list(
                        MODEL_RUNTIME_PROGRESS_OPTIONAL_FIELDS[event_type]
                    ),
                }
                for event_type, fields in MODEL_RUNTIME_PROGRESS_EVENT_FIELDS.items()
            },
        }

        self.assertEqual(manifest, expected)
        web_contract = (
            WORKBENCH_WEB_ROOT / "src" / "lib" / "api" / "training-jobs.ts"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'import modelRuntimeProgressEvents from "./model-runtime-progress-events.json";',
            web_contract,
        )
        self.assertIn("Object.keys(modelRuntimeProgressEvents.events)", web_contract)

    def test_inspection_owns_explicit_semantic_graph_adapters(self) -> None:
        semantics_path = INSPECTION_ROOT / "_graph_semantics.py"
        self.assertTrue(semantics_path.is_file())
        semantics = ast.parse(
            semantics_path.read_text(encoding="utf-8"),
            semantics_path.as_posix(),
        )
        adapter_classes = {
            node.name
            for node in semantics.body
            if isinstance(node, ast.ClassDef) and node.name.endswith("DetailsAdapter")
        }
        self.assertGreaterEqual(len(adapter_classes), 2)

        catalog_source = (INSPECTION_ROOT / "_graph_semantic_catalog.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class SemanticTypePolicy", catalog_source)
        self.assertIn("class SemanticTypeCatalog", catalog_source)
        component_semantics_source = (
            INSPECTION_ROOT / "_graph_component_semantics.py"
        ).read_text(encoding="utf-8")
        self.assertIn("SEMANTIC_TYPE_CATALOG", component_semantics_source)
        self.assertNotIn("def _registered_type_id", component_semantics_source)
        for parallel_type_id_policy in (
            "DESCRIPTION_BY_TYPE_ID",
            "INTERNAL_ROLE_TYPE_IDS",
            "RUNTIME_ROLE_TYPE_IDS",
            "RESIDUAL_FIELD_DESCRIPTIONS_BY_CONFIG_TYPE_ID",
        ):
            self.assertNotIn(parallel_type_id_policy, catalog_source)

        model_graph_source = (INSPECTION_ROOT / "model_graph.py").read_text(
            encoding="utf-8"
        )
        for bare_name_policy in (
            "COMPONENT_DESCRIPTION_BY_CLASS_NAME",
            "INTERNAL_GRAPH_TYPE_NAMES",
            "RUNTIME_GRAPH_TYPE_NAMES",
        ):
            self.assertNotIn(bare_name_policy, model_graph_source)
        self.assertIn("ModuleSemanticAdapter", model_graph_source)

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

    def test_runtime_defaults_metadata_has_one_private_snapshot_owner(self) -> None:
        metadata_path = MODEL_RUNTIME_ROOT / "packages" / "metadata.py"
        metadata_source = metadata_path.read_text(encoding="utf-8")
        metadata_tree = ast.parse(metadata_source, metadata_path.as_posix())
        snapshot_classes = {
            node.name
            for node in metadata_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_MetadataSnapshot"
        }

        self.assertEqual(snapshot_classes, {"_MetadataSnapshot"})
        self.assertIn("return _MetadataSnapshot(metadata)", metadata_source)
        self.assertEqual(metadata_source.count("_nested_metadata("), 3)
        self.assertNotIn("MappingProxyType(dict(value))", metadata_source)

        packages_facade = (MODEL_RUNTIME_ROOT / "packages" / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_MetadataSnapshot", packages_facade)

    def test_preset_definitions_have_one_private_snapshot_owner(self) -> None:
        presets_path = MODEL_RUNTIME_ROOT / "packages" / "presets.py"
        presets_source = presets_path.read_text(encoding="utf-8")
        presets_tree = ast.parse(presets_source, presets_path.as_posix())
        snapshot_classes = {
            node.name
            for node in presets_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_PresetDefinitionSnapshot"
        }

        self.assertEqual(snapshot_classes, {"_PresetDefinitionSnapshot"})
        self.assertIn(
            "self._preset_definition_snapshots = MappingProxyType(", presets_source
        )
        self.assertIn("def _definition_snapshot_for_preset(", presets_source)
        self.assertNotIn("self._preset_definitions = dict(", presets_source)

        packages_facade = (MODEL_RUNTIME_ROOT / "packages" / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_PresetDefinitionSnapshot", packages_facade)

    def test_runtime_defaults_owns_typed_preset_lock_normalization(self) -> None:
        runtime_defaults_path = MODEL_RUNTIME_ROOT / "packages" / "runtime_defaults.py"
        runtime_defaults_source = runtime_defaults_path.read_text(encoding="utf-8")
        self.assertIn("def _snapshot_preset_lock(", runtime_defaults_source)
        self.assertIn("dict[str, PresetLock]", runtime_defaults_source)

        reflected_value_owners: set[str] = set()
        reflected_reason_owners: set[str] = set()
        for source_path in MODEL_RUNTIME_ROOT.rglob("*.py"):
            source = source_path.read_text(encoding="utf-8")
            relative_path = source_path.relative_to(MODEL_RUNTIME_ROOT).as_posix()
            if 'getattr(lock, "value"' in source:
                reflected_value_owners.add(relative_path)
            if 'getattr(lock, "reason"' in source:
                reflected_reason_owners.add(relative_path)

        expected_structural_owners = {
            "cli/_wire_inspection.py",
            "packages/runtime_defaults.py",
        }
        self.assertEqual(reflected_value_owners, expected_structural_owners)
        self.assertEqual(reflected_reason_owners, expected_structural_owners)

        packages_facade = (MODEL_RUNTIME_ROOT / "packages" / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_snapshot_preset_lock", packages_facade)

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
