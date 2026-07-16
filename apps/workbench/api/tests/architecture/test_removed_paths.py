from __future__ import annotations

import importlib
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


class ApiCompatibilityImportTests(unittest.TestCase):
    def test_api_package_is_the_exact_canonical_asgi_interface(self) -> None:
        api = importlib.import_module("emperor_workbench.api")

        self.assertEqual(api.__all__, ["app", "create_app"])
        self.assertFalse(hasattr(api, "COMPATIBILITY_STATUS"))
        self.assertFalse(hasattr(api, "REPLACEMENT_IMPORT"))
        self.assertFalse(hasattr(api, "__getattr__"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("emperor_workbench.main")

    def test_settings_module_is_canonical_and_old_config_path_is_absent(
        self,
    ) -> None:
        settings = importlib.import_module("emperor_workbench.settings")

        self.assertEqual(
            settings.WorkbenchApiSettings.__module__,
            "emperor_workbench.settings",
        )
        self.assertFalse(hasattr(settings, "COMPATIBILITY_STATUS"))
        self.assertFalse(hasattr(settings, "REPLACEMENT_IMPORT"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("emperor_workbench.core.config")

    def test_removed_legacy_route_and_empty_model_packages_stay_absent(self) -> None:
        for module_name in (
            "emperor_workbench.routes.config_snapshots",
            "emperor_workbench.routes.inspect",
            "emperor_workbench.routes.logs",
            "emperor_workbench.routes.models",
            "emperor_workbench.routes.training",
            "emperor_workbench.models",
        ):
            with (
                self.subTest(module=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)


class SchemaCompatibilityImportTests(unittest.TestCase):
    def test_global_schemas_package_is_absent(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("emperor_workbench.schemas")

    def test_config_snapshot_contracts_have_one_owned_interface(self) -> None:
        contracts = importlib.import_module("emperor_workbench.api.v1.config_snapshots")

        self.assertEqual(
            contracts.__all__,
            [
                "ConfigSnapshotCreateRequest",
                "ConfigSnapshotLibraryResponse",
                "ConfigSnapshotResponse",
                "ConfigSnapshotsResponse",
                "ConfigSnapshotUpdateRequest",
                "router",
            ],
        )
        for module_name in (
            "emperor_workbench.api.v1.config_snapshot_mapping",
            "emperor_workbench.api.v1.routers.config_snapshots",
            "emperor_workbench.schemas._config_snapshots",
            "emperor_workbench.services.config_snapshots",
        ):
            with (
                self.subTest(module=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)

    def test_inspection_contracts_have_one_owned_interface(self) -> None:
        contracts = importlib.import_module("emperor_workbench.api.v1.inspection")

        self.assertEqual(
            contracts.__all__,
            [
                "GraphConfigFieldResponse",
                "GraphConfigResponse",
                "GraphEdgeResponse",
                "GraphNodeResponse",
                "InspectRequest",
                "InspectResponse",
                "router",
            ],
        )
        for module_name in (
            "emperor_workbench.api.v1.routers.inspection",
            "emperor_workbench.historical_inspection",
            "emperor_workbench.inspection_adapter",
            "emperor_workbench.inspection_errors",
            "emperor_workbench.inspection_serialization",
            "emperor_workbench.inspection_worker",
            "emperor_workbench.schemas._inspection",
            "emperor_workbench.services.inspection",
        ):
            with (
                self.subTest(module=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)

    def test_run_history_contracts_have_one_owned_interface(self) -> None:
        contracts = importlib.import_module("emperor_workbench.api.v1.run_history")

        self.assertEqual(
            contracts.__all__,
            [
                "LogArchiveImportResponse",
                "LogCheckpointResponse",
                "LogCheckpointsRequest",
                "LogCheckpointsResponse",
                "LogExperimentDeleteResponse",
                "LogExperimentResponse",
                "LogExperimentsResponse",
                "LogImageSummaryResponse",
                "LogMediaRequest",
                "LogMediaResponse",
                "LogParameterStatusRequest",
                "LogParameterStatusResponse",
                "LogPresetDeleteRequest",
                "LogRunArtifactResponse",
                "LogRunArtifactsResponse",
                "LogRunDeleteAffectedValuesResponse",
                "LogRunDeleteBlockerResponse",
                "LogRunDeleteCandidateResponse",
                "LogRunDeleteCountsResponse",
                "LogRunDeleteFiltersRequest",
                "LogRunDeletePlanResponse",
                "LogRunDeleteResponse",
                "LogRunExperimentFacetsResponse",
                "LogRunFacetsResponse",
                "LogRunFacetValueResponse",
                "LogRunModelFacetResponse",
                "LogRunModelFilterRequest",
                "LogRunResponse",
                "LogRunsResponse",
                "LogRunTagsResponse",
                "LogScalarSeriesResponse",
                "LogScalarsRequest",
                "LogScalarsResponse",
                "LogTagsRequest",
                "LogTagsResponse",
                "LogTextSummaryResponse",
                "MonitorDataResponse",
                "router",
            ],
        )
        for module_name in (
            "emperor_workbench.api.v1.log_archive_upload",
            "emperor_workbench.api.v1.logs_mapping",
            "emperor_workbench.api.v1.routers.logs",
            "emperor_workbench.schemas._limits",
            "emperor_workbench.schemas._logs",
        ):
            with (
                self.subTest(module=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)

    def test_run_plan_contracts_have_one_owned_interface(self) -> None:
        contracts = importlib.import_module("emperor_workbench.api.v1.run_plans")

        self.assertEqual(
            contracts.__all__,
            [
                "ConfigSnapshotRevisionResponse",
                "SubmittedTrainingRunPlanRequest",
                "SubmittedTrainingRunRequest",
                "TrainingCommandsResponse",
                "TrainingRunChangeResponse",
                "TrainingRunPlanCreateRequest",
                "TrainingRunPlanResponse",
                "TrainingRunPlanSummaryResponse",
                "TrainingRunResponse",
                "TrainingSearchRequest",
                "TrainingSearchResponse",
                "router",
            ],
        )
        for module_name in (
            "emperor_workbench.training_jobs.run_plan_adapter",
            "emperor_workbench.training_jobs.limits",
        ):
            with (
                self.subTest(module=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)

    def test_training_jobs_have_one_owned_interface(self) -> None:
        http = importlib.import_module("emperor_workbench.api.v1.training_jobs")
        jobs = importlib.import_module("emperor_workbench.training_jobs")

        self.assertEqual(
            http.__all__,
            [
                "TrainingJobCreateRequest",
                "TrainingJobReconcileRequest",
                "TrainingJobResponse",
                "TrainingProgressEventsResponse",
                "router",
            ],
        )
        self.assertEqual(
            jobs.__all__,
            [
                "ActiveTrainingJob",
                "CreateTrainingJobCommand",
                "TrainingCancellationCapability",
                "TrainingCancellationMode",
                "TrainingJobFailure",
                "TrainingJobService",
                "TrainingJobStatus",
                "TrainingJobView",
                "TrainingProgressEventsPage",
                "TrainingResourceLimits",
                "TrainingResultLinkView",
            ],
        )
        for module_name in (
            "emperor_workbench.api.v1.routers.training",
            "emperor_workbench.api.v1.training_commands",
            "emperor_workbench.api.v1.training_mapping",
            "emperor_workbench.schemas._training",
            "emperor_workbench.training_jobs.cgroups",
            "emperor_workbench.training_jobs.contracts",
            "emperor_workbench.training_jobs.errors",
            "emperor_workbench.training_jobs.launcher",
            "emperor_workbench.training_jobs.lifecycle",
            "emperor_workbench.training_jobs.monitoring",
            "emperor_workbench.training_jobs.progress",
            "emperor_workbench.training_jobs.projection",
            "emperor_workbench.training_jobs.runtime",
            "emperor_workbench.training_jobs.service",
            "emperor_workbench.training_jobs.snapshot",
            "emperor_workbench.training_jobs.status",
            "emperor_workbench.training_jobs.store",
            "emperor_workbench.training_worker",
            "emperor_workbench.cgroup_worker_wrapper",
            "emperor_workbench.windows_jobs",
        ):
            with (
                self.subTest(module=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)

    def test_run_history_semantics_have_one_owned_interface(self) -> None:
        run_history = importlib.import_module("emperor_workbench.run_history")

        self.assertEqual(
            run_history.__all__,
            [
                "ActiveLogWriter",
                "ActiveLogWriterSource",
                "ActiveLogRunDeleteBlocker",
                "HistoricalCheckpointCandidate",
                "HistoricalInspectionContext",
                "HistoricalInspectionSource",
                "KnownModelPackageIdentityResolver",
                "LogArchiveImportResult",
                "LogCheckpoint",
                "LogExperiment",
                "LogExperimentDeleteResult",
                "LogExperimentPage",
                "LogImageSummary",
                "LogMedia",
                "LogRun",
                "LogRunArtifact",
                "LogRunArtifacts",
                "LogRunDeleteCandidate",
                "LogRunDeleteFilters",
                "LogRunDeletePlan",
                "LogRunDeleteResult",
                "LogRunExperimentFacets",
                "LogRunFacetValue",
                "LogRunFacets",
                "LogRunModelFacet",
                "LogRunPage",
                "LogRunTags",
                "LogScalarPoint",
                "LogScalarSeries",
                "LogTextSummary",
                "RunHistoryFailure",
                "RunHistoryService",
            ],
        )
        for module_name in (
            "emperor_workbench.run_history.archive",
            "emperor_workbench.run_history.artifacts",
            "emperor_workbench.run_history.contracts",
            "emperor_workbench.run_history.deletion",
            "emperor_workbench.run_history.errors",
            "emperor_workbench.run_history.paths",
            "emperor_workbench.run_history.query",
            "emperor_workbench.run_history.records",
            "emperor_workbench.run_history.scanner",
            "emperor_workbench.run_history.service",
        ):
            with (
                self.subTest(module=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)


class InspectorCompatibilityRemovalTests(unittest.TestCase):
    def test_obsolete_inspector_forwarders_stay_absent(self) -> None:
        for module_name in (
            "emperor_workbench.inspector.checkpoint_shapes",
            "emperor_workbench.inspector.discovery",
            "emperor_workbench.inspector.errors",
            "emperor_workbench.inspector.graph",
            "emperor_workbench.inspector.schema",
            "emperor_workbench.inspector.service",
        ):
            with (
                self.subTest(module=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)


class ExtensionPointImportTests(unittest.TestCase):
    def test_old_security_module_is_absent(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("emperor_workbench.core.security")

    def test_empty_database_extension_point_is_absent(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("emperor_workbench.db.session")


class ProjectCliBoundaryTests(unittest.TestCase):
    def test_workbench_cli_is_canonical_and_launch_alias_is_removed(self) -> None:
        cli = importlib.import_module("emperor_workbench.cli")

        self.assertEqual(cli.__all__, ["main"])
        self.assertTrue(callable(cli.main))
        self.assertFalse(hasattr(cli, "COMPATIBILITY_STATUS"))
        self.assertFalse(hasattr(cli, "REPLACEMENT_IMPORT"))
        self.assertFalse(hasattr(cli, "__getattr__"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("emperor_workbench.launch")

    def test_project_inspection_cli_exposes_callable_entrypoint(self) -> None:
        inspection_cli = importlib.import_module("models.inspection_cli")

        self.assertTrue(callable(inspection_cli.run_inspection))

    def test_project_cli_dispatches_inspection_arguments(self) -> None:
        project_cli = importlib.import_module("models.project_cli")
        arguments = ["--model-type", "linears", "--model", "linear"]

        with patch(
            "models.inspection_cli.run_inspection",
            return_value=0,
        ) as run_inspection:
            result = project_cli.main(["inspect", *arguments])

        self.assertEqual(result, 0)
        run_inspection.assert_called_once_with(arguments)


class ModelPackageCompatibilityTests(unittest.TestCase):
    def test_old_vit_transformer_encoder_package_import_is_unsupported(self) -> None:
        old_package = "models.transformer_encoder." + "vit" + "_linear.presets"

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module(old_package)


if __name__ == "__main__":
    unittest.main()
