from __future__ import annotations

import importlib
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

EXPECTED_SCHEMA_EXPORTS = [
    "ConfigValue",
    "ConfigOverrides",
    "JsonValue",
    "JsonObject",
    "ApiResponseModel",
    "CapabilitiesResponse",
    "ConfigSnapshotResponse",
    "ConfigSnapshotsResponse",
    "ConfigSnapshotLibraryResponse",
    "ConfigSnapshotCreateRequest",
    "ConfigSnapshotRenameRequest",
    "ConfigSnapshotUpdateRequest",
    "HealthResponse",
    "ModelIdentityResponse",
    "ModelsResponse",
    "PresetResponse",
    "PresetsResponse",
    "DatasetResponse",
    "DatasetsResponse",
    "MonitorOptionResponse",
    "MonitorsResponse",
    "ConfigFieldResponse",
    "ConfigSchemaResponse",
    "SearchAxisResponse",
    "SearchSpaceResponse",
    "GraphConfigFieldResponse",
    "GraphConfigResponse",
    "GraphNodeResponse",
    "GraphEdgeResponse",
    "InspectRequest",
    "InspectResponse",
    "TrainingJobCreateRequest",
    "ConfigSnapshotRevisionResponse",
    "TrainingJobReconcileRequest",
    "TrainingRunPlanCreateRequest",
    "TrainingSearchResponse",
    "TrainingSearchRequest",
    "TrainingRunChangeResponse",
    "TrainingRunResponse",
    "SubmittedTrainingRunRequest",
    "TrainingRunPlanSummaryResponse",
    "TrainingRunPlanResponse",
    "SubmittedTrainingRunPlanRequest",
    "TrainingResultLinkResponse",
    "TrainingClusterGrowthAdditionResponse",
    "TrainingClusterGrowthResponse",
    "TrainingJobResponse",
    "TrainingProgressEventsResponse",
    "ScalarPointResponse",
    "ScalarSeriesResponse",
    "HistogramBucketResponse",
    "HistogramResponse",
    "ImageResponse",
    "MonitorDataResponse",
    "ParameterChannelStatusResponse",
    "ParameterNodeStatusResponse",
    "ParameterStatusResponse",
    "LogParameterStatusRequest",
    "LogParameterStatusResponse",
    "LogRunResponse",
    "LogRunsResponse",
    "LogRunFacetValueResponse",
    "LogRunModelFacetResponse",
    "LogRunExperimentFacetsResponse",
    "LogRunFacetsResponse",
    "LogCheckpointsRequest",
    "LogCheckpointResponse",
    "LogCheckpointsResponse",
    "LogRunArtifactResponse",
    "LogRunArtifactsResponse",
    "LogExperimentResponse",
    "LogExperimentsResponse",
    "LogExperimentDeleteResponse",
    "LogArchiveImportResponse",
    "LogMediaRequest",
    "LogImageSummaryResponse",
    "LogTextSummaryResponse",
    "LogMediaResponse",
    "LogPresetDeleteRequest",
    "LogRunDeleteFiltersRequest",
    "LogRunModelFilterRequest",
    "LogRunDeleteCandidateResponse",
    "LogRunDeleteAffectedValuesResponse",
    "LogRunDeleteCountsResponse",
    "LogRunDeleteBlockerResponse",
    "LogRunDeletePlanResponse",
    "LogRunDeleteResponse",
    "LogTagsRequest",
    "LogRunTagsResponse",
    "LogTagsResponse",
    "LogScalarsRequest",
    "LogScalarSeriesResponse",
    "LogScalarsResponse",
]


class ApiCompatibilityImportTests(unittest.TestCase):
    def test_api_package_reexports_public_asgi_symbols(self) -> None:
        api = importlib.import_module("workbench.backend.api")
        main = importlib.import_module("workbench.backend.main")

        self.assertEqual(api.COMPATIBILITY_STATUS, "stable")
        self.assertEqual(api.REPLACEMENT_IMPORT, "workbench.backend.main")
        for name in ("app", "create_app", "WorkbenchApiSettings"):
            with self.subTest(name=name):
                self.assertIs(getattr(api, name), getattr(main, name))

    def test_settings_package_reexports_stable_settings_symbols(self) -> None:
        settings = importlib.import_module("workbench.backend.settings")
        config = importlib.import_module("workbench.backend.core.config")

        self.assertEqual(settings.COMPATIBILITY_STATUS, "stable")
        self.assertEqual(settings.REPLACEMENT_IMPORT, "workbench.backend.core.config")
        for name in (
            "LOCAL_FRONTEND_ORIGINS",
            "WorkbenchApiSettings",
            "get_workbench_api_settings",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(settings, name), getattr(config, name))

    def test_removed_legacy_route_and_empty_model_packages_stay_absent(self) -> None:
        for module_name in (
            "workbench.backend.routes.config_snapshots",
            "workbench.backend.routes.inspect",
            "workbench.backend.routes.logs",
            "workbench.backend.routes.models",
            "workbench.backend.routes.training",
            "workbench.backend.models",
        ):
            with self.subTest(module=module_name), self.assertRaises(
                ModuleNotFoundError
            ):
                importlib.import_module(module_name)


class SchemaCompatibilityImportTests(unittest.TestCase):
    def test_schemas_package_reexports_public_schema_symbols(self) -> None:
        schemas = importlib.import_module("workbench.backend.schemas")

        self.assertEqual(schemas.__all__, EXPECTED_SCHEMA_EXPORTS)
        for name in EXPECTED_SCHEMA_EXPORTS:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(schemas, name))


class InspectorCompatibilityRemovalTests(unittest.TestCase):
    def test_obsolete_inspector_forwarders_stay_absent(self) -> None:
        for module_name in (
            "workbench.backend.inspector.checkpoint_shapes",
            "workbench.backend.inspector.discovery",
            "workbench.backend.inspector.errors",
            "workbench.backend.inspector.graph",
            "workbench.backend.inspector.schema",
            "workbench.backend.inspector.service",
        ):
            with self.subTest(module=module_name), self.assertRaises(
                ModuleNotFoundError
            ):
                importlib.import_module(module_name)


class ExtensionPointImportTests(unittest.TestCase):
    def test_security_module_remains_importable(self) -> None:
        module_name = "workbench.backend.core.security"

        module = importlib.import_module(module_name)

        self.assertEqual(module.__name__, module_name)

    def test_empty_database_extension_point_is_absent(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("workbench.backend.db.session")


class ProjectCliBoundaryTests(unittest.TestCase):
    def test_workbench_cli_compatibility_module_is_removed(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("workbench.backend.cli")

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
