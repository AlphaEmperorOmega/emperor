from __future__ import annotations

import importlib
import os
import sys
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
    "TrainingRunPlanCreateRequest",
    "TrainingSearchResponse",
    "TrainingSearchRequest",
    "TrainingRunChangeResponse",
    "SubmittedTrainingRunChangeRequest",
    "TrainingRunResponse",
    "SubmittedTrainingRunRequest",
    "TrainingRunPlanSummaryResponse",
    "SubmittedTrainingRunPlanSummaryRequest",
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

    def test_legacy_route_shims_reexport_canonical_routers(self) -> None:
        route_modules = {
            "config_snapshots": "config_snapshots",
            "inspect": "inspection",
            "logs": "logs",
            "models": "models",
            "training": "training",
        }

        for legacy_name, canonical_name in route_modules.items():
            with self.subTest(route=legacy_name):
                legacy = importlib.import_module(
                    f"workbench.backend.routes.{legacy_name}"
                )
                canonical = importlib.import_module(
                    f"workbench.backend.api.v1.routers.{canonical_name}"
                )

                self.assertEqual(legacy.COMPATIBILITY_STATUS, "deprecated")
                self.assertEqual(
                    legacy.REPLACEMENT_IMPORT,
                    f"workbench.backend.api.v1.routers.{canonical_name}",
                )
                self.assertTrue(legacy.REMOVAL_CONDITION)
                self.assertIs(legacy.router, canonical.router)


class SchemaCompatibilityImportTests(unittest.TestCase):
    def test_schemas_package_reexports_public_schema_symbols(self) -> None:
        schemas = importlib.import_module("workbench.backend.schemas")

        self.assertEqual(schemas.__all__, EXPECTED_SCHEMA_EXPORTS)
        for name in EXPECTED_SCHEMA_EXPORTS:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(schemas, name))


class InspectorCompatibilityImportTests(unittest.TestCase):
    def test_inspector_package_reexports_documented_symbols(self) -> None:
        inspector = importlib.import_module("workbench.backend.inspector")
        discovery = importlib.import_module("workbench.backend.inspector.discovery")
        schema = importlib.import_module("workbench.backend.inspector.schema")
        service = importlib.import_module("workbench.backend.inspector.service")
        graph = importlib.import_module("workbench.backend.inspector.graph")

        expected_exports = {
            "ModelParts": discovery.ModelParts,
            "config_schema": schema.config_schema,
            "discover_models": discovery.discover_models,
            "inspect_model": service.inspect_model,
            "list_model_presets": discovery.list_model_presets,
            "load_model_parts": discovery.load_model_parts,
            "serialize_graph": graph.serialize_graph,
        }

        self.assertEqual(inspector.__all__, list(expected_exports))
        for name, canonical_object in expected_exports.items():
            with self.subTest(name=name):
                self.assertIs(getattr(inspector, name), canonical_object)


class ExtensionPointImportTests(unittest.TestCase):
    def test_security_module_remains_importable(self) -> None:
        module_name = "workbench.backend.core.security"

        module = importlib.import_module(module_name)

        self.assertEqual(module.__name__, module_name)

    def test_empty_database_extension_point_is_absent(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("workbench.backend.db.session")


class CliCompatibilityImportTests(unittest.TestCase):
    def test_cli_maps_broken_package_parser_imports_to_clean_error(self) -> None:
        from emperor.model_packages import ModelPackage

        from workbench.backend.inspector.errors import InspectorError

        cli = importlib.import_module("workbench.backend.cli")
        package = ModelPackage(
            "broken",
            "missing",
            "models.__inspection_missing__",
        )

        with (
            patch.object(
                sys,
                "argv",
                [
                    "cli",
                    "--model-type",
                    "broken",
                    "--model",
                    "missing",
                    "--preset",
                    "baseline",
                ],
            ),
            patch.object(cli, "model_id_from_parts", return_value="broken/missing"),
            patch.object(cli, "model_package", return_value=package),
            self.assertRaisesRegex(
                InspectorError,
                "Failed to import model package 'broken/missing'",
            ),
        ):
            cli._parse_args()

    def test_cli_module_exposes_callable_main(self) -> None:
        cli = importlib.import_module("workbench.backend.cli")

        self.assertTrue(callable(cli.main))

    def test_cli_main_reaches_parser_without_inspecting_model(self) -> None:
        cli = importlib.import_module("workbench.backend.cli")
        parser_exit = SystemExit("parser reached")

        with (
            patch.object(cli, "_parse_args", side_effect=parser_exit) as parse_args,
            patch.object(cli, "inspect_model") as inspect_model,
        ):
            with self.assertRaises(SystemExit) as raised:
                cli.main()

        self.assertIs(raised.exception, parser_exit)
        parse_args.assert_called_once_with()
        inspect_model.assert_not_called()


class ModelPackageCompatibilityTests(unittest.TestCase):
    def test_old_vit_transformer_encoder_package_import_is_unsupported(self) -> None:
        old_package = "models.transformer_encoder." + "vit" + "_linear.presets"

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module(old_package)


if __name__ == "__main__":
    unittest.main()
