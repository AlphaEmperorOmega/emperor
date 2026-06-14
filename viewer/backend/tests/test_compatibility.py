from __future__ import annotations

import importlib
import os
import unittest
from pathlib import Path
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
    "OperationGraphNodeResponse",
    "OperationGraphEdgeResponse",
    "OperationGraphResponse",
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
    "LogCheckpointsRequest",
    "LogCheckpointResponse",
    "LogCheckpointsResponse",
    "LogRunArtifactResponse",
    "LogRunArtifactsResponse",
    "LogExperimentResponse",
    "LogExperimentsResponse",
    "LogExperimentDeleteResponse",
    "LogMediaRequest",
    "LogImageSummaryResponse",
    "LogTextSummaryResponse",
    "LogMediaResponse",
    "LogRunDeleteFiltersRequest",
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
        api = importlib.import_module("viewer.backend.api")
        main = importlib.import_module("viewer.backend.main")

        self.assertEqual(api.COMPATIBILITY_STATUS, "stable")
        self.assertEqual(api.REPLACEMENT_IMPORT, "viewer.backend.main")
        for name in ("app", "create_app", "ViewerApiSettings"):
            with self.subTest(name=name):
                self.assertIs(getattr(api, name), getattr(main, name))

    def test_settings_package_reexports_stable_settings_symbols(self) -> None:
        settings = importlib.import_module("viewer.backend.settings")
        config = importlib.import_module("viewer.backend.core.config")

        self.assertEqual(settings.COMPATIBILITY_STATUS, "stable")
        self.assertEqual(settings.REPLACEMENT_IMPORT, "viewer.backend.core.config")
        for name in (
            "LOCAL_FRONTEND_ORIGINS",
            "ViewerApiSettings",
            "get_viewer_api_settings",
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
                legacy = importlib.import_module(f"viewer.backend.routes.{legacy_name}")
                canonical = importlib.import_module(
                    f"viewer.backend.api.v1.routers.{canonical_name}"
                )

                self.assertEqual(legacy.COMPATIBILITY_STATUS, "deprecated")
                self.assertEqual(
                    legacy.REPLACEMENT_IMPORT,
                    f"viewer.backend.api.v1.routers.{canonical_name}",
                )
                self.assertTrue(legacy.REMOVAL_CONDITION)
                self.assertIs(legacy.router, canonical.router)


class SchemaCompatibilityImportTests(unittest.TestCase):
    def test_schemas_package_reexports_public_schema_symbols(self) -> None:
        schemas = importlib.import_module("viewer.backend.schemas")

        self.assertEqual(schemas.__all__, EXPECTED_SCHEMA_EXPORTS)
        for name in EXPECTED_SCHEMA_EXPORTS:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(schemas, name))


class InspectorCompatibilityImportTests(unittest.TestCase):
    def test_inspector_package_reexports_documented_symbols(self) -> None:
        inspector = importlib.import_module("viewer.backend.inspector")
        discovery = importlib.import_module("viewer.backend.inspector.discovery")
        schema = importlib.import_module("viewer.backend.inspector.schema")
        service = importlib.import_module("viewer.backend.inspector.service")
        graph = importlib.import_module("viewer.backend.inspector.graph")

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
    def test_no_auth_and_no_database_extension_points_are_importable(self) -> None:
        module_names = (
            "viewer.backend.core.security",
            "viewer.backend.db.session",
        )

        for module_name in module_names:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

    def test_database_session_extension_point_remains_noop(self) -> None:
        session = importlib.import_module("viewer.backend.db.session")
        public_names = [name for name in vars(session) if not name.startswith("_")]
        forbidden_database_names = {
            "Session",
            "SessionLocal",
            "engine",
            "get_db",
            "get_session",
            "sessionmaker",
        }
        forbidden_database_imports = (
            "asyncpg",
            "databases",
            "psycopg",
            "psycopg2",
            "sqlalchemy",
            "sqlmodel",
        )
        session_source = Path(str(session.__file__)).read_text(encoding="utf-8")

        self.assertEqual(public_names, [])
        self.assertTrue(forbidden_database_names.isdisjoint(vars(session)))
        for import_name in forbidden_database_imports:
            with self.subTest(import_name=import_name):
                self.assertNotIn(import_name, session_source)


class CliCompatibilityImportTests(unittest.TestCase):
    def test_cli_module_exposes_callable_main(self) -> None:
        cli = importlib.import_module("viewer.backend.cli")

        self.assertTrue(callable(cli.main))

    def test_cli_main_reaches_parser_without_inspecting_model(self) -> None:
        cli = importlib.import_module("viewer.backend.cli")
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
    def test_bert_linear_and_vit_linear_no_deleted_transformer_utils_imports(
        self,
    ) -> None:
        for module_name in (
            "models.transformer_encoder.bert_linear.presets",
            "models.transformer_encoder.vit_linear.presets",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, "ExperimentPresets"))


if __name__ == "__main__":
    unittest.main()
