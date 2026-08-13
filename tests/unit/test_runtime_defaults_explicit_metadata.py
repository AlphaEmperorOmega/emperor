from __future__ import annotations

import inspect
import unittest
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, cast
from unittest.mock import patch

from model_runtime.inspection import (
    InspectionError,
    configuration_schema,
    search_space_schema,
)
from model_runtime.packages import (
    ModelMetadata,
    ModelPackage,
    RuntimeDefaultsError,
    configuration_field_metadata,
)
from model_runtime.runs import PlanningBudget, RunRequest, SearchSpec, plan_runs
from models.catalog import discover_model_packages, model_package
from models.linears.linear import dataset_options as linears_linear_datasets
from models.linears.linear import monitor_options as linears_linear_monitors


class _MetadataAdapter:
    def __init__(self, package: ModelPackage, metadata: ModelMetadata) -> None:
        self._package = package
        self._metadata = metadata

    def load_metadata(self) -> ModelMetadata:
        return self._metadata

    def load_runtime_options_type(self) -> type[Any]:
        return self._package.runtime_options_type

    def bind_runtime_defaults(self, values: Mapping[str, object] | None) -> Any:
        return self._package.bind_runtime_defaults(values)

    def load_preset_type(self) -> type[Any]:
        return self._package.preset_type

    def load_presets(self) -> Any:
        return self._package.presets

    def build_configuration(
        self,
        presets: Any,
        preset: Any,
        dataset: type[Any],
        **kwargs: Any,
    ) -> Any:
        del presets
        return self._package.build_configuration(preset, dataset, **kwargs)

    def build_model(self, configuration: Any) -> Any:
        return self._package.build_model(configuration)

    def build_experiment(
        self,
        preset: Any,
        *,
        experiment_task: Any,
        model_package: ModelPackage,
        run_artifacts: Any,
    ) -> Any:
        del model_package
        return self._package.build_experiment(
            preset,
            experiment_task=experiment_task,
            run_artifacts=run_artifacts,
        )


def _linears_linear() -> ModelPackage:
    package = model_package("linears/linear")
    if package is None:
        raise AssertionError("Expected the linears/linear Model Package.")
    return package


def _plain_metadata(
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {key: dict(entry) for key, entry in metadata.items()}


def _source_less_package() -> ModelPackage:
    from model_runtime.packages import RuntimeDefaultsSection

    base = _linears_linear()
    config = ModuleType("synthetic.source_less.config")
    config.ALPHA = 1
    config.BETA = 2
    config.__annotations__ = {"ALPHA": int, "BETA": int}
    search = ModuleType("synthetic.source_less.search_space")
    search.SEARCH_SPACE_BETA = [20, 21]
    search.SEARCH_SPACE_ALPHA = [10, 11]
    search.__annotations__ = {
        "SEARCH_SPACE_ALPHA": list[int],
        "SEARCH_SPACE_BETA": list[int],
    }
    metadata = ModelMetadata(
        base.identity,
        config,
        linears_linear_datasets,
        linears_linear_monitors,
        search,
        configuration_metadata_sections=(
            RuntimeDefaultsSection(
                ("Second",),
                (("BETA", 20, (20,)),),
            ),
            RuntimeDefaultsSection(
                ("First",),
                (("ALPHA", 10, (10,)),),
            ),
        ),
        search_metadata_sections=(
            RuntimeDefaultsSection((), (("SEARCH_SPACE_BETA", 20, (20,)),)),
            RuntimeDefaultsSection((), (("SEARCH_SPACE_ALPHA", 10, (10,)),)),
        ),
    )
    return ModelPackage(base.identity, _MetadataAdapter(base, metadata))


def _package_with_declarations(
    configuration_sections: Any,
    search_sections: Any,
    *,
    config_members: Mapping[str, Any] | None = None,
    search_members: Mapping[str, Any] | None = None,
) -> ModelPackage:
    base = _linears_linear()
    config = ModuleType("synthetic.validation.config")
    for key, value in (config_members or {"VALUE": 1}).items():
        setattr(config, key, value)
    search = ModuleType("synthetic.validation.search_space")
    for key, value in (search_members or {"SEARCH_SPACE_VALUE": [1]}).items():
        setattr(search, key, value)
    metadata = ModelMetadata(
        base.identity,
        config,
        linears_linear_datasets,
        linears_linear_monitors,
        search,
        configuration_metadata_sections=configuration_sections,
        search_metadata_sections=search_sections,
    )
    return ModelPackage(base.identity, _MetadataAdapter(base, metadata))


class RuntimeDefaultsExplicitMetadataTests(unittest.TestCase):
    def test_source_less_metadata_drives_schema_and_run_planning(self) -> None:
        package = _source_less_package()

        with patch(
            "model_runtime.packages.metadata.configuration_field_metadata",
            side_effect=AssertionError("source parser must not run"),
        ) as source_parser:
            spec = package.runtime_defaults_spec
            configuration = configuration_schema(package)
            search = search_space_schema(package)
            override_plan = plan_runs(
                package,
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    overrides={"beta": 8, "alpha": 9},
                ),
            )
            search_plan = plan_runs(
                package,
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    search=SearchSpec(mode="grid"),
                ),
                budget=PlanningBudget.unlimited(),
            )

        source_parser.assert_not_called()
        self.assertEqual(tuple(spec.configuration_metadata), ("BETA", "ALPHA"))
        self.assertEqual(
            _plain_metadata(spec.configuration_metadata),
            {
                "BETA": {
                    "line": 20,
                    "sortKey": [20],
                    "section": "Second",
                    "sectionPath": ["Second"],
                },
                "ALPHA": {
                    "line": 10,
                    "sortKey": [10],
                    "section": "First",
                    "sectionPath": ["First"],
                },
            },
        )
        self.assertEqual(
            [(field.key, field.section_path) for field in configuration.fields],
            [("ALPHA", ("First",)), ("BETA", ("Second",))],
        )
        self.assertEqual(
            [axis.search_key for axis in search.axes],
            ["SEARCH_SPACE_ALPHA", "SEARCH_SPACE_BETA"],
        )
        self.assertEqual(
            [parameter.key for parameter in override_plan.runs[0].parameters],
            ["ALPHA", "BETA"],
        )
        self.assertEqual(
            [parameter.key for parameter in search_plan.runs[0].parameters],
            ["ALPHA", "BETA"],
        )

    def test_every_catalog_declaration_exactly_matches_legacy_metadata(self) -> None:
        for catalog_package in discover_model_packages():
            with self.subTest(package=catalog_package.catalog_key):
                metadata = catalog_package._adapter.load_metadata()
                legacy_configuration = configuration_field_metadata(
                    metadata._runtime_defaults_source
                )
                legacy_search = configuration_field_metadata(
                    metadata._search_space_source,
                    include_search_space=True,
                )
                package = ModelPackage(
                    catalog_package.identity,
                    _MetadataAdapter(catalog_package, metadata),
                    catalog_package.inspection_construction_limits,
                )
                with patch(
                    "model_runtime.packages.metadata.configuration_field_metadata",
                    side_effect=AssertionError("catalog parser must not run"),
                ):
                    spec = package.runtime_defaults_spec

                self.assertEqual(
                    _plain_metadata(spec.configuration_metadata),
                    legacy_configuration,
                )
                self.assertEqual(
                    _plain_metadata(spec.search_metadata),
                    legacy_search,
                )

    def test_declarations_are_authoritative_and_all_or_none(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        base = _linears_linear()
        config = ModuleType("synthetic.authoritative.config")
        config.VALUE = 1
        search = ModuleType("synthetic.authoritative.search_space")
        search.SEARCH_SPACE_VALUE = [1]
        common = (
            base.identity,
            config,
            linears_linear_datasets,
            linears_linear_monitors,
            search,
        )
        with self.assertRaisesRegex(
            ValueError,
            "configuration and search metadata declarations together",
        ):
            ModelMetadata(
                *common,
                configuration_metadata_sections=(
                    RuntimeDefaultsSection(
                        ("General",),
                        (("VALUE", 1, (1,)),),
                    ),
                ),
            )

        metadata = ModelMetadata(
            *common,
            configuration_metadata_sections=(),
            search_metadata_sections=(),
        )
        package = ModelPackage(base.identity, _MetadataAdapter(base, metadata))
        with patch(
            "model_runtime.packages.metadata.configuration_field_metadata",
            side_effect=AssertionError("authoritative declarations cannot fallback"),
        ) as source_parser:
            with self.assertRaisesRegex(RuntimeDefaultsError, "missing.*VALUE"):
                _ = package.runtime_defaults_spec
        source_parser.assert_not_called()

        package = ModelPackage(base.identity, _MetadataAdapter(base, metadata))
        with self.assertRaises(InspectionError) as caught:
            configuration_schema(package)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_declaration_inputs_and_projections_are_defensive(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        path = ["Original"]
        sort_key = [7, 0]
        fields = [("VALUE", 7, sort_key)]
        section = RuntimeDefaultsSection(path, fields)
        path.append("Mutated")
        sort_key.append(999)
        fields.append(("OTHER", 8, [8]))

        self.assertEqual(section.path, ("Original",))
        self.assertEqual(section.fields, (("VALUE", 7, (7, 0)),))
        with self.assertRaises(FrozenInstanceError):
            cast(Any, section).path = ()

        config_section = [section]
        search_section = [
            RuntimeDefaultsSection((), (("SEARCH_SPACE_VALUE", 8, (8,)),))
        ]
        base = _linears_linear()
        config = ModuleType("synthetic.snapshot.config")
        config.VALUE = 1
        search = ModuleType("synthetic.snapshot.search_space")
        search.SEARCH_SPACE_VALUE = [1]
        metadata = ModelMetadata(
            base.identity,
            config,
            linears_linear_datasets,
            linears_linear_monitors,
            search,
            configuration_metadata_sections=config_section,
            search_metadata_sections=search_section,
        )
        config_section.clear()
        search_section.clear()
        package = ModelPackage(base.identity, _MetadataAdapter(base, metadata))
        spec = package.runtime_defaults_spec

        first = spec.configuration_metadata["VALUE"]
        cast(list[int], first["sortKey"]).append(999)
        cast(list[str], first["sectionPath"]).append("Mutated")
        second = spec.configuration_metadata["VALUE"]
        self.assertEqual(second["sortKey"], [7, 0])
        self.assertEqual(second["sectionPath"], ["Original"])

    def test_section_declaration_rejects_malformed_values(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        constructor = cast(Any, RuntimeDefaultsSection)
        cases = (
            (("General", (("VALUE", 1, (1,)),)), "path must be a sequence"),
            ((("",), (("VALUE", 1, (1,)),)), "path items"),
            (((" General",), (("VALUE", 1, (1,)),)), "path items"),
            ((("General",), "VALUE"), "fields must be a sequence"),
            ((("General",), ()), "require fields"),
            ((("General",), (("VALUE", 1),)), "require key, line, and sort key"),
            ((("General",), (("", 1, (1,)),)), "field keys"),
            ((("General",), (("VALUE", True, (1,)),)), "lines"),
            ((("General",), (("VALUE", 0, (1,)),)), "lines"),
            ((("General",), (("VALUE", 1, ()),)), "sort keys"),
            ((("General",), (("VALUE", 1, (-1,)),)), "sort keys"),
            ((("General",), (("VALUE", 1, (True,)),)), "sort keys"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    constructor(*arguments)

    def test_explicit_declaration_validation_is_complete_and_stable(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        config = RuntimeDefaultsSection(
            ("General",),
            (("VALUE", 1, (1,)),),
        )
        search = RuntimeDefaultsSection(
            (),
            (("SEARCH_SPACE_VALUE", 1, (1,)),),
        )
        cases = (
            (
                (
                    config,
                    RuntimeDefaultsSection(
                        ("Repeated",),
                        (("VALUE", 2, (2,)),),
                    ),
                ),
                (search,),
                {},
                "duplicate field 'VALUE'",
            ),
            (
                (
                    RuntimeDefaultsSection(
                        ("General",),
                        (("UNKNOWN", 1, (1,)),),
                    ),
                ),
                (search,),
                {},
                "unknown field 'UNKNOWN'",
            ),
            ((), (search,), {}, "missing required fields: VALUE"),
            (
                (RuntimeDefaultsSection((), (("VALUE", 1, (1,)),)),),
                (search,),
                {},
                "Configuration metadata sections require a non-empty path",
            ),
            ((config,), (), {}, "missing required fields: SEARCH_SPACE_VALUE"),
            (
                (config,),
                (
                    search,
                    RuntimeDefaultsSection(
                        (),
                        (("SEARCH_SPACE_UNUSED", 2, (2,)),),
                    ),
                ),
                {
                    "search_members": {
                        "SEARCH_SPACE_VALUE": [1],
                        "SEARCH_SPACE_UNUSED": 1,
                    }
                },
                "search metadata declares fields without values: SEARCH_SPACE_UNUSED",
            ),
        )
        for configuration, search_declaration, kwargs, message in cases:
            with self.subTest(message=message):
                package = _package_with_declarations(
                    configuration,
                    search_declaration,
                    **kwargs,
                )
                with self.assertRaisesRegex(RuntimeDefaultsError, message) as direct:
                    _ = package.runtime_defaults_spec
                self.assertIsInstance(direct.exception.__cause__, ValueError)

                package = _package_with_declarations(
                    configuration,
                    search_declaration,
                    **kwargs,
                )
                with self.assertRaisesRegex(InspectionError, message) as inspected:
                    configuration_schema(package)
                self.assertIsInstance(inspected.exception.__cause__, ValueError)

    def test_configuration_declaration_preserves_supported_extras(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection

        package = _package_with_declarations(
            (
                RuntimeDefaultsSection(
                    ("General",),
                    (
                        ("VALUE", 1, (1,)),
                        ("PRESENTATION_ONLY", 2, (2,)),
                    ),
                ),
            ),
            (
                RuntimeDefaultsSection(
                    (),
                    (("SEARCH_SPACE_VALUE", 1, (1,)),),
                ),
            ),
            config_members={"VALUE": 1, "PRESENTATION_ONLY": ["metadata"]},
        )

        self.assertEqual(
            tuple(package.runtime_defaults_spec.configuration_metadata),
            ("VALUE", "PRESENTATION_ONLY"),
        )

    def test_public_constructor_and_facade_compatibility(self) -> None:
        from model_runtime.packages import RuntimeDefaultsSection
        from model_runtime.packages.configuration_metadata import (
            RuntimeDefaultsSection as OwnerRuntimeDefaultsSection,
        )

        self.assertIs(RuntimeDefaultsSection, OwnerRuntimeDefaultsSection)
        parameters = inspect.signature(ModelMetadata).parameters
        self.assertEqual(
            tuple(parameters),
            (
                "identity",
                "runtime_defaults",
                "dataset_options",
                "monitor_options_source",
                "search_space",
                "configuration_metadata_sections",
                "search_metadata_sections",
            ),
        )
        for name in tuple(parameters)[:5]:
            self.assertIs(
                parameters[name].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        for name in tuple(parameters)[5:]:
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameters[name].default)

    def test_authoring_generator_bootstraps_a_missing_declaration(self) -> None:
        from tools import generate_model_package_inspection_metadata as generator

        package = _linears_linear()
        with TemporaryDirectory() as temporary_directory:
            models_root = Path(temporary_directory)
            target = models_root / "linears" / "linear" / "_inspection_metadata.py"
            target.parent.mkdir(parents=True)
            with (
                patch.object(generator, "_MODELS_ROOT", models_root),
                patch.object(
                    generator,
                    "discover_model_packages",
                    return_value=[package],
                ),
                patch.object(
                    type(package._adapter),
                    "load_metadata",
                    side_effect=AssertionError("generator cannot load declarations"),
                ),
            ):
                self.assertEqual(generator._synchronize(check=True), (target,))
                self.assertFalse(target.exists())
                self.assertEqual(generator._synchronize(check=False), (target,))
                self.assertTrue(target.is_file())
                self.assertEqual(generator._synchronize(check=True), ())


if __name__ == "__main__":
    unittest.main()
