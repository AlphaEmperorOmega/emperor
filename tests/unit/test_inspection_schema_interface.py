from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from model_runtime.inspection import (
    ConfigurationSchema,
    InspectionError,
    InspectionRequest,
    SearchSpace,
    canonicalize_overrides,
    configuration_schema,
    parse_overrides,
    reject_locked_overrides,
    search_space_schema,
    serialize_overrides,
    supported_config_keys,
    validate_configuration,
)
from model_runtime.packages import ModelIdentity, ModelPackage
from models.catalog import model_package


class _BrokenPackageAdapter:
    @staticmethod
    def _missing(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'models.__inspection_missing__'")

    load_metadata = _missing
    load_runtime_options_type = _missing
    bind_runtime_defaults = _missing
    load_preset_type = _missing
    load_presets = _missing
    build_configuration = _missing
    build_model = _missing
    build_experiment = _missing


def _broken_package() -> ModelPackage:
    return ModelPackage(
        ModelIdentity("broken", "missing"),
        _BrokenPackageAdapter(),
    )


class InspectionSchemaInterfaceTests(unittest.TestCase):
    def test_broken_package_override_failures_are_transport_neutral(self) -> None:
        package = _broken_package()
        calls = (
            lambda: supported_config_keys(package),
            lambda: parse_overrides(package, {"HIDDEN_DIM": "1"}),
            lambda: canonicalize_overrides(package, {"HIDDEN_DIM": "1"}),
            lambda: serialize_overrides(package, {"HIDDEN_DIM": "1"}),
            lambda: reject_locked_overrides(package, "baseline", {}),
        )

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    InspectionError,
                    "Failed to import model package 'broken/missing'",
                ):
                    call()

    def test_broken_package_schema_failures_are_transport_neutral(self) -> None:
        package = _broken_package()

        for inspect_schema in (configuration_schema, search_space_schema):
            with self.subTest(call=inspect_schema.__name__):
                with self.assertRaisesRegex(
                    InspectionError,
                    "Failed to import model package 'broken/missing'",
                ):
                    inspect_schema(package)

    def test_selected_package_produces_frozen_configuration_records(self) -> None:
        package = model_package("linears/linear")
        assert package is not None

        schema = configuration_schema(package, preset="gating")

        self.assertIsInstance(schema, ConfigurationSchema)
        self.assertEqual(schema.identity, package.identity)
        self.assertIsInstance(schema.fields, tuple)
        fields = {field.key: field for field in schema.fields}
        self.assertEqual(fields["HIDDEN_DIM"].value_type, "int")
        self.assertEqual(fields["HIDDEN_DIM"].default, 32)
        self.assertEqual(fields["HIDDEN_DIM"].section_path, ("Global",))
        self.assertEqual(
            fields["HIDDEN_DIM"].maximum,
            package.inspection_construction_limits.maximum_hidden_dimension,
        )
        self.assertEqual(
            fields["STACK_NUM_LAYERS"].maximum,
            package.inspection_construction_limits.maximum_layer_count,
        )
        self.assertTrue(fields["STACK_GATE_FLAG"].locked)
        self.assertEqual(fields["STACK_GATE_FLAG"].locked_value, True)

    def test_selected_adaptive_package_produces_search_metadata_records(self) -> None:
        package = model_package("linears/linear_adaptive")
        assert package is not None

        search = search_space_schema(
            package,
            preset="baseline",
            presets=("full-stack", "dual-weight-gating"),
        )

        self.assertIsInstance(search, SearchSpace)
        self.assertIsInstance(search.axes, tuple)
        axes = {axis.key: axis for axis in search.axes}
        self.assertEqual(axes["HIDDEN_DIM"].value_type, "int")
        self.assertEqual(axes["WEIGHT_OPTION"].search_key, "SEARCH_SPACE_WEIGHT_OPTION")
        self.assertTrue(axes["WEIGHT_OPTION"].locked)
        self.assertIn("FULL_STACK", axes["WEIGHT_OPTION"].locked_by_presets)

    def test_override_parsing_uses_runtime_default_types_and_model_parameters(
        self,
    ) -> None:
        package = model_package("linears/linear")
        assert package is not None

        parsed = parse_overrides(
            package,
            {"hidden-dim": "128", "stack_gate_flag": "true"},
        )

        self.assertEqual(
            dict(parsed.values),
            {"hidden_dim": 128, "stack_gate_flag": True},
        )

    def test_invalid_and_locked_overrides_raise_transport_neutral_error(self) -> None:
        package = model_package("linears/linear")
        assert package is not None

        with self.assertRaisesRegex(InspectionError, "Unknown override"):
            parse_overrides(package, {"NO_SUCH_FIELD": "1"})
        with self.assertRaisesRegex(InspectionError, "locked fields"):
            parse_overrides(
                package,
                {"stack_gate_flag": "false"},
                preset="gating",
            )

    def test_expert_lock_aliases_are_canonical_across_inspection(self) -> None:
        package = model_package("transformer/expert_linear")
        assert package is not None

        fields = {
            field.key: field
            for field in configuration_schema(package, preset="top1-switch-aux").fields
        }
        axes = {
            axis.key: axis
            for axis in search_space_schema(
                package,
                preset="top1-switch-aux",
            ).axes
        }

        self.assertTrue(fields["TOP_K"].locked)
        self.assertEqual(fields["TOP_K"].locked_value, 1)
        self.assertTrue(axes["TOP_K"].locked)
        self.assertEqual(axes["TOP_K"].locked_value, 1)
        with self.assertRaisesRegex(InspectionError, "locked fields: top_k"):
            parse_overrides(
                package,
                {"top_k": "1"},
                preset="top1-switch-aux",
            )

    def test_configuration_validation_builds_without_constructing_a_graph(self) -> None:
        package = model_package("linears/linear_adaptive")
        assert package is not None

        validate_configuration(
            package,
            InspectionRequest(
                preset="baseline",
                overrides={
                    "weight_option_flag": "true",
                    "weight_option": "SingleModelDynamicWeightConfig",
                },
            ),
        )
        with self.assertRaisesRegex(InspectionError, "weight_option.*must be set"):
            validate_configuration(
                package,
                InspectionRequest(
                    preset="baseline",
                    overrides={"weight_option_flag": "true"},
                ),
            )


if __name__ == "__main__":
    unittest.main()
