from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.layers import (
    AdditiveResidualConfig,
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
    WeightedResidualConfig,
)
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
from model_runtime.inspection.runtime_defaults import runtime_defaults_spec
from model_runtime.packages import ModelIdentity, ModelPackage
from models.catalog import model_package
from models.transformer.linear import config as transformer_linear_config


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


def _fresh_package(catalog_key: str) -> ModelPackage:
    catalog_package = model_package(catalog_key)
    assert catalog_package is not None
    return ModelPackage(
        catalog_package.identity,
        catalog_package._adapter,
        catalog_package.inspection_construction_limits,
    )


class InspectionSchemaInterfaceTests(unittest.TestCase):
    def test_runtime_defaults_policy_is_compiled_once_per_selected_package(
        self,
    ) -> None:
        catalog_package = model_package("linears/linear")
        assert catalog_package is not None
        package = ModelPackage(
            catalog_package.identity,
            catalog_package._adapter,
            catalog_package.inspection_construction_limits,
        )
        original = ModelPackage.configuration_field_metadata

        with patch.object(
            ModelPackage,
            "configuration_field_metadata",
            autospec=True,
            side_effect=original,
        ) as metadata:
            first = runtime_defaults_spec(package)
            second = runtime_defaults_spec(package)
            configuration_schema(package)
            parse_overrides(package, {"hidden-dim": "64"})

        self.assertIs(first, second)
        self.assertEqual(metadata.call_count, 2)

    def test_runtime_defaults_cache_uses_selected_package_identity(self) -> None:
        catalog_package = model_package("linears/linear")
        assert catalog_package is not None
        first_package = ModelPackage(
            catalog_package.identity,
            catalog_package._adapter,
            catalog_package.inspection_construction_limits,
        )
        second_package = ModelPackage(
            catalog_package.identity,
            catalog_package._adapter,
            catalog_package.inspection_construction_limits,
        )

        first = runtime_defaults_spec(first_package)
        second = runtime_defaults_spec(second_package)

        self.assertIsNot(first, second)
        self.assertEqual(first.resolve_key("hidden-dim"), "HIDDEN_DIM")
        self.assertEqual(first.resolve_key("hidden_dim"), "HIDDEN_DIM")

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
        self.assertEqual(fields["HIDDEN_DIM"].applicable_when, ())

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

    def test_residual_override_uses_concrete_config_class_names(self) -> None:
        package = model_package("linears/linear")
        assert package is not None

        fields = {field.key: field for field in configuration_schema(package).fields}
        residual_selector = fields["STACK_RESIDUAL_CONNECTION_OPTION"]
        residual_model_flag = fields["STACK_RESIDUAL_MODEL_FLAG"]
        residual_stack_hidden_dim = fields["RESIDUAL_STACK_HIDDEN_DIM"]

        self.assertEqual(residual_selector.value_type, "class")
        self.assertIsNone(residual_selector.default)
        self.assertTrue(residual_selector.nullable)
        self.assertEqual(
            residual_selector.choices,
            (
                "AdditiveResidualConfig",
                "AttentionResidualConfig",
                "WeightedBlendResidualConfig",
                "WeightedResidualConfig",
            ),
        )
        self.assertEqual(residual_model_flag.value_type, "bool")
        self.assertIs(residual_model_flag.default, False)
        self.assertFalse(residual_model_flag.nullable)
        self.assertIn(
            "Residual Stack Options as a data-dependent coefficient model",
            residual_model_flag.description,
        )
        self.assertTupleEqual(residual_model_flag.applicable_when, ())
        self.assertEqual(
            residual_stack_hidden_dim.section_path,
            ("Residual Options", "Residual Stack Options"),
        )
        self.assertEqual(
            residual_stack_hidden_dim.flag,
            "--residual-stack-hidden-dim",
        )
        parsed = parse_overrides(
            package,
            {
                "stack_residual_connection_option": "WeightedResidualConfig",
                "stack_residual_model_flag": "true",
                "residual_stack_independent_flag": "true",
                "residual_stack_hidden_dim": "48",
            },
        )
        self.assertIs(
            parsed.values["stack_residual_connection_option"],
            WeightedResidualConfig,
        )
        self.assertIs(parsed.values["stack_residual_model_flag"], True)
        self.assertIs(parsed.values["residual_stack_independent_flag"], True)
        self.assertEqual(parsed.values["residual_stack_hidden_dim"], 48)
        self.assertEqual(
            serialize_overrides(
                package,
                {
                    "stack_residual_connection_option": "WeightedResidualConfig",
                    "stack_residual_model_flag": "true",
                    "residual_stack_independent_flag": "true",
                    "residual_stack_hidden_dim": "48",
                },
            ),
            {
                "STACK_RESIDUAL_CONNECTION_OPTION": "WeightedResidualConfig",
                "STACK_RESIDUAL_MODEL_FLAG": True,
                "RESIDUAL_STACK_INDEPENDENT_FLAG": True,
                "RESIDUAL_STACK_HIDDEN_DIM": 48,
            },
        )
        with self.assertRaisesRegex(InspectionError, "unknown config class 'RESIDUAL'"):
            parse_overrides(
                package,
                {"stack_residual_connection_option": "RESIDUAL"},
            )

    def test_transformer_residual_selectors_use_updated_public_configs(self) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        fields = {field.key: field for field in configuration_schema(package).fields}
        expected_choices = (
            "AdditiveResidualConfig",
            "AttentionResidualConfig",
            "WeightedBlendResidualConfig",
            "WeightedResidualConfig",
        )

        for key in (
            "STACK_RESIDUAL_CONNECTION_OPTION",
            "RECURRENT_RESIDUAL_CONNECTION_OPTION",
        ):
            with self.subTest(key=key):
                selector = fields[key]
                self.assertEqual(selector.value_type, "class")
                self.assertIsNone(selector.default)
                self.assertTrue(selector.nullable)
                self.assertEqual(selector.choices, expected_choices)

        parsed = parse_overrides(
            package,
            {
                "stack_residual_connection_option": "WeightedResidualConfig",
                "recurrent_residual_connection_option": "AdditiveResidualConfig",
            },
        )
        self.assertIs(
            parsed.values["stack_residual_connection_option"],
            WeightedResidualConfig,
        )
        self.assertIs(
            parsed.values["recurrent_residual_connection_option"],
            AdditiveResidualConfig,
        )

    def test_recurrent_override_uses_concrete_config_class_names(self) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        fields = {field.key: field for field in configuration_schema(package).fields}
        selector = fields["RECURRENT_COMPOSITION_OPTION"]

        self.assertEqual(selector.value_type, "class")
        self.assertEqual(selector.default, "RecurrentLayerConfig")
        self.assertFalse(selector.nullable)
        self.assertEqual(
            selector.choices,
            (
                "HierarchicalReasoningModelRecurrentConfig",
                "RecurrentLayerConfig",
                "TinyRecursiveModelRecurrentConfig",
            ),
        )
        parsed = parse_overrides(
            package,
            {"recurrent_composition_option": "TinyRecursiveModelRecurrentConfig"},
        )
        self.assertIs(
            parsed.values["recurrent_composition_option"],
            TinyRecursiveModelRecurrentConfig,
        )
        hierarchical_reasoning_model_parsed = parse_overrides(
            package,
            {
                "recurrent_composition_option": "HierarchicalReasoningModelRecurrentConfig"
            },
        )
        self.assertIs(
            hierarchical_reasoning_model_parsed.values["recurrent_composition_option"],
            HierarchicalReasoningModelRecurrentConfig,
        )
        self.assertEqual(
            serialize_overrides(
                package,
                {"recurrent_composition_option": "TinyRecursiveModelRecurrentConfig"},
            ),
            {"RECURRENT_COMPOSITION_OPTION": "TinyRecursiveModelRecurrentConfig"},
        )
        self.assertEqual(
            serialize_overrides(
                package,
                {
                    "recurrent_composition_option": "HierarchicalReasoningModelRecurrentConfig"
                },
            ),
            {
                "RECURRENT_COMPOSITION_OPTION": "HierarchicalReasoningModelRecurrentConfig"
            },
        )
        with self.assertRaisesRegex(InspectionError, "abstract"):
            parse_overrides(
                package,
                {"recurrent_composition_option": "RecurrentCompositionConfig"},
            )
        for retired_name in ("HRMRecurrentConfig", "TRMRecurrentConfig"):
            with self.subTest(retired_name=retired_name):
                with self.assertRaisesRegex(InspectionError, "unknown config class"):
                    parse_overrides(
                        package,
                        {"recurrent_composition_option": retired_name},
                    )

    def test_recurrent_reinjection_runtime_defaults_are_boolean_fields(self) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        fields = {field.key: field for field in configuration_schema(package).fields}
        keys = (
            "RECURRENT_REINJECT_ORIGINAL_HIDDEN_FLAG",
            "ATTN_RECURRENT_REINJECT_ORIGINAL_HIDDEN_FLAG",
            "FF_RECURRENT_REINJECT_ORIGINAL_HIDDEN_FLAG",
        )
        for key in keys:
            with self.subTest(key=key):
                field = fields[key]
                self.assertEqual(field.value_type, "bool")
                self.assertIs(field.default, False)
                self.assertFalse(field.nullable)
                self.assertEqual(field.choices, (True, False))

        parsed = parse_overrides(
            package,
            {
                "recurrent_reinject_original_hidden_flag": "true",
                "attn-recurrent-reinject-original-hidden-flag": "true",
                "ff_recurrent_reinject_original_hidden_flag": "false",
            },
        )
        self.assertEqual(
            dict(parsed.values),
            {
                "recurrent_reinject_original_hidden_flag": True,
                "attn_recurrent_reinject_original_hidden_flag": True,
                "ff_recurrent_reinject_original_hidden_flag": False,
            },
        )

    def test_recurrent_fields_expose_exact_sections_and_types(self) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        fields = {field.key: field for field in configuration_schema(package).fields}
        variants = (
            "HierarchicalReasoningModelRecurrentConfig",
            "RecurrentLayerConfig",
            "TinyRecursiveModelRecurrentConfig",
        )
        restricted_types = {
            "RECURRENT_MAX_STEPS": "int",
            "RECURRENT_REINJECT_ORIGINAL_HIDDEN_FLAG": "bool",
            "RECURRENT_LATENT_UPDATES_PER_ANSWER_UPDATE": "int",
            "RECURRENT_ANSWER_UPDATE_COUNT": "int",
            "RECURRENT_HIGH_CYCLES": "int",
            "RECURRENT_LOW_CYCLES": "int",
            "RECURRENT_INITIALIZATION_STANDARD_DEVIATION": "float",
        }
        scopes = (
            ("", ("Controller Options", "Recurrent Layer Options")),
            (
                "ATTN_",
                (
                    "Attention Options",
                    "Attention Projection Stack Options",
                    "Attention Projection Recurrent Layer Options",
                ),
            ),
            (
                "FF_",
                (
                    "Feed-Forward Stack Options",
                    "Feed-Forward Recurrent Layer Options",
                ),
            ),
        )

        for prefix, section_path in scopes:
            with self.subTest(prefix=prefix or "top-level"):
                selector_key = f"{prefix}RECURRENT_COMPOSITION_OPTION"
                selector = fields[selector_key]
                self.assertEqual(selector.section_path, section_path)
                self.assertEqual(selector.value_type, "class")
                self.assertEqual(selector.choices, variants)
                self.assertEqual(selector.applicable_when, ())

                for suffix, value_type in restricted_types.items():
                    target = fields[f"{prefix}{suffix}"]
                    self.assertEqual(target.section_path, section_path)
                    self.assertEqual(target.value_type, value_type)
                    self.assertEqual(target.applicable_when, ())

                shared = fields[f"{prefix}RECURRENT_NO_GRADIENT_TRANSITION_COUNT"]
                self.assertEqual(shared.value_type, "int")
                self.assertIsNone(shared.default)
                self.assertTrue(shared.nullable)
                self.assertEqual(shared.applicable_when, ())

        self.assertEqual(
            fields["RECURRENT_STACK_GATE_FLAG"].section_path,
            (
                "Controller Options",
                "Recurrent Layer Options",
                "Recurrent Gate Options",
            ),
        )
        self.assertEqual(
            fields["RECURRENT_STACK_HALTING_FLAG"].section_path,
            (
                "Controller Options",
                "Recurrent Layer Options",
                "Recurrent Halting Options",
            ),
        )
        self.assertEqual(fields["RECURRENT_STACK_GATE_FLAG"].applicable_when, ())
        self.assertEqual(
            fields["RECURRENT_RESIDUAL_CONNECTION_OPTION"].applicable_when,
            (),
        )
        self.assertEqual(
            fields["ATTN_RECURRENT_LAYER_NORM_POSITION"].applicable_when,
            (),
        )
        self.assertEqual(fields["ATTN_MEMORY_FLAG"].applicable_when, ())
        self.assertEqual(fields["FF_RECURRENT_STACK_HALTING_FLAG"].applicable_when, ())

    def test_top_level_recurrent_no_gradient_count_parses_int_and_none(self) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        count = parse_overrides(
            package,
            {"recurrent_no_gradient_transition_count": "7"},
        )
        disabled = parse_overrides(
            package,
            {"recurrent_no_gradient_transition_count": "null"},
        )

        self.assertEqual(count.values["recurrent_no_gradient_transition_count"], 7)
        self.assertIsNone(disabled.values["recurrent_no_gradient_transition_count"])

    def test_invalid_recurrent_applicability_metadata_is_rejected(self) -> None:
        cases = (
            (
                "unknown target key",
                {
                    "NOT_A_RUNTIME_DEFAULT": {
                        "RECURRENT_COMPOSITION_OPTION": (RecurrentLayerConfig,)
                    }
                },
            ),
            (
                "unknown controller key",
                {
                    "RECURRENT_MAX_STEPS": {
                        "NOT_A_RUNTIME_DEFAULT": (RecurrentLayerConfig,)
                    }
                },
            ),
            (
                "cannot be empty",
                {"RECURRENT_MAX_STEPS": {"RECURRENT_COMPOSITION_OPTION": ()}},
            ),
            (
                "depend on itself",
                {"RECURRENT_MAX_STEPS": {"RECURRENT_MAX_STEPS": (2,)}},
            ),
            (
                "dependency cycle",
                {
                    "RECURRENT_MAX_STEPS": {
                        "RECURRENT_NO_GRADIENT_TRANSITION_COUNT": (None,)
                    },
                    "RECURRENT_NO_GRADIENT_TRANSITION_COUNT": {
                        "RECURRENT_MAX_STEPS": (2,)
                    },
                },
            ),
        )

        for message, metadata in cases:
            with (
                self.subTest(message=message),
                patch.object(
                    transformer_linear_config,
                    "CONFIG_FIELD_APPLICABILITY",
                    metadata,
                    create=True,
                ),
                self.assertRaisesRegex(InspectionError, message),
            ):
                configuration_schema(_fresh_package("transformer/linear"))

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
