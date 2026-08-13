from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import model_runtime.inspection.overrides as override_module
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
    config_field_description,
    configuration_schema,
    parse_overrides,
    reject_locked_overrides,
    search_space_schema,
    serialize_overrides,
    supported_config_keys,
    validate_configuration,
)
from model_runtime.inspection.runtime_defaults import runtime_defaults_spec
from model_runtime.inspection.schema import _configuration_field_applicability
from model_runtime.packages import (
    ModelIdentity,
    ModelPackage,
    RuntimeDefaultsError,
    configuration_field_metadata,
)
from models.catalog import model_package
from models.transformer.linear import config as transformer_linear_config
from support.metadata_fixture import config as metadata_fixture_config


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
    def test_public_schema_value_kinds_and_choices_are_stable(self) -> None:
        package = model_package("bert/linear")
        assert package is not None
        fields = {field.key: field for field in configuration_schema(package).fields}
        expected = {
            "TRAINER_DETERMINISTIC": ("bool", False, False, (True, False)),
            "INPUT_DIM": ("int", 30_522, False, ()),
            "LEARNING_RATE": ("float", 0.001, False, ()),
            "TRAINER_ACCELERATOR": ("string", "cpu", False, ()),
            "STACK_ACTIVATION": (
                "enum",
                "GELU",
                False,
                (
                    "DISABLED",
                    "RELU",
                    "GELU",
                    "SIGMOID",
                    "TANH",
                    "LEAKY_RELU",
                    "ELU",
                    "SELU",
                    "SOFTPLUS",
                    "SOFTSIGN",
                    "SILU",
                    "MISH",
                ),
            ),
            "STACK_RESIDUAL_CONNECTION_OPTION": (
                "class",
                None,
                True,
                (
                    "AdditiveResidualConfig",
                    "AttentionResidualConfig",
                    "WeightedBlendResidualConfig",
                    "WeightedResidualConfig",
                ),
            ),
            "RESIDUAL_STACK_ACTIVATION": (
                "enum",
                None,
                True,
                (
                    "DISABLED",
                    "RELU",
                    "GELU",
                    "SIGMOID",
                    "TANH",
                    "LEAKY_RELU",
                    "ELU",
                    "SELU",
                    "SOFTPLUS",
                    "SOFTSIGN",
                    "SILU",
                    "MISH",
                ),
            ),
        }

        for key, value in expected.items():
            with self.subTest(key=key):
                field = fields[key]
                self.assertEqual(
                    (field.value_type, field.default, field.nullable, field.choices),
                    value,
                )

        ambiguous_package = model_package("transformer/linear_adaptive")
        assert ambiguous_package is not None
        ambiguous_fields = {
            field.key: field for field in configuration_schema(ambiguous_package).fields
        }
        ambiguous = ambiguous_fields["STACK_RESIDUAL_CONNECTION_OPTION"]
        self.assertEqual(ambiguous.value_type, "unknown")
        self.assertIsNone(ambiguous.default)
        self.assertTrue(ambiguous.nullable)
        self.assertTupleEqual(ambiguous.choices, ())

    def test_configuration_metadata_import_and_alias_precedence_is_stable(
        self,
    ) -> None:
        metadata = configuration_field_metadata(metadata_fixture_config)
        search_metadata = configuration_field_metadata(
            metadata_fixture_config,
            include_search_space=True,
        )

        self.assertEqual(
            list(metadata),
            [
                "SHARED",
                "BASE_ONLY",
                "IMPORTED_COLLISION",
                "STAR_COLLISION",
                "NESTED",
                "SECONDARY_ONLY",
                "RECURSIVE_B",
                "RECURSIVE_A",
                "EXPLICIT_ONLY",
                "ALIASED_ONLY",
                "LOCAL_ONLY",
                "ALIAS_CHAIN",
                "ALIAS_TARGET",
            ],
        )
        self.assertEqual(metadata["SHARED"]["sectionPath"], ["Local Options"])
        self.assertEqual(metadata["BASE_ONLY"]["sectionPath"], ["Base Options"])
        self.assertEqual(
            metadata["NESTED"]["sectionPath"],
            ["Base Options", "Nested Options"],
        )
        self.assertEqual(
            metadata["EXPLICIT_ONLY"]["sectionPath"],
            ["Explicit Options"],
        )
        self.assertEqual(
            metadata["IMPORTED_COLLISION"]["sectionPath"],
            ["Explicit Options"],
        )
        self.assertEqual(
            metadata["STAR_COLLISION"]["sectionPath"],
            ["Secondary Options"],
        )
        self.assertEqual(
            metadata["RECURSIVE_B"]["sectionPath"],
            ["Recursive B Options"],
        )
        self.assertEqual(
            metadata["RECURSIVE_A"]["sectionPath"],
            ["Recursive A Options"],
        )
        self.assertEqual(
            metadata["ALIASED_ONLY"]["sectionPath"],
            ["Aliased Options"],
        )
        self.assertNotIn("SEARCH_SPACE_IMPORTED", metadata)
        self.assertNotIn("SEARCH_SPACE_LOCAL", metadata)
        self.assertNotIn("EXPLICIT_NOT_IMPORTED", metadata)
        self.assertNotIn("UNSECTIONED", metadata)
        self.assertEqual(
            metadata["ALIAS_CHAIN"]["sortKey"][:-1],
            metadata["SHARED"]["sortKey"],
        )
        self.assertEqual(
            metadata["ALIAS_TARGET"]["sortKey"][:-1],
            metadata["ALIAS_CHAIN"]["sortKey"],
        )
        self.assertNotIn("CYCLE_A", metadata)
        self.assertNotIn("CYCLE_B", metadata)
        self.assertNotIn("INVALID_ALIAS", metadata)

        self.assertNotIn("BASE_ONLY", search_metadata)
        self.assertNotIn("STAR_COLLISION", search_metadata)
        self.assertNotIn("RECURSIVE_A", search_metadata)
        self.assertNotIn("RECURSIVE_B", search_metadata)
        self.assertIn("SEARCH_SPACE_IMPORTED", search_metadata)
        self.assertIn("SEARCH_SPACE_LOCAL", search_metadata)
        unsectioned = search_metadata["UNSECTIONED"]
        self.assertEqual(set(unsectioned), {"line", "sortKey"})
        self.assertEqual(unsectioned["sortKey"], [unsectioned["line"]])
        self.assertIn("EXPLICIT_ONLY", search_metadata)
        self.assertNotIn("EXPLICIT_NOT_IMPORTED", search_metadata)
        self.assertEqual(
            search_metadata["IMPORTED_COLLISION"]["sectionPath"],
            ["Explicit Options"],
        )
        self.assertIn("ALIASED_ONLY", search_metadata)

    def test_field_description_rule_precedence_is_stable(self) -> None:
        cases = (
            (
                "GATE_STACK_INDEPENDENT_FLAG",
                "Gate Stack Options",
                "bool",
                True,
                None,
                "Controls whether the dedicated gate stack uses its own stack "
                "settings instead of inheriting shared submodule settings.",
            ),
            (
                "STACK_RESIDUAL_MODEL_FLAG",
                "Layer Stack Options",
                "bool",
                False,
                False,
                "Uses the Residual Stack Options as a data-dependent coefficient "
                "model for the main layer stack. This is supported only when the "
                "paired residual selector uses a weighted or weighted-blend "
                "residual.",
            ),
            (
                "STACK_BIAS_FLAG",
                "Layer Stack Options",
                "bool",
                False,
                True,
                "Controls whether linear layers in the main layer stack include "
                "bias terms.",
            ),
            (
                "STACK_GATE_FLAG",
                "Layer Stack Options",
                "bool",
                False,
                False,
                "Enables or disables gate for the main layer stack.",
            ),
            (
                "GATE_HIDDEN_DIM",
                "Gate Options",
                "int",
                False,
                32,
                "Sets the hidden feature width used by the layer gating controller. "
                "Larger values increase capacity and memory use. Only applies when "
                "the gate feature is enabled.",
            ),
            (
                "FF_GATE_HIDDEN_DIM",
                "Feed-Forward Gate Stack Options",
                "int",
                False,
                32,
                "Sets the hidden feature width used by the dedicated feed-forward "
                "gate stack. Larger values increase capacity and memory use. Only "
                "matters when this stack is configured independently.",
            ),
            (
                "TRAINER_CUSTOM_LIMIT",
                "Trainer",
                "int",
                True,
                None,
                "Passes the custom limit setting through to the PyTorch Lightning "
                "trainer. Adjust it to control runtime training behavior rather "
                "than model architecture.",
            ),
            (
                "CUSTOM_WIDTH",
                "Custom Options",
                "float",
                True,
                None,
                "Sets the numeric width value for the custom options. Tune it when "
                "changing capacity, regularization, or runtime limits. Use None to "
                "inherit the builder default when inheritance is supported.",
            ),
        )

        for key, section, kind, nullable, default, expected in cases:
            with self.subTest(key=key):
                self.assertEqual(
                    config_field_description(
                        key,
                        section=section,
                        kind=kind,
                        nullable=nullable,
                        default=default,
                    ),
                    expected,
                )

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
        with patch(
            "model_runtime.packages.metadata.configuration_field_metadata",
            autospec=True,
            side_effect=configuration_field_metadata,
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
        self.assertEqual(canonicalize_overrides(package, {}), {})
        self.assertEqual(canonicalize_overrides(package, None), {})
        calls = (
            lambda: supported_config_keys(package),
            lambda: parse_overrides(package, {"HIDDEN_DIM": "1"}),
            lambda: canonicalize_overrides(package, {"HIDDEN_DIM": "1"}),
            lambda: serialize_overrides(package, {"HIDDEN_DIM": "1"}),
            lambda: reject_locked_overrides(package, "baseline", {}),
            lambda: override_module.reject_conflicting_locked_overrides(
                package,
                "baseline",
                {},
            ),
        )

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    InspectionError,
                    "Failed to import model package 'broken/missing'",
                ) as raised:
                    call()
                self.assertIsInstance(raised.exception.__cause__, ModuleNotFoundError)

    def test_override_module_preserves_all_function_owners_and_exports(self) -> None:
        expected_exports = [
            "canonicalize_overrides",
            "parse_overrides",
            "reject_conflicting_locked_overrides",
            "reject_locked_overrides",
            "resolve_override_key",
            "serialize_overrides",
            "supported_config_keys",
        ]

        self.assertEqual(override_module.__all__, expected_exports)
        for name in expected_exports:
            with self.subTest(name=name):
                operation = getattr(override_module, name)
                self.assertEqual(operation.__module__, override_module.__name__)

    def test_semantic_override_failure_keeps_runtime_defaults_error_as_cause(
        self,
    ) -> None:
        package = model_package("linears/linear")
        assert package is not None

        with self.assertRaises(InspectionError) as raised:
            parse_overrides(package, {"NO_SUCH_FIELD": "1"})

        self.assertIsInstance(
            raised.exception.__cause__,
            RuntimeDefaultsError,
        )

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

    def test_applicability_projection_and_validation_order_are_stable(self) -> None:
        metadata = {
            "RECURRENT_MAX_STEPS": {
                "RECURRENT_COMPOSITION_OPTION": (
                    RecurrentLayerConfig,
                    TinyRecursiveModelRecurrentConfig,
                ),
                "RECURRENT_REINJECT_ORIGINAL_HIDDEN_FLAG": [True, False],
            },
            "RECURRENT_NO_GRADIENT_TRANSITION_COUNT": {
                "RECURRENT_MAX_STEPS": (2, 3),
            },
        }
        with patch.object(
            transformer_linear_config,
            "CONFIG_FIELD_APPLICABILITY",
            metadata,
            create=True,
        ):
            fields = {
                field.key: field
                for field in configuration_schema(
                    _fresh_package("transformer/linear")
                ).fields
            }

        self.assertEqual(
            tuple(
                (condition.key, condition.values)
                for condition in fields["RECURRENT_MAX_STEPS"].applicable_when
            ),
            (
                (
                    "RECURRENT_COMPOSITION_OPTION",
                    (
                        "RecurrentLayerConfig",
                        "TinyRecursiveModelRecurrentConfig",
                    ),
                ),
                ("RECURRENT_REINJECT_ORIGINAL_HIDDEN_FLAG", (True, False)),
            ),
        )
        self.assertEqual(
            tuple(
                (condition.key, condition.values)
                for condition in fields[
                    "RECURRENT_NO_GRADIENT_TRANSITION_COUNT"
                ].applicable_when
            ),
            (("RECURRENT_MAX_STEPS", (2, 3)),),
        )

        cycle_before_unknown_target = {
            "RECURRENT_MAX_STEPS": {
                "RECURRENT_NO_GRADIENT_TRANSITION_COUNT": (None,),
            },
            "RECURRENT_NO_GRADIENT_TRANSITION_COUNT": {
                "RECURRENT_MAX_STEPS": (2,),
            },
            "NOT_A_RUNTIME_DEFAULT": {},
        }
        with (
            patch.object(
                transformer_linear_config,
                "CONFIG_FIELD_APPLICABILITY",
                cycle_before_unknown_target,
                create=True,
            ),
            self.assertRaises(InspectionError) as raised,
        ):
            configuration_schema(_fresh_package("transformer/linear"))
        self.assertEqual(
            str(raised.exception),
            "Model 'transformer/linear' CONFIG_FIELD_APPLICABILITY contains "
            "unknown target key 'NOT_A_RUNTIME_DEFAULT'.",
        )

    def test_applicability_mapping_is_validated_before_supported_keys(self) -> None:
        class HashFailure(str):
            def __hash__(self) -> int:
                raise AssertionError("supported key was hashed")

        spec = SimpleNamespace(
            configuration_applicability=lambda: [],
            package=SimpleNamespace(catalog_key="fixture/model"),
        )

        with self.assertRaisesRegex(
            InspectionError,
            "CONFIG_FIELD_APPLICABILITY must be a mapping",
        ):
            _configuration_field_applicability(
                spec,  # type: ignore[arg-type]
                [HashFailure("FIELD")],
            )

    def test_applicability_errors_precede_missing_heading_metadata(self) -> None:
        package = _fresh_package("transformer/linear")
        with (
            patch.object(
                transformer_linear_config,
                "CONFIG_FIELD_APPLICABILITY",
                {"NOT_A_RUNTIME_DEFAULT": {}},
                create=True,
            ),
            patch(
                "model_runtime.packages.metadata.configuration_field_metadata",
                return_value={},
            ),
            self.assertRaisesRegex(
                InspectionError,
                "unknown target key 'NOT_A_RUNTIME_DEFAULT'",
            ),
        ):
            configuration_schema(package)

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
