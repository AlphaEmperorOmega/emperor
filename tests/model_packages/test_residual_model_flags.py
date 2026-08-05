from __future__ import annotations

import unittest
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from importlib import import_module

import torch

from emperor.augmentations.adaptive_parameters import SingleModelDynamicWeightConfig
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    AttentionResidualConfig,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.linears import LinearLayer, LinearLayerConfig
from model_runtime.inspection import configuration_schema
from model_runtime.inspection.model_graph import inspect_model_graph
from models.catalog import discover_model_packages, model_package
from models.linears.linear.config_builder import LinearConfigBuilder

_SELECTOR_SUFFIX = "_RESIDUAL_CONNECTION_OPTION"
_MODEL_FLAG_SUFFIX = "_RESIDUAL_MODEL_FLAG"
_SUPPORTED_MODEL_SELECTORS = (
    WeightedResidualConfig,
    WeightedBlendResidualConfig,
)
_STACK_OPTION_SUFFIXES = (
    "INDEPENDENT_FLAG",
    "HIDDEN_DIM",
    "LAYER_NORM_POSITION",
    "NUM_LAYERS",
    "ACTIVATION",
    "RESIDUAL_CONNECTION_OPTION",
    "RESIDUAL_MODEL_FLAG",
    "DROPOUT_PROBABILITY",
    "LAST_LAYER_BIAS_OPTION",
    "APPLY_OUTPUT_PIPELINE_FLAG",
    "BIAS_FLAG",
)


def _model_flag_for(selector_key: str) -> str:
    return f"{selector_key.removesuffix(_SELECTOR_SUFFIX)}{_MODEL_FLAG_SUFFIX}"


def _walk_objects(value: object, *, path: str = "root"):
    pending = [(path, value)]
    seen: set[int] = set()
    while pending:
        current_path, current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current_path, current
        if is_dataclass(current):
            pending.extend(
                (
                    f"{current_path}.{field.name}",
                    getattr(current, field.name),
                )
                for field in fields(current)
            )
        elif isinstance(current, Mapping):
            pending.extend(
                (f"{current_path}[{key!r}]", nested) for key, nested in current.items()
            )
        elif isinstance(current, (list, tuple, set, frozenset)):
            pending.extend(
                (f"{current_path}[{index}]", nested)
                for index, nested in enumerate(current)
            )
        elif hasattr(current, "_values"):
            pending.append((f"{current_path}._values", current._values))


def _enabled_residual_model_flag_paths(runtime: object) -> list[str]:
    paths: list[str] = []
    for path, value in _walk_objects(runtime, path="runtime"):
        if is_dataclass(value):
            paths.extend(
                f"{path}.{field.name}"
                for field in fields(value)
                if field.name.endswith("residual_model_flag")
                and getattr(value, field.name) is True
            )
        elif isinstance(value, Mapping):
            paths.extend(
                f"{path}[{key!r}]"
                for key, nested in value.items()
                if isinstance(key, str)
                and key.endswith("residual_model_flag")
                and nested is True
            )
    return paths


def _modeled_residuals(configuration: object):
    return [
        (path, value)
        for path, value in _walk_objects(configuration, path="configuration")
        if isinstance(value, _SUPPORTED_MODEL_SELECTORS)
        and value.model_config is not None
    ]


class TestResidualModelFlagCatalogContract(unittest.TestCase):
    def test_every_package_exposes_the_exact_gate_parity_residual_stack(self) -> None:
        for package in discover_model_packages():
            config = package.runtime_defaults
            metadata = package.configuration_field_metadata()
            schema_fields = {
                field.key: field for field in configuration_schema(package).fields
            }
            residual_stack_keys = [
                key for key in vars(config) if key.startswith("RESIDUAL_STACK_")
            ]
            residual_stack_suffixes = [
                key.removeprefix("RESIDUAL_STACK_") for key in residual_stack_keys
            ]
            gate_stack_suffixes = [
                key.removeprefix("GATE_STACK_")
                for key in vars(config)
                if key.startswith("GATE_STACK_")
            ]

            with self.subTest(package=package.catalog_key):
                self.assertTupleEqual(
                    tuple(residual_stack_suffixes),
                    _STACK_OPTION_SUFFIXES,
                )
                if gate_stack_suffixes:
                    self.assertListEqual(residual_stack_suffixes, gate_stack_suffixes)
                self.assertFalse(
                    any("RESIDUAL_MODEL_STACK" in key for key in vars(config))
                )
                self.assertIs(config.RESIDUAL_STACK_INDEPENDENT_FLAG, False)
                self.assertIs(config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG, False)

                for key in residual_stack_keys:
                    self.assertEqual(
                        metadata[key]["sectionPath"][-2:],
                        ["Residual Options", "Residual Stack Options"],
                    )
                    self.assertIn(key, schema_fields)

    def test_every_residual_selector_has_a_disabled_schema_paired_model_flag(
        self,
    ) -> None:
        for package in discover_model_packages():
            config = package.runtime_defaults
            metadata = package.configuration_field_metadata()
            schema_fields = {
                field.key: field for field in configuration_schema(package).fields
            }
            selectors = sorted(
                key for key in vars(config) if key.endswith(_SELECTOR_SUFFIX)
            )

            self.assertTrue(selectors, package.catalog_key)
            for selector_key in selectors:
                model_flag_key = _model_flag_for(selector_key)
                with self.subTest(
                    package=package.catalog_key,
                    selector=selector_key,
                ):
                    self.assertTrue(hasattr(config, model_flag_key), model_flag_key)
                    self.assertIs(config.__annotations__.get(model_flag_key), bool)
                    self.assertIs(getattr(config, model_flag_key), False)
                    self.assertEqual(
                        metadata[model_flag_key]["sectionPath"],
                        metadata[selector_key]["sectionPath"],
                    )

                    if selector_key not in schema_fields:
                        self.assertNotIn(model_flag_key, schema_fields)
                        continue

                    field = schema_fields[model_flag_key]
                    self.assertEqual(field.value_type, "bool")
                    self.assertIs(field.default, False)
                    self.assertIn(
                        "Residual Stack Options as a data-dependent coefficient model",
                        field.description,
                    )
                    self.assertTupleEqual(field.applicable_when, ())

    def test_every_package_local_factory_builds_only_supported_residual_stacks(
        self,
    ) -> None:
        for package in discover_model_packages():
            package_module = package.runtime_defaults.__package__
            residuals = import_module(f"{package_module}._residual")
            build = residuals.build_residual_config
            residual_stack = residuals.ResidualStackOptions(
                hidden_dim=8,
                num_layers=2,
                activation=ActivationOptions.GELU,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_connection_option=None,
                residual_model_flag=False,
                dropout_probability=0.0,
                last_layer_bias_option=LastLayerBiasOptions.ENABLED,
                apply_output_pipeline_flag=False,
                bias_flag=True,
            )

            with self.subTest(package=package.catalog_key, selector="none"):
                self.assertIsNone(build(None, False))

            for selector in _SUPPORTED_MODEL_SELECTORS:
                with self.subTest(
                    package=package.catalog_key,
                    selector=selector.__name__,
                ):
                    scalar = build(selector, False)
                    self.assertIsNone(scalar.model_config)

                    modeled = build(selector, True, residual_stack)
                    self.assertIsInstance(modeled.model_config, LayerStackConfig)
                    self.assertEqual(modeled.model_config.hidden_dim, 8)
                    self.assertEqual(modeled.model_config.num_layers, 2)
                    self.assertIsInstance(
                        modeled.model_config.layer_config.layer_model_config,
                        LinearLayerConfig,
                    )
                    self.assertIs(
                        modeled.model_config.layer_config.layer_model_config.bias_flag,
                        True,
                    )

            for selector in (None, AdditiveResidualConfig, AttentionResidualConfig):
                with self.subTest(
                    package=package.catalog_key,
                    invalid=getattr(selector, "__name__", None),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "CUSTOM_RESIDUAL_MODEL_FLAG.*CUSTOM_RESIDUAL_CONNECTION_OPTION",
                    ):
                        build(
                            selector,
                            True,
                            selector_field="CUSTOM_RESIDUAL_CONNECTION_OPTION",
                            model_flag_field="CUSTOM_RESIDUAL_MODEL_FLAG",
                        )

            with self.subTest(package=package.catalog_key, invalid="recursive"):
                with self.assertRaisesRegex(
                    ValueError,
                    "RESIDUAL_STACK_RESIDUAL_MODEL_FLAG.*"
                    "RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION",
                ):
                    build(
                        WeightedResidualConfig,
                        True,
                        replace(
                            residual_stack,
                            residual_connection_option=WeightedResidualConfig,
                            residual_model_flag=True,
                        ),
                    )

    def test_every_flat_residual_model_flag_reaches_runtime_options(self) -> None:
        for package in discover_model_packages():
            config = package.runtime_defaults
            for model_flag_key in sorted(
                key for key in vars(config) if key.endswith(_MODEL_FLAG_SUFFIX)
            ):
                overrides = {model_flag_key.lower(): True}
                independent_key = (
                    f"{model_flag_key.removesuffix(_MODEL_FLAG_SUFFIX)}"
                    "_INDEPENDENT_FLAG"
                )
                if hasattr(config, independent_key):
                    overrides[independent_key.lower()] = True

                with self.subTest(
                    package=package.catalog_key,
                    model_flag=model_flag_key,
                ):
                    runtime = package.bind_runtime_defaults(overrides)
                    self.assertTrue(
                        _enabled_residual_model_flag_paths(runtime),
                        model_flag_key,
                    )

    def test_representative_package_sites_build_residual_stacks(self) -> None:
        cases = (
            (
                "linears/linear",
                {
                    "stack_residual_connection_option": WeightedResidualConfig,
                    "stack_residual_model_flag": True,
                },
                WeightedResidualConfig,
            ),
            (
                "transformer/linear",
                {
                    "recurrent_flag": True,
                    "recurrent_residual_connection_option": WeightedResidualConfig,
                    "recurrent_residual_model_flag": True,
                },
                WeightedResidualConfig,
            ),
            (
                "transformer/linear",
                {
                    "attn_stack_residual_connection_option": (
                        WeightedBlendResidualConfig
                    ),
                    "attn_stack_residual_model_flag": True,
                },
                WeightedBlendResidualConfig,
            ),
            (
                "experts/linear",
                {
                    "expert_stack_residual_connection_option": WeightedResidualConfig,
                    "expert_stack_residual_model_flag": True,
                },
                WeightedResidualConfig,
            ),
            (
                "experts/linear",
                {
                    "router_stack_residual_connection_option": (
                        WeightedBlendResidualConfig
                    ),
                    "router_stack_residual_model_flag": True,
                },
                WeightedBlendResidualConfig,
            ),
            (
                "linears/linear",
                {
                    "stack_gate_flag": True,
                    "gate_stack_independent_flag": True,
                    "gate_stack_residual_connection_option": WeightedResidualConfig,
                    "gate_stack_residual_model_flag": True,
                },
                WeightedResidualConfig,
            ),
            (
                "linears/linear_adaptive",
                {
                    "weight_option_flag": True,
                    "weight_option": SingleModelDynamicWeightConfig,
                    "adaptive_generator_stack_residual_connection_option": (
                        WeightedBlendResidualConfig
                    ),
                    "adaptive_generator_stack_residual_model_flag": True,
                },
                WeightedBlendResidualConfig,
            ),
            (
                "mlp_mixer/linear",
                {
                    "mixer_residual_connection_option": WeightedResidualConfig,
                    "mixer_residual_model_flag": True,
                },
                WeightedResidualConfig,
            ),
            (
                "parametric/parametric_vector",
                {
                    "stack_residual_connection_option": (WeightedBlendResidualConfig),
                    "stack_residual_model_flag": True,
                },
                WeightedBlendResidualConfig,
            ),
        )

        for package_key, overrides, expected_type in cases:
            with self.subTest(package=package_key, overrides=tuple(overrides)):
                package = model_package(package_key)
                assert package is not None
                configuration = package.build_configuration(
                    config_overrides=overrides,
                )
                residuals = _modeled_residuals(configuration)
                self.assertTrue(residuals)
                for path, residual in residuals:
                    self.assertIs(type(residual), expected_type, path)
                    self.assertIsInstance(residual.model_config, LayerStackConfig)
                    self.assertIsInstance(
                        residual.model_config.layer_config.layer_model_config,
                        LinearLayerConfig,
                    )
                    self.assertIs(
                        residual.model_config.layer_config.layer_model_config.bias_flag,
                        True,
                    )

    def test_every_package_builds_all_independent_residual_stack_options(self) -> None:
        for package in discover_model_packages():
            config = package.runtime_defaults
            selector_prefix = (
                "STACK"
                if hasattr(config, "STACK_RESIDUAL_CONNECTION_OPTION")
                else "ATTN_STACK"
            )
            configuration = package.build_configuration(
                config_overrides={
                    f"{selector_prefix.lower()}_residual_connection_option": (
                        WeightedResidualConfig
                    ),
                    f"{selector_prefix.lower()}_residual_model_flag": True,
                    "residual_stack_independent_flag": True,
                    "residual_stack_hidden_dim": 7,
                    "residual_stack_num_layers": 1,
                    "residual_stack_activation": ActivationOptions.RELU,
                    "residual_stack_layer_norm_position": (
                        LayerNormPositionOptions.AFTER
                    ),
                    "residual_stack_residual_connection_option": (
                        AdditiveResidualConfig
                    ),
                    "residual_stack_residual_model_flag": False,
                    "residual_stack_dropout_probability": 0.2,
                    "residual_stack_last_layer_bias_option": (
                        LastLayerBiasOptions.ENABLED
                    ),
                    "residual_stack_apply_output_pipeline_flag": True,
                    "residual_stack_bias_flag": True,
                }
            )
            residuals = _modeled_residuals(configuration)

            with self.subTest(package=package.catalog_key):
                self.assertTrue(residuals)
                for path, residual in residuals:
                    stack = residual.model_config
                    self.assertIsInstance(stack, LayerStackConfig, path)
                    self.assertEqual(stack.hidden_dim, 7)
                    self.assertEqual(stack.num_layers, 1)
                    self.assertIs(stack.layer_config.activation, ActivationOptions.RELU)
                    self.assertIs(
                        stack.layer_config.layer_norm_position,
                        LayerNormPositionOptions.AFTER,
                    )
                    self.assertIsInstance(
                        stack.layer_config.residual_config,
                        AdditiveResidualConfig,
                    )
                    self.assertEqual(stack.layer_config.dropout_probability, 0.2)
                    self.assertIs(
                        stack.last_layer_bias_option,
                        LastLayerBiasOptions.ENABLED,
                    )
                    self.assertIs(stack.apply_output_pipeline_flag, True)
                    self.assertIs(
                        stack.layer_config.layer_model_config.bias_flag,
                        True,
                    )

    def test_independent_residual_stack_options_reach_the_model_config(self) -> None:
        package = model_package("linears/linear")
        assert package is not None
        configuration = package.build_configuration(
            config_overrides={
                "stack_residual_connection_option": WeightedResidualConfig,
                "stack_residual_model_flag": True,
                "residual_stack_independent_flag": True,
                "residual_stack_hidden_dim": 19,
                "residual_stack_num_layers": 3,
                "residual_stack_activation": ActivationOptions.RELU,
                "residual_stack_layer_norm_position": (LayerNormPositionOptions.AFTER),
                "residual_stack_dropout_probability": 0.2,
                "residual_stack_last_layer_bias_option": (LastLayerBiasOptions.ENABLED),
                "residual_stack_apply_output_pipeline_flag": True,
                "residual_stack_bias_flag": True,
            }
        )
        residuals = _modeled_residuals(configuration)
        self.assertTrue(residuals)
        for path, residual in residuals:
            stack = residual.model_config
            self.assertIsInstance(stack, LayerStackConfig, path)
            self.assertEqual(stack.hidden_dim, 19)
            self.assertEqual(stack.num_layers, 3)
            self.assertIs(stack.layer_config.activation, ActivationOptions.RELU)
            self.assertIs(
                stack.layer_config.layer_norm_position,
                LayerNormPositionOptions.AFTER,
            )
            self.assertEqual(stack.layer_config.dropout_probability, 0.2)
            self.assertIs(
                stack.last_layer_bias_option,
                LastLayerBiasOptions.ENABLED,
            )
            self.assertIs(stack.apply_output_pipeline_flag, True)
            self.assertIs(stack.layer_config.layer_model_config.bias_flag, True)

    def test_invalid_flat_and_direct_runtime_pairs_name_both_fields(self) -> None:
        package = model_package("linears/linear")
        assert package is not None
        for selector in (None, AdditiveResidualConfig, AttentionResidualConfig):
            with self.subTest(source="flat", selector=selector):
                with self.assertRaisesRegex(
                    ValueError,
                    "STACK_RESIDUAL_MODEL_FLAG.*STACK_RESIDUAL_CONNECTION_OPTION",
                ):
                    package.build_configuration(
                        config_overrides={
                            "stack_residual_connection_option": selector,
                            "stack_residual_model_flag": True,
                        }
                    )

        runtime = package.bind_runtime_defaults()
        direct_runtime = replace(
            runtime,
            stack=replace(
                runtime.stack,
                residual_connection_option=AdditiveResidualConfig,
                residual_model_flag=True,
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "STACK_RESIDUAL_MODEL_FLAG.*STACK_RESIDUAL_CONNECTION_OPTION",
        ):
            LinearConfigBuilder(runtime=direct_runtime).build()

    def test_existing_presets_keep_weighted_residuals_scalar(self) -> None:
        for package in discover_model_packages():
            for preset in package.preset_type:
                with self.subTest(package=package.catalog_key, preset=preset.name):
                    configuration = package.build_configuration(preset)
                    for path, value in _walk_objects(configuration):
                        if isinstance(value, _SUPPORTED_MODEL_SELECTORS):
                            self.assertIsNone(value.model_config, path)

    def test_package_built_model_exposes_coefficient_model_in_graph(self) -> None:
        package = model_package("linears/linear")
        assert package is not None
        configuration = package.build_configuration(
            config_overrides={
                "stack_residual_connection_option": WeightedBlendResidualConfig,
                "stack_residual_model_flag": True,
            }
        )
        model = package.build_model(configuration)
        residuals = [
            module
            for module in model.modules()
            if isinstance(getattr(module, "cfg", None), WeightedBlendResidualConfig)
        ]
        self.assertTrue(residuals)
        for residual in residuals:
            self.assertIsNone(residual.raw_weight)
            self.assertIsInstance(residual.model, LayerStack)
            self.assertEqual(residual.model.input_dim, 64)
            self.assertEqual(residual.model.output_dim, 32)
            coefficient_output = residual.model[-1].model
            self.assertIsInstance(coefficient_output, LinearLayer)
            self.assertIsNotNone(coefficient_output.bias_params)
            torch.testing.assert_close(
                coefficient_output.weight_params,
                torch.zeros_like(coefficient_output.weight_params),
            )

        output = model(torch.randn(2, 1, 28, 28))
        if isinstance(output, tuple):
            output = output[0]
        output.sum().backward()
        self.assertTrue(
            any(
                residual.model[-1].model.weight_params.grad is not None
                and torch.count_nonzero(
                    residual.model[-1].model.weight_params.grad
                ).item()
                > 0
                for residual in residuals
            )
        )

        graph = inspect_model_graph(model)
        layer_nodes = [
            node
            for node in graph.nodes
            if node.configuration is not None
            and {field.key: field.value for field in node.configuration.fields}.get(
                "residual_model_config"
            )
            == "LayerStackConfig"
        ]
        self.assertTrue(layer_nodes)
        node_ids = {node.id for node in graph.nodes}
        for node in layer_nodes:
            residual_node = f"{node.id}.residual_connection"
            coefficient_node = f"{residual_node}.model"
            self.assertIn(residual_node, node_ids)
            self.assertIn(coefficient_node, node_ids)


if __name__ == "__main__":
    unittest.main()
