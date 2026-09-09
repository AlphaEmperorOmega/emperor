import inspect
import math
import unittest
from dataclasses import fields
from itertools import product

import torch
import torch.nn as nn

from emperor.convs import Conv2dLayerConfig
from emperor.halting import (
    HaltingHiddenStateModeOptions,
    SoftHaltingConfig,
    SoftHaltingState,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    AttentionResidualConfig,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    NormalizationOptions,
    ResidualConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.layers._composition.gate import LayerGate
from emperor.linears import LinearLayerConfig
from emperor.nn import Module

DEFAULT_LAYER_MODEL_CONFIG = object()


class AddConstantModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return input + self.value


class ConstantGate(nn.Module):
    def __init__(self, value: float | torch.Tensor):
        super().__init__()
        self.value = value
        self.received_state = None
        self.received_hidden = None

    def forward(self, state: LayerState) -> LayerState:
        self.received_state = state
        self.received_hidden = state.hidden
        if torch.is_tensor(self.value):
            hidden = self.value.to(state.hidden).expand_as(state.hidden)
        else:
            hidden = torch.full_like(state.hidden, self.value)
        return LayerState(hidden=hidden)


class FakeHaltingState:
    def __init__(self, halt_mask: torch.Tensor):
        self.halt_mask = halt_mask
        self.output_hidden = None


class FakeHaltingModel(nn.Module):
    def __init__(
        self,
        halting_state: FakeHaltingState,
        halting_output_offset: float = 1.0,
        finalized_hidden_offset: float = 10.0,
        loss: torch.Tensor | None = None,
    ):
        super().__init__()
        self.halting_state = halting_state
        self.halting_output_offset = halting_output_offset
        self.finalized_hidden_offset = finalized_hidden_offset
        self.loss = loss if loss is not None else torch.tensor([1.0, 3.0])
        self.update_calls = 0
        self.finalize_calls = 0
        self.finalize_input = None

    def update_halting_state(
        self,
        previous_state,
        model_hidden_state: torch.Tensor,
    ) -> tuple[FakeHaltingState, torch.Tensor]:
        self.update_calls += 1
        output_hidden = model_hidden_state + self.halting_output_offset
        self.halting_state.output_hidden = output_hidden
        return self.halting_state, output_hidden

    def finalize_weighted_accumulation(
        self,
        state: FakeHaltingState,
        current_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.finalize_calls += 1
        self.finalize_input = current_hidden
        return current_hidden + self.finalized_hidden_offset, self.loss.to(
            current_hidden
        )


class TestLayer(unittest.TestCase):
    def test_layer_package_public_exports_are_exact(self):
        import emperor.layers as layer_package

        expected_exports = {
            "ActivationOptions",
            "AdditiveResidualConfig",
            "AttentionResidualConfig",
            "GateConfig",
            "LastLayerBiasOptions",
            "LayerConfig",
            "LayerGateOptions",
            "LayerNormPositionOptions",
            "NormalizationOptions",
            "LayerStackConfig",
            "MirroredLayerStackConfig",
            "HierarchicalReasoningModelRecurrentConfig",
            "InnerThinkingRecurrentConfig",
            "RecurrentCompositionConfig",
            "RecurrentLayerConfig",
            "TinyRecursiveModelRecurrentConfig",
            "ResidualConfig",
            "WeightedBlendResidualConfig",
            "WeightedResidualConfig",
            "LayerState",
            "Layer",
            "LayerStack",
            "MirroredLayerStack",
            "RecurrentLayer",
            "LayerControllerMonitorCallback",
            "RecurrentLayerMonitorCallback",
        }

        self.assertEqual(set(layer_package.__all__), expected_exports)
        for name in expected_exports:
            with self.subTest(name=name):
                self.assertTrue(hasattr(layer_package, name))
        for retired_export in (
            "LayerGate",
            "RecurrentCompositionAbstract",
            "RecurrentLayerValidator",
            "ResidualConnection",
            "ResidualConnectionOptions",
            "HRMRecurrent",
            "HRMRecurrentConfig",
            "HierarchicalReasoningModelRecurrent",
            "TRMRecurrent",
            "TRMRecurrentConfig",
            "TinyRecursiveModelRecurrent",
        ):
            with self.subTest(retired_export=retired_export):
                self.assertFalse(hasattr(layer_package, retired_export))

    def test_gate_config_is_public_and_builds_the_private_implementation(self):
        from emperor.layers import (
            GateConfig as LayerPackageGateConfig,
        )
        from emperor.layers import (
            LayerGateOptions as LayerPackageGateOptions,
        )

        self.assertIs(LayerPackageGateConfig, GateConfig)
        self.assertIs(LayerPackageGateOptions, LayerGateOptions)

        gate_config = GateConfig(
            gate_dim=1,
            option=LayerGateOptions.MULTIPLIER,
            model_config=self.gate_stack_config(dim=1),
        )
        self.assertIsInstance(gate_config.build(), LayerGate)

    def test_gate_config_uses_single_gate_dimension(self):
        field_names = {field.name for field in fields(GateConfig)}
        gate_config = GateConfig(
            gate_dim=3,
            model_config=self.gate_stack_config(dim=1),
            option=LayerGateOptions.MULTIPLIER,
        )

        built_gate = gate_config.build()

        self.assertIn("gate_dim", field_names)
        self.assertNotIn("input_dim", field_names)
        self.assertNotIn("output_dim", field_names)
        self.assertEqual(built_gate.gate_dim, 3)
        self.assertFalse(hasattr(built_gate, "input_dim"))
        self.assertFalse(hasattr(built_gate, "output_dim"))
        self.assertEqual(built_gate.model.input_dim, 3)
        self.assertEqual(built_gate.model.output_dim, 3)

    def test_residual_configs_are_public_and_build_private_implementations(self):
        import emperor.layers as layer_package

        weighted_config_fields = {
            config_field.name: config_field
            for config_field in fields(WeightedBlendResidualConfig)
        }
        residual_config = WeightedBlendResidualConfig(
            residual_dim=3,
            model_config=LinearLayerConfig(bias_flag=True),
        )

        residual_connection = residual_config.build()

        self.assertIs(layer_package.ResidualConfig, ResidualConfig)
        self.assertEqual(
            tuple(config_field.name for config_field in fields(ResidualConfig)),
            ("residual_dim",),
        )
        self.assertEqual(
            tuple(weighted_config_fields), ("residual_dim", "model_config")
        )
        model_config_help = weighted_config_fields["model_config"].metadata["help"]
        self.assertIn("data-dependent", model_config_help.lower())
        self.assertEqual(type(residual_connection).__name__, "WeightedBlendResidual")
        self.assertEqual(residual_connection.residual_dim, 3)
        self.assertEqual(residual_connection.model.input_dim, 6)
        self.assertEqual(residual_connection.model.output_dim, 3)

    def test_attention_residual_config_declares_depth_mixing_parameters(self):
        config = AttentionResidualConfig(
            residual_dim=7,
            block_size=3,
            rms_norm_epsilon=1e-5,
        )

        self.assertEqual(
            tuple(config_field.name for config_field in fields(type(config))),
            ("residual_dim", "block_size", "rms_norm_epsilon", "model_config"),
        )
        self.assertEqual(config.residual_dim, 7)
        self.assertEqual(config.block_size, 3)
        self.assertEqual(config.rms_norm_epsilon, 1e-5)
        self.assertIsNone(config.model_config)
        built_residual = config.build()
        self.assertEqual(built_residual.residual_dim, 7)
        self.assertEqual(built_residual.block_size, 3)
        self.assertEqual(built_residual.rms_norm_epsilon, 1e-5)
        self.assertIsNone(built_residual.query_model)

    def test_concrete_residual_configs_share_the_abstract_interface(self):
        for config_type in (
            AdditiveResidualConfig,
            WeightedResidualConfig,
            WeightedBlendResidualConfig,
            AttentionResidualConfig,
        ):
            with self.subTest(config_type=config_type.__name__):
                self.assertTrue(issubclass(config_type, ResidualConfig))

    def bare_config(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        bias_flag: bool = True,
        activation: ActivationOptions = ActivationOptions.DISABLED,
        residual_connection_option: type[ResidualConfig] | None = None,
        dropout_probability: float = 0.0,
        layer_norm_position: LayerNormPositionOptions = (
            LayerNormPositionOptions.DISABLED
        ),
        gate_config: "LayerStackConfig | None" = None,
        gate_option: LayerGateOptions | None = None,
        gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID,
        halting_config: "StickBreakingConfig | None" = None,
        memory_config=None,
        layer_model_config=DEFAULT_LAYER_MODEL_CONFIG,
        residual_model_config: LinearLayerConfig | None = None,
    ) -> LayerConfig:
        if layer_model_config is DEFAULT_LAYER_MODEL_CONFIG:
            layer_model_config = LinearLayerConfig(bias_flag=bias_flag)
        return LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            residual_config=(
                None
                if residual_connection_option is None
                else residual_connection_option(
                    **(
                        {}
                        if residual_model_config is None
                        else {"model_config": residual_model_config}
                    )
                )
            ),
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=self.layer_gate_config(
                gate_config,
                gate_option,
                output_dim,
                gate_activation,
            ),
            halting_config=halting_config,
            memory_config=memory_config,
            layer_model_config=layer_model_config,
        )

    def layer_gate_config(
        self,
        model_config: "LayerStackConfig | GateConfig | None",
        option: LayerGateOptions | None,
        dim: int,
        activation: ActivationOptions | None = ActivationOptions.SIGMOID,
    ) -> GateConfig | None:
        if isinstance(model_config, GateConfig):
            return model_config
        if model_config is None:
            if option is None:
                return None
            model_config = self.gate_stack_config(dim)
        if option is None:
            option = LayerGateOptions.MULTIPLIER
        return GateConfig(
            model_config=model_config,
            option=option,
            activation=activation,
        )

    def gate_stack_config(
        self,
        dim: int,
        bias_flag: bool = False,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=False,
            layer_config=LayerConfig(
                input_dim=dim,
                output_dim=dim,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=bias_flag),
            ),
        )

    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 8,
        bias_flag: bool = True,
        activation: ActivationOptions = ActivationOptions.RELU,
        residual_connection_option: type[ResidualConfig] | None = None,
        dropout_probability: float = 0.2,
        layer_norm_position: LayerNormPositionOptions = (
            LayerNormPositionOptions.DISABLED
        ),
        gate_config: "LayerStackConfig | None" = None,
        gate_num_layers: int = 1,
        gate_activation: ActivationOptions = ActivationOptions.DISABLED,
        gate_residual_connection_option: type[ResidualConfig] | None = None,
        gate_dropout_probability: float = 0.0,
        gate_bias_flag: bool = True,
        gate_option: LayerGateOptions | None = None,
        halting_config: "StickBreakingConfig | None" = None,
        residual_model_config: LinearLayerConfig | None = None,
    ) -> LayerConfig:
        if gate_config is None and (input_dim == output_dim or gate_option is not None):
            gate_config = LayerStackConfig(
                hidden_dim=output_dim,
                num_layers=gate_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_postprocessing_flag=False,
                layer_config=LayerConfig(
                    activation=gate_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None
                    if gate_residual_connection_option is None
                    else gate_residual_connection_option(),
                    dropout_probability=gate_dropout_probability,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=gate_bias_flag,
                    ),
                ),
            )

        if halting_config is None and input_dim == output_dim:
            halting_config = StickBreakingConfig(
                threshold=0.99,
                ponder_cost_weight=1.0,
                min_steps=1,
                dropout_probability=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=LayerStackConfig(
                    hidden_dim=output_dim,
                    output_dim=2,
                    num_layers=gate_num_layers,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_postprocessing_flag=False,
                    layer_config=LayerConfig(
                        activation=gate_activation,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        residual_config=None
                        if gate_residual_connection_option is None
                        else gate_residual_connection_option(),
                        dropout_probability=gate_dropout_probability,
                        halting_config=None,
                        gate_config=None,
                        layer_model_config=LinearLayerConfig(
                            bias_flag=True,
                        ),
                    ),
                ),
            )

        return LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            residual_config=(
                None
                if residual_connection_option is None
                else residual_connection_option(
                    **(
                        {}
                        if residual_model_config is None
                        else {"model_config": residual_model_config}
                    )
                )
            ),
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=self.layer_gate_config(
                gate_config,
                gate_option,
                output_dim,
                ActivationOptions.SIGMOID,
            ),
            halting_config=halting_config,
            layer_model_config=LinearLayerConfig(
                bias_flag=bias_flag,
            ),
        )

    def _halting_config(
        self,
        dim: int,
        threshold: float = 1e-9,
    ) -> StickBreakingConfig:
        return StickBreakingConfig(
            input_dim=dim,
            threshold=threshold,
            ponder_cost_weight=1.0,
            min_steps=1,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=2,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_postprocessing_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=True,
                    ),
                ),
            ),
        )

    def test_gate_config_registry_owner_and_build_return_layer_gate(self):
        gate_config = GateConfig(
            model_config=self.gate_stack_config(dim=3),
            option=LayerGateOptions.MULTIPLIER,
        )

        self.assertIs(gate_config._registry_owner(), LayerGate)
        built_gate = gate_config.build()
        self.assertIsInstance(built_gate, LayerGate)
        self.assertEqual(built_gate.option, LayerGateOptions.MULTIPLIER)

    def test_gate_config_build_rejects_missing_option(self):
        gate_config = GateConfig(model_config=self.gate_stack_config(dim=3))

        with self.assertRaisesRegex(ValueError, "GateConfig.option.*LayerGateOptions"):
            gate_config.build()

    def test_gate_config_rejects_non_positive_gate_dimension(self):
        gate_config = GateConfig(
            gate_dim=0,
            model_config=self.gate_stack_config(dim=3),
            option=LayerGateOptions.MULTIPLIER,
        )

        with self.assertRaisesRegex(ValueError, "gate_dim"):
            gate_config.build()

    def test_gate_config_requires_model_config_when_provided_to_layer(self):
        gate_config = GateConfig(option=LayerGateOptions.MULTIPLIER)

        with self.assertRaisesRegex(ValueError, "model_config"):
            Layer(self.bare_config(gate_config=gate_config))

    def test_init_delegates_feature_config_and_modules(self):
        cfg = self.preset(input_dim=8, output_dim=8)
        layer = Layer(cfg)

        self.assertIsInstance(layer, Layer)
        for delegate in (
            layer.postprocessing,
            layer.halting,
            layer.memory,
            layer.residual,
            layer.normalization,
        ):
            with self.subTest(delegate=type(delegate).__name__):
                self.assertIsInstance(delegate, Module)
        self.assertEqual(layer.input_dim, cfg.input_dim)
        self.assertEqual(layer.output_dim, cfg.output_dim)
        self.assertIs(layer.cfg, cfg)
        self.assertEqual(layer.postprocessing.activation_function, cfg.activation)
        self.assertEqual(layer.residual.config, cfg.residual_config)
        self.assertEqual(
            layer.postprocessing.dropout_probability,
            cfg.dropout_probability,
        )
        self.assertEqual(layer.normalization.position, cfg.layer_norm_position)
        self.assertEqual(layer.postprocessing.gate_config, cfg.gate_config)
        self.assertIsNotNone(layer.postprocessing.gate)
        self.assertEqual(layer.postprocessing.gate.gate_dim, cfg.output_dim)
        self.assertEqual(layer.postprocessing.gate.option, LayerGateOptions.MULTIPLIER)
        self.assertEqual(layer.halting.config, cfg.halting_config)
        self.assertEqual(layer.memory.config, cfg.memory_config)
        self.assertEqual(layer.cfg.layer_model_config, cfg.layer_model_config)
        self.assertFalse(layer.halting.is_terminal)
        for retired_attribute in (
            "activation_function",
            "dropout_module",
            "gate_model",
            "halting_model",
            "layer_norm_module",
            "memory_model",
            "_maybe_apply_memory_by_position",
            "residual_connection",
        ):
            with self.subTest(retired_attribute=retired_attribute):
                self.assertFalse(hasattr(layer, retired_attribute))

    def test_pipeline_delegates_do_not_declare_protected_helpers(self):
        layer = Layer(self.preset(input_dim=8, output_dim=8))

        for delegate in (
            layer.postprocessing,
            layer.halting,
            layer.memory,
            layer.residual,
            layer.normalization,
        ):
            protected_methods = sorted(
                method_name
                for method_name, member in vars(type(delegate)).items()
                if method_name.startswith("_")
                and "__" not in method_name
                and inspect.isroutine(member)
            )
            with self.subTest(delegate=type(delegate).__name__):
                self.assertEqual(protected_methods, [])

    def test_init_with_overrides(self):
        cfg = self.preset(input_dim=12, output_dim=8)
        overrides = LayerConfig(input_dim=16, output_dim=32)
        layer = Layer(cfg, overrides)

        self.assertEqual(layer.input_dim, 16)
        self.assertEqual(layer.output_dim, 32)
        self.assertEqual(layer.postprocessing.activation_function, cfg.activation)
        self.assertEqual(
            layer.postprocessing.dropout_probability,
            cfg.dropout_probability,
        )

    def test_init_raises_on_missing_required_config_fields(self):
        required_fields = [
            "input_dim",
            "output_dim",
            "activation",
            "dropout_probability",
            "layer_norm_position",
        ]

        for field_name in required_fields:
            with self.subTest(field_name=field_name):
                cfg = self.bare_config()
                setattr(cfg, field_name, None)

                with self.assertRaisesRegex(ValueError, field_name):
                    Layer(cfg)

        with self.assertRaisesRegex(ValueError, "layer_model_config is required"):
            Layer(self.bare_config(layer_model_config=None))

        cfg = self.bare_config()
        cfg.residual_config = ResidualConfig()
        with self.assertRaisesRegex(ValueError, "concrete residual config"):
            Layer(cfg)

    def test_init_raises_on_wrong_config_field_types(self):
        cases = [
            ("input_dim", "4", TypeError, "input_dim"),
            ("output_dim", "4", TypeError, "output_dim"),
            ("activation", object(), TypeError, "activation"),
            (
                "residual_config",
                object(),
                TypeError,
                "residual_config",
            ),
            ("dropout_probability", "0.2", TypeError, "dropout_probability"),
            ("layer_norm_position", object(), TypeError, "layer_norm_position"),
            ("normalization", "rms-norm", TypeError, "NormalizationOptions"),
            ("normalization", 0, TypeError, "NormalizationOptions"),
            ("normalization", False, TypeError, "NormalizationOptions"),
            (
                "normalization",
                ActivationOptions.RELU,
                TypeError,
                "NormalizationOptions",
            ),
            ("layer_model_config", object(), TypeError, "ConfigBase"),
            ("gate_config", object(), TypeError, "GateConfig"),
            ("halting_config", object(), TypeError, "HaltingConfig"),
            ("memory_config", object(), TypeError, "DynamicMemoryConfig"),
        ]

        for field_name, value, error_type, pattern in cases:
            with self.subTest(field_name=field_name):
                cfg = self.bare_config()
                setattr(cfg, field_name, value)

                with self.assertRaisesRegex(error_type, pattern):
                    Layer(cfg)

    def test_init_raises_on_invalid_numeric_config_values(self):
        cases = [
            ("input_dim", 0, "input_dim"),
            ("output_dim", 0, "output_dim"),
            ("dropout_probability", -0.1, "dropout_probability"),
            ("dropout_probability", 1.1, "dropout_probability"),
        ]

        for field_name, value, pattern in cases:
            with self.subTest(field_name=field_name, value=value):
                cfg = self.bare_config()
                setattr(cfg, field_name, value)

                with self.assertRaisesRegex(ValueError, pattern):
                    Layer(cfg)

    def test_non_default_halting_minimum_requires_owner_step_semantics(self):
        halting_config = self._halting_config(dim=4)
        halting_config.min_steps = 2

        with self.assertRaisesRegex(ValueError, "min_steps.*LayerConfig"):
            Layer(self.bare_config(halting_config=halting_config))

    def test_residual_connection_rejects_mismatched_dimensions(self):
        residual_options = [
            AdditiveResidualConfig,
            WeightedResidualConfig,
            WeightedBlendResidualConfig,
        ]

        for option in residual_options:
            with self.subTest(option=option):
                cfg = self.bare_config(
                    input_dim=3,
                    output_dim=4,
                    residual_connection_option=option,
                )

                with self.assertRaisesRegex(
                    ValueError, "input_dim and output_dim must be equal"
                ):
                    Layer(cfg)

    def test_layer_norm_rejects_spatial_model_config(self):
        dim = 3
        spatial_config = Conv2dLayerConfig(
            input_dim=dim,
            output_dim=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_flag=True,
        )
        invalid_positions = [
            LayerNormPositionOptions.BEFORE,
            LayerNormPositionOptions.DEFAULT,
            LayerNormPositionOptions.AFTER,
        ]

        for position in invalid_positions:
            with self.subTest(position=position):
                cfg = self.bare_config(
                    input_dim=dim,
                    output_dim=dim,
                    layer_norm_position=position,
                    layer_model_config=spatial_config,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "layer_norm_position must be DISABLED",
                ):
                    Layer(cfg)

        layer = Layer(
            self.bare_config(
                input_dim=dim,
                output_dim=dim,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                layer_model_config=spatial_config,
            )
        )
        self.assertIsNone(layer.normalization.module)

    def test_residual_connection_rejects_strided_spatial_model(self):
        dim = 3
        spatial_config = Conv2dLayerConfig(
            input_dim=dim,
            output_dim=dim,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_flag=True,
        )

        with self.assertRaisesRegex(ValueError, "stride > 1"):
            Layer(
                self.bare_config(
                    input_dim=dim,
                    output_dim=dim,
                    residual_connection_option=AdditiveResidualConfig,
                    layer_model_config=spatial_config,
                )
            )

        layer = Layer(
            self.bare_config(
                input_dim=dim,
                output_dim=dim,
                residual_connection_option=None,
                layer_model_config=spatial_config,
            )
        )
        self.assertIsNone(layer.residual.connection)

    def test_mark_as_last_layer_is_private_stack_integration(self):
        layer = Layer(self.preset())

        self.assertFalse(layer.halting.is_terminal)
        layer._mark_as_last_layer()
        self.assertTrue(layer.halting.is_terminal)
        self.assertFalse(hasattr(layer, "mark_as_last_layer"))

    def test_disabled_residual_connection_builds_no_module(self):
        disabled_layer = Layer(
            self.preset(
                input_dim=8,
                output_dim=8,
                residual_connection_option=None,
            )
        )
        enabled_layer = Layer(
            self.preset(
                input_dim=8,
                output_dim=8,
                residual_connection_option=AdditiveResidualConfig,
            )
        )

        self.assertIsNone(disabled_layer.residual.connection)
        self.assertIsNotNone(enabled_layer.residual.connection)
        self.assertEqual(
            type(enabled_layer.residual.connection.cfg),
            AdditiveResidualConfig,
        )

    def test_layer_state_contains_layer_and_stack_execution_fields(self):
        state_fields = fields(LayerState)

        self.assertEqual(
            [field.name for field in state_fields],
            ["hidden", "loss", "halting_state", "residual_state"],
        )
        self.assertTrue(state_fields[-1].kw_only)

    def test_layer_state_scopes_a_residual_state_replacement(self):
        enclosing_residual_state = object()
        replacement_residual_state = object()
        state = LayerState(
            hidden=torch.ones(1, 2),
            residual_state=enclosing_residual_state,
        )

        with state.scoped_residual_state(replacement_residual_state):
            self.assertIs(state.residual_state, replacement_residual_state)

        self.assertIs(state.residual_state, enclosing_residual_state)

    def test_layer_state_restores_nested_residual_scopes_in_lifo_order(self):
        enclosing_residual_state = object()
        outer_residual_state = object()
        inner_residual_state = object()
        state = LayerState(
            hidden=torch.ones(1, 2),
            residual_state=enclosing_residual_state,
        )

        with state.scoped_residual_state(outer_residual_state):
            with state.scoped_residual_state(inner_residual_state):
                self.assertIs(state.residual_state, inner_residual_state)
            self.assertIs(state.residual_state, outer_residual_state)

        self.assertIs(state.residual_state, enclosing_residual_state)

    def test_layer_state_restores_residual_state_after_an_exception(self):
        enclosing_residual_state = object()
        state = LayerState(
            hidden=torch.ones(1, 2),
            residual_state=enclosing_residual_state,
        )

        with self.assertRaisesRegex(RuntimeError, "deliberate scope failure"):
            with state.scoped_residual_state(None):
                raise RuntimeError("deliberate scope failure")

        self.assertIs(state.residual_state, enclosing_residual_state)

    def test_layer_forward_accepts_only_layer_state(self):
        self.assertEqual(
            tuple(inspect.signature(Layer.forward).parameters),
            ("self", "state"),
        )

    def test_shared_controller_configs_belong_to_layer_stack_config(self):
        layer_fields = {field.name for field in fields(LayerConfig)}
        stack_fields = {field.name for field in fields(LayerStackConfig)}

        for field_name in (
            "shared_gate_config",
            "shared_halting_config",
            "shared_memory_config",
        ):
            with self.subTest(field_name=field_name):
                self.assertNotIn(field_name, layer_fields)
                self.assertIn(field_name, stack_fields)

    def test_build_model_remains_direct_on_layer(self):
        cfg = self.preset()
        layer = Layer(cfg)
        self.assertEqual(layer.model.input_dim, cfg.input_dim)
        self.assertEqual(layer.model.output_dim, cfg.output_dim)

    def test_postprocessing_delegate_builds_optional_gate(self):
        enabled = Layer(self.preset(input_dim=8, output_dim=8))
        disabled = Layer(self.bare_config())

        self.assertIsInstance(enabled.postprocessing.gate, LayerGate)
        self.assertIsInstance(enabled.postprocessing.gate.model, LayerStack)
        self.assertIsNone(disabled.postprocessing.gate)

    def test_halting_delegate_builds_optional_model(self):
        dim = 4
        enabled = Layer(
            self.bare_config(
                input_dim=dim,
                output_dim=dim,
                halting_config=self._halting_config(dim),
            )
        )
        disabled = Layer(self.bare_config(input_dim=dim, output_dim=dim))

        self.assertIsNotNone(enabled.halting.model)
        self.assertIsNone(disabled.halting.model)

    def test_postprocessing_delegate_initializes_dropout(self):
        dropout_probabilities = [0.0, 0.2, 0.5]

        for dropout_probability in dropout_probabilities:
            message = f"dropout_probability={dropout_probability}"
            with self.subTest(msg=message):
                cfg = self.preset(dropout_probability=dropout_probability)
                layer = Layer(cfg)
                result = layer.postprocessing.dropout

                if dropout_probability > 0.0:
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result, nn.Dropout)
                else:
                    self.assertIsNone(result)

    def test_normalization_delegate_resolves_dimension(self):
        positions = [
            (LayerNormPositionOptions.DISABLED, None),
            (LayerNormPositionOptions.BEFORE, 12),
            (LayerNormPositionOptions.DEFAULT, 8),
            (LayerNormPositionOptions.AFTER, 8),
        ]
        for position, expected_dim in positions:
            message = f"position={position}, expected_dim={expected_dim}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=12,
                    output_dim=8,
                    layer_norm_position=position,
                )
                layer = Layer(cfg)
                result = layer.normalization.dimension

                self.assertEqual(result, expected_dim)
                if expected_dim is not None:
                    self.assertIsInstance(layer.normalization.module, nn.RMSNorm)
                    self.assertEqual(layer.normalization.module.eps, 1e-5)
                    self.assertEqual(
                        set(layer.normalization.module.state_dict()), {"weight"}
                    )

    def test_normalization_options_round_trip_by_name(self):
        for option in NormalizationOptions:
            with self.subTest(option=option):
                name = NormalizationOptions.cli_name(option.name)
                self.assertIs(NormalizationOptions.get_member(name), option)

    def test_normalization_selection_survives_partial_config_overrides(self):
        cfg = self.bare_config(layer_norm_position=LayerNormPositionOptions.BEFORE)
        cfg.normalization = NormalizationOptions.LAYER_NORM

        inherited = Layer(cfg, LayerConfig(output_dim=6))
        overridden = Layer(
            cfg, LayerConfig(normalization=NormalizationOptions.RMS_NORM)
        )

        self.assertIsInstance(inherited.normalization.module, nn.LayerNorm)
        self.assertEqual(inherited.normalization.module.normalized_shape, (4,))
        self.assertIsInstance(overridden.normalization.module, nn.RMSNorm)
        self.assertIs(cfg.normalization, NormalizationOptions.LAYER_NORM)

    def test_postprocessing_delegate_stores_activation_option(self):
        activations = [
            ActivationOptions.DISABLED,
            ActivationOptions.RELU,
            ActivationOptions.GELU,
            ActivationOptions.SIGMOID,
        ]
        for activation in activations:
            message = f"activation={activation}"
            with self.subTest(msg=message):
                cfg = self.preset(activation=activation)
                layer = Layer(cfg)
                self.assertEqual(
                    layer.postprocessing.activation_function,
                    activation,
                )

    def test_forward_exposes_overridable_layer_lifecycle_hooks(self):
        lifecycle_events = []

        class LifecycleLayer(Layer):
            def _handle_layer_input(self, state):
                lifecycle_events.append("input")
                return super()._handle_layer_input(state)

            def _handle_layer_processing(self, state):
                lifecycle_events.append("processing")
                return super()._handle_layer_processing(state)

            def _handle_layer_output(self, state):
                lifecycle_events.append("output")
                return super()._handle_layer_output(state)

        layer = LifecycleLayer(self.bare_config())
        state = LayerState(hidden=torch.randn(2, 4))

        result = layer(state)

        self.assertIs(result, state)
        self.assertEqual(lifecycle_events, ["input", "processing", "output"])

    def test_forward_output_shape(self):
        batch_size = 4
        input_dims = [8, 12]
        output_dims = [8, 16]

        for input_dim in input_dims:
            for output_dim in output_dims:
                message = f"input_dim={input_dim}, output_dim={output_dim}"
                with self.subTest(msg=message):
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        residual_connection_option=(
                            AdditiveResidualConfig if input_dim == output_dim else None
                        ),
                    )
                    layer = Layer(cfg)
                    x = torch.randn(batch_size, input_dim)
                    state = LayerState(hidden=x)
                    output = layer(state)

                    self.assertEqual(output.hidden.shape, (batch_size, output_dim))

    def test_forward_pipeline_disabled_returns_exact_affine_hidden(self):
        cfg = self.bare_config(input_dim=3, output_dim=2, bias_flag=True)
        layer = Layer(cfg)
        weight = torch.tensor(
            [
                [1.0, 2.0],
                [0.5, -1.0],
                [-2.0, 0.25],
            ]
        )
        bias = torch.tensor([0.1, -0.2])
        with torch.no_grad():
            layer.model.weight_params.copy_(weight)
            layer.model.bias_params.copy_(bias)
        x = torch.tensor(
            [
                [2.0, -1.0, 0.5],
                [-3.0, 4.0, 1.5],
            ]
        )
        original_x = x.clone()
        existing_loss = torch.tensor(7.0)
        state = LayerState(hidden=x, loss=existing_loss)

        result = layer(state)

        expected = original_x @ weight + bias
        self.assertIs(result, state)
        torch.testing.assert_close(result.hidden, expected)
        torch.testing.assert_close(x, original_x)
        self.assertIs(result.loss, existing_loss)
        self.assertIsNone(result.halting_state)

    def test_forward_composes_model_activation_gate_eval_dropout_residual_in_order(
        self,
    ):
        dim = 2
        cfg = self.bare_config(
            input_dim=dim,
            output_dim=dim,
            activation=ActivationOptions.RELU,
            residual_connection_option=AdditiveResidualConfig,
            dropout_probability=0.5,
            gate_config=self.gate_stack_config(dim),
        )
        layer = Layer(cfg)
        layer.eval()
        with torch.no_grad():
            layer.model.weight_params.copy_(torch.eye(dim))
            layer.model.bias_params.copy_(torch.tensor([1.0, -5.0]))
        gate_values = torch.tensor([[2.0, 3.0]])
        layer.postprocessing.gate.model = ConstantGate(gate_values)
        x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        state = LayerState(hidden=x)

        result = layer(state)

        model_output = x + torch.tensor([1.0, -5.0])
        activated = torch.relu(model_output)
        gate = torch.sigmoid(gate_values.expand_as(activated))
        expected = gate * activated + x
        torch.testing.assert_close(
            layer.postprocessing.gate.model.received_hidden, activated
        )
        torch.testing.assert_close(result.hidden, expected)

    def test_forward_training_dropout_zeros_current_before_residual(self):
        dim = 3
        cases = [
            (None, torch.zeros(2, dim)),
            (
                AdditiveResidualConfig,
                torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]),
            ),
        ]

        for residual_option, expected in cases:
            with self.subTest(residual_option=residual_option):
                cfg = self.bare_config(
                    input_dim=dim,
                    output_dim=dim,
                    residual_connection_option=residual_option,
                    dropout_probability=1.0,
                    layer_model_config=LinearLayerConfig(bias_flag=False),
                )
                layer = Layer(cfg)
                layer.train()
                with torch.no_grad():
                    layer.model.weight_params.copy_(torch.eye(dim))
                x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])

                result = layer(LayerState(hidden=x.clone()))

                torch.testing.assert_close(result.hidden, expected)

    def test_forward_layer_norm_before_uses_original_hidden_for_residual(self):
        dim = 3
        cfg = self.bare_config(
            input_dim=dim,
            output_dim=dim,
            residual_connection_option=AdditiveResidualConfig,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        )
        layer = Layer(cfg)
        with torch.no_grad():
            layer.model.weight_params.copy_(torch.eye(dim))
        x = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.0, 2.0]])

        result = layer(LayerState(hidden=x.clone()))

        expected = layer.normalization.module(x) + x
        torch.testing.assert_close(result.hidden, expected)

    def test_forward_layer_norm_after_normalizes_after_residual(self):
        dim = 3
        cfg = self.bare_config(
            input_dim=dim,
            output_dim=dim,
            residual_connection_option=AdditiveResidualConfig,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        )
        layer = Layer(cfg)
        with torch.no_grad():
            layer.model.weight_params.copy_(2.0 * torch.eye(dim))
        x = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.0, 2.0]])

        result = layer(LayerState(hidden=x.clone()))

        expected = layer.normalization.module(2.0 * x + x)
        torch.testing.assert_close(result.hidden, expected)

    def test_postprocessing_delegate_applies_configured_activation(self):
        batch_size = 4
        input_dim = 8
        activations = [
            ActivationOptions.DISABLED,
            ActivationOptions.RELU,
            ActivationOptions.GELU,
            ActivationOptions.SIGMOID,
        ]
        for activation in activations:
            message = f"activation={activation}"
            with self.subTest(msg=message):
                cfg = self.bare_config(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    activation=activation,
                    dropout_probability=0.0,
                    gate_config=None,
                )
                layer = Layer(cfg)
                x = torch.randn(batch_size, input_dim)
                result = layer.postprocessing.process(LayerState(hidden=x)).hidden

                self.assertEqual(result.shape, (batch_size, input_dim))
                if activation == ActivationOptions.DISABLED:
                    self.assertTrue(torch.equal(result, x))
                else:
                    expected = activation(x)
                    self.assertTrue(torch.equal(result, expected))

    def test_residual_delegate_applies_configured_connection(self):
        batch_size = 4
        dim = 12
        option_cases = [
            (
                None,
                None,
                lambda current, previous: current,
            ),
            (
                AdditiveResidualConfig,
                None,
                lambda current, previous: current + previous,
            ),
            (
                WeightedResidualConfig,
                None,
                lambda current, previous: previous,
            ),
            (
                WeightedBlendResidualConfig,
                None,
                lambda current, previous: 0.9 * current + 0.1 * previous,
            ),
            (
                WeightedResidualConfig,
                LinearLayerConfig(bias_flag=True),
                None,
            ),
            (
                WeightedBlendResidualConfig,
                LinearLayerConfig(bias_flag=True),
                None,
            ),
        ]

        for option, residual_model_config, expected_fn in option_cases:
            message = (
                f"residual_connection_option={option}, "
                f"data_dependent={residual_model_config is not None}"
            )
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=dim,
                    output_dim=dim,
                    residual_connection_option=option,
                    residual_model_config=residual_model_config,
                )
                layer = Layer(cfg)
                x = torch.randn(batch_size, dim)
                model_output = torch.randn(batch_size, dim)
                state = LayerState(hidden=model_output)
                result = layer.residual.apply_residual(state, x)
                if residual_model_config is None:
                    expected = expected_fn(model_output, x)
                else:
                    raw_coefficients = layer.residual.connection.model(
                        torch.cat((model_output, x), dim=-1)
                    )
                    if option is WeightedResidualConfig:
                        expected = x + torch.tanh(raw_coefficients) * model_output
                    else:
                        blend = torch.sigmoid(raw_coefficients)
                        expected = blend * model_output + (1.0 - blend) * x

                self.assertIs(result, state)
                torch.testing.assert_close(result.hidden, expected)

    def test_data_dependent_weighted_blend_uses_model_predictions_for_feature_last_inputs(
        self,
    ):
        output_dim = 3
        connection = WeightedBlendResidualConfig(
            residual_dim=output_dim,
            model_config=LinearLayerConfig(bias_flag=True),
        ).build()
        current = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[-1.0, -2.0, -3.0], [0.5, 1.5, 2.5]],
            ]
        )
        previous = torch.tensor(
            [
                [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
                [[2.0, 1.0, 0.0], [-0.5, -1.5, -2.5]],
            ]
        )

        result = connection(current, previous)

        raw_coefficients = connection.model(torch.cat((current, previous), dim=-1))
        blend = torch.sigmoid(raw_coefficients)
        expected = blend * current + (1.0 - blend) * previous
        torch.testing.assert_close(result, expected)
        self.assertEqual(
            tuple(connection.state_dict()),
            ("model.weight_params", "model.bias_params"),
        )

    def test_data_dependent_residual_builds_its_model_from_configuration(self):
        output_dim = 3
        model_config = LinearLayerConfig(
            input_dim=99,
            output_dim=99,
            bias_flag=True,
        )

        layer = Layer(
            self.bare_config(
                input_dim=output_dim,
                output_dim=output_dim,
                residual_connection_option=WeightedBlendResidualConfig,
                residual_model_config=model_config,
            )
        )
        connection = layer.residual.connection

        self.assertEqual(connection.model_config, model_config)
        self.assertIsNot(connection.model_config, model_config)
        self.assertEqual(connection.model.input_dim, output_dim * 2)
        self.assertEqual(connection.model.output_dim, output_dim)

    def test_data_dependent_weighted_blend_routes_by_position_and_feature(self):
        output_dim = 2
        connection = WeightedBlendResidualConfig(
            residual_dim=output_dim,
            model_config=LinearLayerConfig(bias_flag=True),
        ).build()
        with torch.no_grad():
            current_feature_weights = torch.eye(output_dim)
            previous_feature_weights = -torch.eye(output_dim)
            connection.model.weight_params.copy_(
                torch.cat(
                    (current_feature_weights, previous_feature_weights),
                    dim=0,
                )
            )
            connection.model.bias_params.zero_()
        current = torch.tensor(
            [
                [[2.0, -1.0], [0.0, 3.0]],
                [[-2.0, 1.0], [4.0, -3.0]],
            ]
        )
        previous = torch.tensor(
            [
                [[0.0, 1.0], [2.0, -1.0]],
                [[1.0, -2.0], [-1.0, 2.0]],
            ]
        )

        result = connection(current, previous)

        current_blend_coefficients = torch.sigmoid(current - previous)
        previous_blend_coefficients = 1.0 - current_blend_coefficients
        expected = (
            current_blend_coefficients * current
            + previous_blend_coefficients * previous
        )
        torch.testing.assert_close(result, expected)

    def test_data_dependent_weighted_blend_preserves_all_gradients(self):
        connection = WeightedBlendResidualConfig(
            residual_dim=2,
            model_config=LinearLayerConfig(bias_flag=True),
        ).build()
        current = torch.tensor(
            [[2.0, 3.0], [4.0, 5.0]],
            requires_grad=True,
        )
        previous = torch.tensor(
            [[0.5, 1.0], [1.5, 2.0]],
            requires_grad=True,
        )

        connection(current, previous).square().sum().backward()

        gradients = (
            current.grad,
            previous.grad,
            connection.model.weight_params.grad,
            connection.model.bias_params.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue(torch.any(gradient.abs() > 0))

    def test_data_dependent_residual_requires_a_positive_integer_dimension(self):
        invalid_dimensions = (
            (None, TypeError),
            (True, TypeError),
            (0, ValueError),
        )

        for output_dim, error_type in invalid_dimensions:
            with self.subTest(output_dim=output_dim):
                with self.assertRaisesRegex(error_type, "residual_dim"):
                    WeightedBlendResidualConfig(
                        residual_dim=output_dim,
                        model_config=LinearLayerConfig(bias_flag=True),
                    ).build()

    def test_data_dependent_residual_rejects_an_unsupported_model_config(self):
        with self.assertRaisesRegex(TypeError, "model_config"):
            WeightedBlendResidualConfig(
                residual_dim=2,
                model_config=object(),
            ).build()

    def test_attention_residual_requires_a_positive_integer_dimension(self):
        invalid_dimensions = (
            (None, ValueError),
            (True, ValueError),
            (0, ValueError),
        )

        for residual_dim, error_type in invalid_dimensions:
            with self.subTest(residual_dim=residual_dim):
                with self.assertRaisesRegex(error_type, "residual_dim"):
                    AttentionResidualConfig(residual_dim=residual_dim).build()

    def test_attention_residual_defaults_to_full_depth_mixing(self):
        connection = AttentionResidualConfig(residual_dim=2).build()
        initial = torch.tensor([[2.0, 6.0]])
        current = torch.tensor([[4.0, 10.0]])
        residual_state = connection.new_state(initial)

        result = connection(current, initial, residual_state=residual_state)

        torch.testing.assert_close(result, (initial + current) / 2.0)
        self.assertEqual(connection.block_size, 1)
        self.assertEqual(connection.rms_norm_epsilon, 1e-6)
        self.assertFalse(hasattr(connection, "raw_weight"))
        self.assertFalse(hasattr(connection, "model"))
        self.assertEqual(
            tuple(connection.state_dict()),
            (
                "query",
                "key_norm.weight",
            ),
        )

    def test_attention_residual_build_applies_dimension_override_without_mutation(self):
        attention_config = AttentionResidualConfig(
            residual_dim=99,
            block_size=3,
            rms_norm_epsilon=1e-5,
        )
        connection = attention_config.build(
            overrides=AttentionResidualConfig(residual_dim=4)
        )

        self.assertEqual(attention_config.residual_dim, 99)
        self.assertEqual(connection.residual_dim, 4)
        self.assertEqual(connection.block_size, 3)
        self.assertEqual(connection.rms_norm_epsilon, 1e-5)

    def test_attention_residual_requires_explicit_forward_local_state(self):
        connection = AttentionResidualConfig(residual_dim=2).build()
        hidden = torch.ones(1, 2)

        with self.assertRaisesRegex(
            TypeError, "residual_state.*AttentionResidualState"
        ):
            connection(hidden, hidden)

    def test_pairwise_residual_uses_the_default_stateless_lifecycle(self):
        connection = AdditiveResidualConfig().build()

        self.assertIsNone(connection.new_state(torch.ones(1, 2)))

    def test_legacy_residual_ignores_optional_attention_residual_state(self):
        connection = AdditiveResidualConfig().build()
        current = torch.tensor([[2.0, 3.0]])
        previous = torch.tensor([[5.0, 7.0]])

        result = connection(
            current,
            previous,
            residual_state=object(),
        )

        torch.testing.assert_close(result, current + previous)

    def test_existing_weighted_residual_state_is_unchanged_by_residual_dimension(self):
        connection = WeightedBlendResidualConfig(residual_dim=4).build()

        self.assertEqual(tuple(connection.state_dict()), ("raw_weight",))
        self.assertIsNone(connection.model)
        self.assertIsNone(connection.model_config)

    def test_residual_options_initialize_only_their_owned_components(self):
        model_config = LinearLayerConfig(bias_flag=True)
        cases = (
            (AdditiveResidualConfig, None, False, False),
            (WeightedResidualConfig, None, True, False),
            (WeightedBlendResidualConfig, None, True, False),
            (WeightedResidualConfig, model_config, False, True),
            (WeightedBlendResidualConfig, model_config, False, True),
        )

        for option, coefficient_model_config, owns_raw_weight, owns_model in cases:
            with self.subTest(
                option=option,
                data_dependent=coefficient_model_config is not None,
            ):
                if option is AdditiveResidualConfig:
                    connection = option(residual_dim=4).build()
                else:
                    connection = option(
                        residual_dim=4,
                        model_config=coefficient_model_config,
                    ).build()

                self.assertEqual(
                    getattr(connection, "raw_weight", None) is not None,
                    owns_raw_weight,
                )
                self.assertEqual(
                    getattr(connection, "model", None) is not None,
                    owns_model,
                )

    def test_weighted_residual_rejects_a_missing_coefficient_source(self):
        connection = WeightedResidualConfig().build()
        connection.raw_weight = None
        hidden = torch.ones(2, 3)

        with self.assertRaisesRegex(
            RuntimeError,
            "weighted residual requires either raw_weight or a coefficient model",
        ):
            connection(hidden, hidden)

    def test_data_dependent_weighted_residual_uses_model_coefficients(self):
        output_dim = 2
        connection = WeightedResidualConfig(
            residual_dim=output_dim,
            model_config=LinearLayerConfig(bias_flag=True),
        ).build()
        with torch.no_grad():
            connection.model.weight_params.copy_(
                torch.cat((torch.eye(output_dim), -torch.eye(output_dim)), dim=0)
            )
            connection.model.bias_params.zero_()
        current = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
        previous = torch.tensor([[0.0, 1.0], [-0.5, 2.0]])

        result = connection(current, previous)

        expected = previous + torch.tanh(current - previous) * current
        torch.testing.assert_close(result, expected)

    def test_weighted_residual_uses_tanh_constrained_scalar(self):
        dim = 4
        cfg = self.preset(
            input_dim=dim,
            output_dim=dim,
            residual_connection_option=WeightedResidualConfig,
        )
        layer = Layer(cfg)
        layer.residual.connection.raw_weight.data.fill_(0.5)
        current = torch.full((2, dim), 3.0)
        previous = torch.full((2, dim), 2.0)
        state = LayerState(hidden=current)

        result = layer.residual.apply_residual(state, previous)

        expected = previous + torch.tanh(torch.tensor(0.5)) * current
        self.assertIs(result, state)
        torch.testing.assert_close(result.hidden, expected)

    def test_forward_backward_reaches_model_and_gate_parameters(self):
        dim = 3
        cfg = self.bare_config(
            input_dim=dim,
            output_dim=dim,
            gate_config=self.gate_stack_config(dim),
            layer_model_config=LinearLayerConfig(bias_flag=False),
        )
        layer = Layer(cfg)
        with torch.no_grad():
            layer.model.weight_params.copy_(2.0 * torch.eye(dim))
            layer.postprocessing.gate.model[0].model.weight_params.copy_(
                0.5 * torch.eye(dim)
            )
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
            requires_grad=True,
        )

        result = layer(LayerState(hidden=x))
        result.hidden.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad.abs() > 0))
        self.assertIsNotNone(layer.model.weight_params.grad)
        self.assertTrue(torch.any(layer.model.weight_params.grad.abs() > 0))
        gate_weight_grad = layer.postprocessing.gate.model[0].model.weight_params.grad
        self.assertIsNotNone(gate_weight_grad)
        self.assertTrue(torch.any(gate_weight_grad.abs() > 0))

    def test_forward_backward_reaches_addition_gate_parameters(self):
        dim = 3
        cfg = self.bare_config(
            input_dim=dim,
            output_dim=dim,
            gate_config=self.gate_stack_config(dim),
            gate_option=LayerGateOptions.ADDITION,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        )
        layer = Layer(cfg)
        with torch.no_grad():
            layer.model.weight_params.copy_(2.0 * torch.eye(dim))
            layer.postprocessing.gate.model[0].model.weight_params.copy_(
                0.5 * torch.eye(dim)
            )
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
            requires_grad=True,
        )

        result = layer(LayerState(hidden=x))
        result.hidden.sum().backward()

        gate_weight_grad = layer.postprocessing.gate.model[0].model.weight_params.grad
        self.assertIsNotNone(gate_weight_grad)
        self.assertTrue(torch.any(gate_weight_grad.abs() > 0))

    def test_forward_backward_reaches_weighted_residual_parameters(self):
        dim = 3
        residual_options = [
            WeightedResidualConfig,
            WeightedBlendResidualConfig,
        ]

        for residual_option in residual_options:
            with self.subTest(residual_option=residual_option):
                cfg = self.bare_config(
                    input_dim=dim,
                    output_dim=dim,
                    residual_connection_option=residual_option,
                    layer_model_config=LinearLayerConfig(bias_flag=False),
                )
                layer = Layer(cfg)
                with torch.no_grad():
                    layer.model.weight_params.copy_(2.0 * torch.eye(dim))
                x = torch.tensor(
                    [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
                    requires_grad=True,
                )

                result = layer(LayerState(hidden=x))
                result.hidden.sum().backward()

                self.assertIsNotNone(layer.residual.connection.raw_weight.grad)
                self.assertTrue(
                    torch.any(layer.residual.connection.raw_weight.grad.abs() > 0)
                )

    def test_postprocessing_dropout_respects_training_mode(self):
        batch_size = 64
        dim = 128
        dropout_probabilities = [0.0, 0.5]
        training_modes = [True, False]

        for dropout_probability in dropout_probabilities:
            for training in training_modes:
                message = (
                    f"dropout_probability={dropout_probability}, training={training}"
                )
                with self.subTest(msg=message):
                    cfg = self.preset(
                        input_dim=dim,
                        output_dim=dim,
                        dropout_probability=dropout_probability,
                    )
                    layer = Layer(cfg)
                    layer.train() if training else layer.eval()
                    x = torch.randn(batch_size, dim)
                    dropout = layer.postprocessing.dropout
                    result = x if dropout is None else dropout(x)

                    self.assertEqual(result.shape, x.shape)
                    if dropout_probability > 0.0 and training:
                        has_zeros = (result == 0.0).any()
                        not_identical = not torch.equal(result, x)
                        self.assertTrue(has_zeros or not_identical)
                    else:
                        self.assertTrue(torch.equal(result, x))

    def test_normalization_delegate_dispatches_by_position(self):
        dim = 12
        positions = [
            LayerNormPositionOptions.DISABLED,
            LayerNormPositionOptions.DEFAULT,
            LayerNormPositionOptions.BEFORE,
            LayerNormPositionOptions.AFTER,
        ]
        methods = [
            ("before_model", LayerNormPositionOptions.BEFORE),
            ("after_model", LayerNormPositionOptions.DEFAULT),
            ("after_residual", LayerNormPositionOptions.AFTER),
        ]
        cases = product(
            (None, *NormalizationOptions), positions, methods, ((4, dim), (2, 3, dim))
        )
        for normalization, position, (method_name, active_position), shape in cases:
            with self.subTest(
                normalization=normalization,
                position=position,
                method=method_name,
                shape=shape,
            ):
                cfg = self.bare_config(
                    input_dim=dim,
                    output_dim=dim,
                    layer_norm_position=position,
                )
                cfg.normalization = normalization
                layer = Layer(cfg)
                x = torch.linspace(
                    -3, 4, math.prod(shape), dtype=torch.float64
                ).reshape(shape)
                x.requires_grad_()
                module = layer.normalization.module
                if module is not None:
                    module.double()
                    with torch.no_grad():
                        module.weight.copy_(torch.linspace(0.5, 1.5, dim))
                        if getattr(module, "bias", None) is not None:
                            module.bias.fill_(0.25)
                method = getattr(layer.normalization, method_name)
                state = LayerState(hidden=x)
                expected = x
                if position == active_position:
                    expected = module(x)
                result = method(state)

                self.assertIs(result, state)
                self.assertEqual(result.hidden.shape, x.shape)
                torch.testing.assert_close(result.hidden, expected)
                actual_gradient = torch.autograd.grad(result.hidden.sum(), x)[0]
                expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
                torch.testing.assert_close(actual_gradient, expected_gradient)
                if position == LayerNormPositionOptions.DISABLED:
                    self.assertIsNone(module)
                if position != active_position:
                    self.assertIs(result.hidden, x)

    def test_postprocessing_delegate_applies_optional_gate(self):
        batch_size = 4
        dim = 8
        cfg_with_gate = self.preset(input_dim=dim, output_dim=dim)
        cfg_without_gate = self.preset(input_dim=dim, output_dim=dim)

        layer_with_gate = Layer(cfg_with_gate)
        layer_without_gate = Layer(cfg_without_gate)
        layer_without_gate.postprocessing.gate = None

        layers = [
            (layer_with_gate, True),
            (layer_without_gate, False),
        ]
        for layer, has_gate in layers:
            message = f"has_gate={has_gate}"
            with self.subTest(msg=message):
                x = torch.randn(batch_size, dim)
                gate_module = layer.postprocessing.gate
                result = x if gate_module is None else gate_module(x)

                self.assertEqual(result.shape, x.shape)
                if has_gate:
                    gate_state = Layer.run_model_from_hidden(
                        layer.postprocessing.gate.model, x
                    )
                    gate_output = gate_state.hidden
                    gate = torch.sigmoid(gate_output)
                    expected = gate * x
                    torch.testing.assert_close(result, expected)
                else:
                    self.assertTrue(torch.equal(result, x))

    def test_gate_output_is_applied_as_sigmoid_multiplier(self):
        dim = 3
        layer = Layer(
            self.bare_config(
                input_dim=dim,
                output_dim=dim,
                gate_config=self.gate_stack_config(dim),
            )
        )
        gate_values = torch.tensor([[-1.0, 0.0, 2.0]])
        layer.postprocessing.gate.model = ConstantGate(gate_values)
        x = torch.tensor([[2.0, -3.0, 4.0], [-1.0, 5.0, 0.5]])

        result = layer.postprocessing.gate(x)

        gate = torch.sigmoid(gate_values.expand_as(x))
        expected = gate * x
        torch.testing.assert_close(result, expected)

    def test_gate_option_none_is_rejected_when_gate_config_is_provided(self):
        dim = 3
        gate_config = GateConfig(
            model_config=self.gate_stack_config(dim),
            option=None,
        )

        with self.assertRaisesRegex(
            ValueError,
            "LayerConfig.gate_config.option.*LayerGateOptions.*MULTIPLIER",
        ):
            Layer(
                self.bare_config(
                    input_dim=dim,
                    output_dim=dim,
                    gate_config=gate_config,
                )
            )

    def test_missing_gate_config_bypasses_gate_model(self):
        dim = 3
        layer = Layer(
            self.bare_config(
                input_dim=dim,
                output_dim=dim,
            )
        )
        self.assertIsNone(layer.postprocessing.gate)
        x = torch.tensor([[2.0, -3.0, 4.0], [-1.0, 5.0, 0.5]])
        state = LayerState(hidden=x)

        result = layer.postprocessing.process(state)

        self.assertIs(result, state)
        torch.testing.assert_close(result.hidden, x)

    def test_layer_gate_rejects_missing_model_at_forward(self):
        dim = 3
        layer = Layer(
            self.bare_config(
                input_dim=dim,
                output_dim=dim,
                gate_config=self.gate_stack_config(dim),
            )
        )
        layer.postprocessing.gate.model = None

        with self.assertRaisesRegex(ValueError, "LayerGate requires a gate model"):
            layer.postprocessing.gate(torch.ones(2, dim))

    def test_layer_gate_rejects_invalid_model_output(self):
        class ObjectGate(nn.Module):
            def forward(self, state: LayerState):
                return object()

        class WrongShapeGate(nn.Module):
            def forward(self, state: LayerState) -> LayerState:
                hidden = torch.ones(
                    state.hidden.shape[0],
                    state.hidden.shape[-1] + 1,
                    dtype=state.hidden.dtype,
                    device=state.hidden.device,
                )
                return LayerState(hidden=hidden)

        dim = 3
        current = torch.ones(2, dim)
        cases = [
            (ObjectGate(), TypeError, "Tensor or LayerState.hidden Tensor"),
            (WrongShapeGate(), ValueError, "gate output and current shapes"),
        ]

        for gate_model, error_type, message in cases:
            with self.subTest(gate_model=type(gate_model).__name__):
                layer = Layer(
                    self.bare_config(
                        input_dim=dim,
                        output_dim=dim,
                        gate_config=self.gate_stack_config(dim),
                    )
                )
                layer.postprocessing.gate.model = gate_model

                with self.assertRaisesRegex(error_type, message):
                    layer.postprocessing.gate(current)

    def test_layer_gate_options_apply_expected_formula(self):
        current = torch.tensor([[2.0, -3.0, 4.0], [-1.0, 5.0, 0.5]])
        gate_values = torch.tensor([[-1.0, 0.0, 2.0], [0.5, -0.5, 1.0]])
        cases = [
            (LayerGateOptions.MULTIPLIER, None, gate_values * current),
            (
                LayerGateOptions.MULTIPLIER,
                ActivationOptions.SIGMOID,
                torch.sigmoid(gate_values) * current,
            ),
            (
                LayerGateOptions.MULTIPLIER,
                ActivationOptions.TANH,
                torch.tanh(gate_values) * current,
            ),
            (LayerGateOptions.ADDITION, None, current + gate_values),
            (
                LayerGateOptions.ADDITION,
                ActivationOptions.SIGMOID,
                current + torch.sigmoid(gate_values),
            ),
            (
                LayerGateOptions.ADDITION,
                ActivationOptions.TANH,
                current + torch.tanh(gate_values),
            ),
        ]

        for option, gate_activation, expected in cases:
            with self.subTest(option=option, gate_activation=gate_activation):
                layer = Layer(
                    self.bare_config(
                        input_dim=current.shape[-1],
                        output_dim=current.shape[-1],
                        gate_config=GateConfig(
                            model_config=self.gate_stack_config(current.shape[-1]),
                            option=option,
                            activation=gate_activation,
                        ),
                    )
                )
                layer.postprocessing.gate.model = ConstantGate(gate_values)

                result = layer.postprocessing.gate(current)

                torch.testing.assert_close(result, expected)

    def test_layer_gate_options_do_not_require_residual(self):
        current = torch.tensor([[2.0, -3.0, 4.0], [-1.0, 5.0, 0.5]])
        gate_values = torch.tensor([[-1.0, 0.0, 2.0], [0.5, -0.5, 1.0]])
        cases = [
            (LayerGateOptions.MULTIPLIER, torch.sigmoid(gate_values) * current),
            (LayerGateOptions.ADDITION, current + torch.sigmoid(gate_values)),
        ]

        for option, expected in cases:
            with self.subTest(option=option):
                layer = Layer(
                    self.bare_config(
                        input_dim=current.shape[-1],
                        output_dim=current.shape[-1],
                        gate_option=option,
                        gate_activation=ActivationOptions.SIGMOID,
                    )
                )
                layer.postprocessing.gate.model = ConstantGate(gate_values)

                result = layer.postprocessing.gate(current)

                torch.testing.assert_close(result, expected)

    def test_layer_gate_addition_allows_dimension_change(self):
        layer = Layer(
            self.bare_config(
                input_dim=2,
                output_dim=3,
                gate_config=self.gate_stack_config(3),
                gate_option=LayerGateOptions.ADDITION,
                gate_activation=ActivationOptions.SIGMOID,
            )
        )

        self.assertEqual(layer.input_dim, 2)
        self.assertEqual(layer.output_dim, 3)

    def test_layer_gate_multiplier_allows_dimension_change(self):
        layer = Layer(
            self.bare_config(
                input_dim=2,
                output_dim=3,
                gate_config=self.gate_stack_config(3),
                gate_option=LayerGateOptions.MULTIPLIER,
                gate_activation=ActivationOptions.SIGMOID,
            )
        )

        self.assertEqual(layer.input_dim, 2)
        self.assertEqual(layer.output_dim, 3)

    def test_gate_config_rejects_layer_config_subclasses(self):
        class StateAwareLayerConfig(LayerConfig):
            pass

        dim = 4
        gate_config = LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=False,
            layer_config=StateAwareLayerConfig(
                input_dim=dim,
                output_dim=dim,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=True),
            ),
        )

        with self.assertRaises(TypeError):
            Layer(self.preset(input_dim=dim, output_dim=dim, gate_config=gate_config))

    def test_gates_use_plain_state_from_current_hidden(self):
        class SpyGate(nn.Module):
            def __init__(self):
                super().__init__()
                self.received_state = None
                self.received_hidden = None

            def forward(self, state: LayerState) -> LayerState:
                self.received_state = state
                self.received_hidden = state.hidden
                state.hidden = torch.ones_like(state.hidden)
                return state

        batch_size = 2
        dim = 4
        layer = Layer(self.preset(input_dim=dim, output_dim=dim))
        spy = SpyGate()
        layer.postprocessing.gate.model = spy
        x = torch.randn(batch_size, dim)

        result = layer.postprocessing.gate(x)

        self.assertIs(type(spy.received_state), LayerState)
        self.assertIs(spy.received_hidden, x)
        self.assertIsNone(spy.received_state.loss)
        self.assertIsNone(spy.received_state.halting_state)
        torch.testing.assert_close(result, torch.sigmoid(torch.ones_like(x)) * x)

    def test_run_model_from_hidden_static_method_returns_layer_state(self):
        batch_size = 4
        dims = [8, 12]

        for dim in dims:
            message = f"dim={dim}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=dim,
                    output_dim=dim,
                )
                layer = Layer(cfg)
                layer.eval()
                x = torch.randn(batch_size, dim)
                output_state = Layer.run_model_from_hidden(layer, x)

                self.assertIs(type(output_state), LayerState)
                self.assertEqual(output_state.hidden.shape, (batch_size, dim))

        descriptor = inspect.getattr_static(Layer, "run_model_from_hidden")
        self.assertIsInstance(descriptor, staticmethod)
        self.assertFalse(hasattr(Layer, "run_model_returning_hidden"))
        self.assertFalse(hasattr(Layer, "run_model_returning_state"))

    def test_run_model_from_hidden_constructs_plain_state_and_rejects_masks(self):
        class StateSpy(nn.Module):
            def __init__(self):
                super().__init__()
                self.received_state = None

            def forward(self, state: LayerState) -> LayerState:
                self.received_state = state
                return state

        spy = StateSpy()
        x = torch.randn(2, 3)

        result = Layer.run_model_from_hidden(spy, x)

        self.assertIs(result, spy.received_state)
        self.assertIs(type(spy.received_state), LayerState)
        self.assertIs(spy.received_state.hidden, x)

        with self.assertRaises(TypeError):
            Layer.run_model_from_hidden(
                spy,
                x,
                key_padding_mask=torch.ones(2, 3, dtype=torch.bool),
            )
        with self.assertRaises(TypeError):
            Layer.run_model_from_hidden(
                spy,
                x,
                attention_mask=torch.zeros(3, 3),
            )

    def test_forward_output_shape_all_config_combinations(self):
        batch_size = 4
        input_dims = [8, 12]
        output_dims = [8, 16]
        activations = [ActivationOptions.RELU, ActivationOptions.DISABLED]
        residual_options = [
            AdditiveResidualConfig,
            None,
        ]
        dropout_probabilities = [0.0, 0.2]
        layer_norm_positions = [
            LayerNormPositionOptions.DISABLED,
            LayerNormPositionOptions.DEFAULT,
            LayerNormPositionOptions.BEFORE,
            LayerNormPositionOptions.AFTER,
        ]

        for input_dim in input_dims:
            for output_dim in output_dims:
                for activation in activations:
                    for residual_option in residual_options:
                        for dropout in dropout_probabilities:
                            for layer_norm in layer_norm_positions:
                                message = (
                                    f"input_dim={input_dim}, "
                                    f"output_dim={output_dim}, "
                                    f"activation={activation}, "
                                    f"residual_connection_option={residual_option}, "
                                    f"dropout={dropout}, "
                                    f"layer_norm={layer_norm}"
                                )
                                with self.subTest(msg=message):
                                    cfg = self.preset(
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        activation=activation,
                                        residual_connection_option=(
                                            residual_option
                                            if input_dim == output_dim
                                            else None
                                        ),
                                        dropout_probability=dropout,
                                        layer_norm_position=layer_norm,
                                    )
                                    layer = Layer(cfg)
                                    x = torch.randn(batch_size, input_dim)
                                    state = LayerState(hidden=x)
                                    output = layer(state)

                                    self.assertEqual(
                                        output.hidden.shape,
                                        (batch_size, output_dim),
                                    )

    def test_halting_rejects_mismatched_dims(self):
        dim = 8
        halting_config = StickBreakingConfig(
            input_dim=dim,
            threshold=0.99,
            ponder_cost_weight=1.0,
            min_steps=1,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=2,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_postprocessing_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=dim,
                        output_dim=dim,
                        bias_flag=True,
                    ),
                ),
            ),
        )

        invalid_cases = [
            {"input_dim": dim, "output_dim": dim * 2},
            {"input_dim": dim * 2, "output_dim": dim},
        ]

        for case in invalid_cases:
            message = ", ".join(f"{k}={v}" for k, v in case.items())
            with self.subTest(msg=message):
                with self.assertRaises(ValueError):
                    Layer(self.preset(halting_config=halting_config, **case))

    def test_layer_composes_soft_through_the_halting_interface(self):
        dim = 4
        halting_config = SoftHaltingConfig(
            input_dim=dim,
            threshold=0.999,
            ponder_cost_weight=1.0,
            min_steps=1,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        )
        layer = Layer(
            self.preset(
                input_dim=dim,
                output_dim=dim,
                halting_config=halting_config,
            )
        ).eval()
        state = LayerState(hidden=torch.randn(3, dim))

        result = layer(state)

        self.assertIs(result, state)
        self.assertIsInstance(result.halting_state, SoftHaltingState)
        torch.testing.assert_close(
            result.hidden,
            result.halting_state.output_hidden,
        )

    def test_halting_uses_the_common_lifecycle_to_update_state(self):
        batch_size = 4
        dim = 8
        existing_loss = torch.tensor(2.0)
        cfg = self.preset(
            input_dim=dim,
            output_dim=dim,
            activation=ActivationOptions.DISABLED,
            dropout_probability=0.0,
            halting_config=self._halting_config(dim, threshold=1.0),
        )
        layer = Layer(cfg)
        layer.eval()
        state = LayerState(
            hidden=torch.randn(batch_size, dim),
            loss=existing_loss,
        )
        result = layer(state)

        self.assertIs(result, state)
        self.assertIs(result.loss, existing_loss)
        self.assertIsNotNone(result.halting_state)
        torch.testing.assert_close(
            result.hidden,
            result.halting_state.output_hidden,
        )

    def test_halting_last_layer_finalizes_and_accumulates_vector_loss(self):
        dim = 3
        cfg = self.bare_config(
            input_dim=dim,
            output_dim=dim,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        )
        layer = Layer(cfg)
        layer._mark_as_last_layer()
        with torch.no_grad():
            layer.model.weight_params.copy_(torch.eye(dim))
        fake_halting_state = FakeHaltingState(
            halt_mask=torch.tensor([False, False]),
        )
        fake_halting = FakeHaltingModel(fake_halting_state)
        layer.halting.model = fake_halting
        existing_loss = torch.tensor(5.0)
        x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
        state = LayerState(hidden=x.clone(), loss=existing_loss)

        result = layer(state)

        expected_hidden = x + fake_halting.finalized_hidden_offset
        expected_loss = existing_loss + fake_halting.loss.mean()
        self.assertIs(result, state)
        self.assertEqual(fake_halting.update_calls, 1)
        self.assertEqual(fake_halting.finalize_calls, 1)
        torch.testing.assert_close(fake_halting.finalize_input, x)
        torch.testing.assert_close(result.hidden, expected_hidden)
        self.assertIs(result.halting_state, fake_halting_state)
        torch.testing.assert_close(result.loss, expected_loss)

    def test_halting_finalizes_when_all_items_halt_before_last_layer(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            output_dim=dim,
            activation=ActivationOptions.DISABLED,
            dropout_probability=0.0,
            halting_config=self._halting_config(dim),
        )
        layer = Layer(cfg)
        layer.eval()

        state = LayerState(hidden=torch.randn(batch_size, dim))
        result = layer(state)

        self.assertFalse(layer.halting.is_terminal)

        self.assertTrue(result.halting_state.halt_mask.all().item())
        self.assertIsNotNone(result.loss)
        self.assertTrue(
            torch.allclose(result.hidden, result.halting_state.accumulated_hidden)
        )

    def test_halted_state_skip_bypasses_layer_pipeline(self):
        dim = 3
        lifecycle_events = []

        class LifecycleLayer(Layer):
            def _handle_layer_input(self, state):
                lifecycle_events.append("input")
                return super()._handle_layer_input(state)

            def _handle_layer_processing(self, state):
                lifecycle_events.append("processing")
                return super()._handle_layer_processing(state)

            def _handle_layer_output(self, state):
                lifecycle_events.append("output")
                return super()._handle_layer_output(state)

        layer = LifecycleLayer(self.bare_config(input_dim=dim, output_dim=dim))
        layer.model = AddConstantModel(100.0)
        fake_halting_state = FakeHaltingState(halt_mask=torch.ones(2, dtype=torch.bool))
        layer.halting.model = FakeHaltingModel(fake_halting_state)
        hidden = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
        loss = torch.tensor(2.0)
        state = LayerState(
            hidden=hidden,
            loss=loss,
            halting_state=fake_halting_state,
        )

        result = layer(state)

        self.assertIs(result, state)
        self.assertIs(result.hidden, hidden)
        self.assertIs(result.loss, loss)
        self.assertIs(result.halting_state, fake_halting_state)
        self.assertEqual(layer.model.calls, 0)
        self.assertIsNone(layer.postprocessing.gate)
        self.assertEqual(layer.halting.model.update_calls, 0)
        self.assertEqual(lifecycle_events, [])

    def test_halting_skips_layer_when_all_items_already_halted(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            output_dim=dim,
            activation=ActivationOptions.DISABLED,
            dropout_probability=0.0,
            halting_config=self._halting_config(dim),
        )
        first_layer = Layer(cfg)
        first_layer.eval()
        state = first_layer(LayerState(hidden=torch.randn(batch_size, dim)))
        hidden_before_skip = state.hidden.clone()
        loss_before_skip = state.loss.clone()

        layer = Layer(cfg)
        layer.eval()

        result = layer(state)

        self.assertIs(result, state)
        self.assertTrue(result.halting_state.halt_mask.all().item())
        self.assertTrue(torch.equal(result.hidden, hidden_before_skip))
        self.assertTrue(torch.equal(result.loss, loss_before_skip))
