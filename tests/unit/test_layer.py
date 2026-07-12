from dataclasses import fields

import unittest

import torch
import torch.nn as nn

from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.base.layer import (
    Layer,
    LayerConfig,
    LayerStack,
    LayerStackConfig,
    LayerState,
)
from emperor.base.layer.residual import (
    ResidualConnectionOptions,
)
from emperor.base.layer.gate import GateConfig, LayerGate, LayerGateOptions
from emperor.convs.core.config import Conv2dLayerConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.base.options import LastLayerBiasOptions
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions


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
        return self.halting_state, model_hidden_state + self.halting_output_offset

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
        import emperor.base.layer as layer_package

        expected_exports = {
            "LayerState",
            "LayerConfig",
            "LayerStackConfig",
            "GateConfig",
            "RecurrentLayerConfig",
            "LayerGateOptions",
            "ResidualConnectionOptions",
            "Layer",
            "LayerStack",
            "RecurrentLayer",
            "RecurrentLayerMonitorCallback",
            "LayerControllerMonitorCallback",
        }

        self.assertEqual(set(layer_package.__all__), expected_exports)
        for name in expected_exports:
            with self.subTest(name=name):
                self.assertTrue(hasattr(layer_package, name))

    def test_gate_import_paths_are_compatible(self):
        from emperor.base.layer import (
            GateConfig as LayerPackageGateConfig,
            LayerGateOptions as LayerPackageGateOptions,
        )
        from emperor.base.layer.config import GateConfig as LayerConfigGateConfig
        from emperor.base.layer.gate import (
            GateConfig as GatePackageGateConfig,
            LayerGate as GatePackageLayerGate,
            LayerGateOptions as GatePackageGateOptions,
        )

        self.assertIs(GatePackageGateConfig, GateConfig)
        self.assertIs(GatePackageLayerGate, LayerGate)
        self.assertIs(GatePackageGateOptions, LayerGateOptions)
        self.assertIs(LayerPackageGateConfig, GateConfig)
        self.assertIs(LayerPackageGateOptions, LayerGateOptions)
        self.assertIs(LayerConfigGateConfig, GateConfig)

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

    def bare_config(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        bias_flag: bool = True,
        activation: ActivationOptions = ActivationOptions.DISABLED,
        residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.DISABLED
        ),
        dropout_probability: float = 0.0,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        gate_config: "LayerStackConfig | None" = None,
        gate_option: LayerGateOptions | None = None,
        gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID,
        halting_config: "StickBreakingConfig | None" = None,
        memory_config=None,
        layer_model_config=DEFAULT_LAYER_MODEL_CONFIG,
    ) -> LayerConfig:
        if layer_model_config is DEFAULT_LAYER_MODEL_CONFIG:
            layer_model_config = LinearLayerConfig(bias_flag=bias_flag)
        return LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            residual_connection_option=residual_connection_option,
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
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=dim,
                output_dim=dim,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
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
        residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.DISABLED
        ),
        dropout_probability: float = 0.2,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        gate_config: "LayerStackConfig | None" = None,
        gate_num_layers: int = 1,
        gate_activation: ActivationOptions = ActivationOptions.DISABLED,
        gate_residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.DISABLED
        ),
        gate_dropout_probability: float = 0.0,
        gate_bias_flag: bool = True,
        gate_option: LayerGateOptions | None = None,
        halting_config: "StickBreakingConfig | None" = None,
    ) -> LayerConfig:

        if gate_config is None and (input_dim == output_dim or gate_option is not None):
            gate_config = LayerStackConfig(
                hidden_dim=output_dim,
                num_layers=gate_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=gate_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=gate_residual_connection_option,
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
                halting_dropout=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=LayerStackConfig(
                    hidden_dim=output_dim,
                    output_dim=2,
                    num_layers=gate_num_layers,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_pipeline_flag=False,
                    layer_config=LayerConfig(
                        activation=gate_activation,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        residual_connection_option=gate_residual_connection_option,
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
            residual_connection_option=residual_connection_option,
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
            halting_dropout=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=2,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
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

    def test_init_stores_all_config_attributes(self):
        cfg = self.preset(input_dim=8, output_dim=8)
        layer = Layer(cfg)

        self.assertIsInstance(layer, Layer)
        self.assertEqual(layer.input_dim, cfg.input_dim)
        self.assertEqual(layer.output_dim, cfg.output_dim)
        self.assertEqual(layer.activation_function, cfg.activation)
        self.assertEqual(
            layer.residual_connection_option,
            cfg.residual_connection_option,
        )
        self.assertEqual(layer.dropout_probability, cfg.dropout_probability)
        self.assertEqual(layer.layer_norm_position, cfg.layer_norm_position)
        self.assertEqual(layer.gate_config, cfg.gate_config)
        self.assertIsNotNone(layer.gate_model)
        self.assertEqual(layer.gate_model.gate_dim, cfg.output_dim)
        self.assertEqual(layer.gate_model.option, LayerGateOptions.MULTIPLIER)
        self.assertEqual(layer.halting_config, cfg.halting_config)
        self.assertEqual(layer.layer_model_config, cfg.layer_model_config)
        self.assertFalse(layer.last_layer_flag)

    def test_init_with_overrides(self):
        cfg = self.preset(input_dim=12, output_dim=8)
        overrides = LayerConfig(input_dim=16, output_dim=32)
        layer = Layer(cfg, overrides)

        self.assertEqual(layer.input_dim, 16)
        self.assertEqual(layer.output_dim, 32)
        self.assertEqual(layer.activation_function, cfg.activation)
        self.assertEqual(layer.dropout_probability, cfg.dropout_probability)

    def test_init_raises_on_missing_required_config_fields(self):
        required_fields = [
            "input_dim",
            "output_dim",
            "activation",
            "residual_connection_option",
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

    def test_init_raises_on_wrong_config_field_types(self):
        cases = [
            ("input_dim", "4", TypeError, "input_dim"),
            ("output_dim", "4", TypeError, "output_dim"),
            ("activation", object(), TypeError, "activation"),
            (
                "residual_connection_option",
                object(),
                TypeError,
                "residual_connection_option",
            ),
            ("dropout_probability", "0.2", TypeError, "dropout_probability"),
            ("layer_norm_position", object(), TypeError, "layer_norm_position"),
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

    def test_residual_connection_rejects_mismatched_dimensions(self):
        residual_options = [
            ResidualConnectionOptions.RESIDUAL,
            ResidualConnectionOptions.WEIGHTED_RESIDUAL,
            ResidualConnectionOptions.WEIGHTED_BLEND,
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
        self.assertIsNone(layer.layer_norm_module)

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
                    residual_connection_option=ResidualConnectionOptions.RESIDUAL,
                    layer_model_config=spatial_config,
                )
            )

        layer = Layer(
            self.bare_config(
                input_dim=dim,
                output_dim=dim,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                layer_model_config=spatial_config,
            )
        )
        self.assertIsNone(layer.residual_connection)

    def test_mark_as_last_layer(self):
        layer = Layer(self.preset())

        self.assertFalse(layer.last_layer_flag)
        layer.mark_as_last_layer()
        self.assertTrue(layer.last_layer_flag)

    def test_disabled_residual_connection_builds_no_module(self):
        disabled_layer = Layer(
            self.preset(
                input_dim=8,
                output_dim=8,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
            )
        )
        enabled_layer = Layer(
            self.preset(
                input_dim=8,
                output_dim=8,
                residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            )
        )

        self.assertIsNone(disabled_layer.residual_connection)
        self.assertIsNotNone(enabled_layer.residual_connection)
        self.assertEqual(
            enabled_layer.residual_connection.option,
            ResidualConnectionOptions.RESIDUAL,
        )

    def test_layer_state_contains_only_generic_fields(self):
        state_fields = [field.name for field in fields(LayerState)]

        self.assertEqual(state_fields, ["hidden", "loss", "halting_state"])

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

    def test_build_model(self):
        cfg = self.preset()
        layer = Layer(cfg)
        model_configs = [cfg.layer_model_config, None]

        for model_config in model_configs:
            with self.subTest(has_config=model_config is not None):
                layer.layer_model_config = model_config
                result = layer._Layer__build_model()

                if model_config is not None:
                    self.assertIsNotNone(result)
                    self.assertEqual(result.input_dim, cfg.input_dim)
                    self.assertEqual(result.output_dim, cfg.output_dim)
                else:
                    self.assertIsNone(result)

    def test_build_gate(self):
        cfg = self.preset()
        layer = Layer(cfg)
        gate_configs = [cfg.gate_config, None]

        for gate_config in gate_configs:
            message = f"has_config={gate_config is not None}"
            with self.subTest(msg=message):
                layer.gate_config = gate_config
                result = layer._Layer__build_gate()

                if gate_config is not None:
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result, LayerGate)
                    self.assertIsInstance(result.model, LayerStack)
                else:
                    self.assertIsNone(result)

    def test_build_optional_controller_helpers_return_none_without_configs(self):
        dim = 4
        cfg = self.bare_config(
            input_dim=dim,
            output_dim=dim,
            halting_config=self._halting_config(dim),
        )
        layer = Layer(cfg)

        helper_cases = [
            (
                "halting_config",
                "_Layer__build_halting_model",
                cfg.halting_config,
            ),
        ]

        for attr_name, method_name, valid_config in helper_cases:
            with self.subTest(attr_name=attr_name, has_config=True):
                setattr(layer, attr_name, valid_config)
                result = getattr(layer, method_name)()
                self.assertIsNotNone(result)

            with self.subTest(attr_name=attr_name, has_config=False):
                setattr(layer, attr_name, None)
                result = getattr(layer, method_name)()
                self.assertIsNone(result)

    def test_init_dropout_module(self):
        dropout_probabilities = [0.0, 0.2, 0.5]

        for dropout_probability in dropout_probabilities:
            message = f"dropout_probability={dropout_probability}"
            with self.subTest(msg=message):
                cfg = self.preset(dropout_probability=dropout_probability)
                layer = Layer(cfg)
                result = layer._Layer__init_dropout_module()

                if dropout_probability > 0.0:
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result, nn.Dropout)
                    self.assertTrue(layer._Layer__should_apply_dropout())
                else:
                    self.assertIsNone(result)
                    self.assertFalse(layer._Layer__should_apply_dropout())

    def test_resolve_layer_norm_dim(self):
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
                result = layer._Layer__resolve_layer_norm_dim()

                self.assertEqual(result, expected_dim)

    def test_should_apply_activation(self):
        activations = [
            (ActivationOptions.DISABLED, False),
            (ActivationOptions.RELU, True),
            (ActivationOptions.GELU, True),
            (ActivationOptions.SIGMOID, True),
        ]
        for activation, expected in activations:
            message = f"activation={activation}, expected={expected}"
            with self.subTest(msg=message):
                cfg = self.preset(activation=activation)
                layer = Layer(cfg)
                self.assertEqual(layer._Layer__should_apply_activation(), expected)

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
                            ResidualConnectionOptions.RESIDUAL
                            if input_dim == output_dim
                            else ResidualConnectionOptions.DISABLED
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
            residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            dropout_probability=0.5,
            gate_config=self.gate_stack_config(dim),
        )
        layer = Layer(cfg)
        layer.eval()
        with torch.no_grad():
            layer.model.weight_params.copy_(torch.eye(dim))
            layer.model.bias_params.copy_(torch.tensor([1.0, -5.0]))
        gate_values = torch.tensor([[2.0, 3.0]])
        layer.gate_model.model = ConstantGate(gate_values)
        x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        state = LayerState(hidden=x)

        result = layer(state)

        model_output = x + torch.tensor([1.0, -5.0])
        activated = torch.relu(model_output)
        gate = torch.sigmoid(gate_values.expand_as(activated))
        expected = gate * activated + x
        torch.testing.assert_close(layer.gate_model.model.received_hidden, activated)
        torch.testing.assert_close(result.hidden, expected)

    def test_forward_training_dropout_zeros_current_before_residual(self):
        dim = 3
        cases = [
            (ResidualConnectionOptions.DISABLED, torch.zeros(2, dim)),
            (
                ResidualConnectionOptions.RESIDUAL,
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
            residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        )
        layer = Layer(cfg)
        with torch.no_grad():
            layer.model.weight_params.copy_(torch.eye(dim))
        x = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.0, 2.0]])

        result = layer(LayerState(hidden=x.clone()))

        expected = layer.layer_norm_module(x) + x
        torch.testing.assert_close(result.hidden, expected)

    def test_forward_layer_norm_after_normalizes_after_residual(self):
        dim = 3
        cfg = self.bare_config(
            input_dim=dim,
            output_dim=dim,
            residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        )
        layer = Layer(cfg)
        with torch.no_grad():
            layer.model.weight_params.copy_(2.0 * torch.eye(dim))
        x = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.0, 2.0]])

        result = layer(LayerState(hidden=x.clone()))

        expected = layer.layer_norm_module(2.0 * x + x)
        torch.testing.assert_close(result.hidden, expected)

    def test_maybe_apply_activation(self):
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
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    activation=activation,
                )
                layer = Layer(cfg)
                x = torch.randn(batch_size, input_dim)
                result = layer._Layer__maybe_apply_activation(x)

                self.assertEqual(result.shape, (batch_size, input_dim))
                if activation == ActivationOptions.DISABLED:
                    self.assertTrue(torch.equal(result, x))
                else:
                    expected = activation(x)
                    self.assertTrue(torch.equal(result, expected))

    def test_maybe_apply_residual_connection(self):
        batch_size = 4
        dim = 12
        option_cases = [
            (
                ResidualConnectionOptions.DISABLED,
                lambda current, previous: current,
            ),
            (
                ResidualConnectionOptions.RESIDUAL,
                lambda current, previous: current + previous,
            ),
            (
                ResidualConnectionOptions.WEIGHTED_RESIDUAL,
                lambda current, previous: previous,
            ),
            (
                ResidualConnectionOptions.WEIGHTED_BLEND,
                lambda current, previous: 0.9 * current + 0.1 * previous,
            ),
        ]

        for option, expected_fn in option_cases:
            message = f"residual_connection_option={option}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=dim,
                    output_dim=dim,
                    residual_connection_option=option,
                )
                layer = Layer(cfg)
                x = torch.randn(batch_size, dim)
                model_output = torch.randn(batch_size, dim)
                result = layer._Layer__maybe_apply_residual_connection(model_output, x)
                expected = expected_fn(model_output, x)

                torch.testing.assert_close(result, expected)

    def test_weighted_residual_uses_tanh_constrained_scalar(self):
        dim = 4
        cfg = self.preset(
            input_dim=dim,
            output_dim=dim,
            residual_connection_option=ResidualConnectionOptions.WEIGHTED_RESIDUAL,
        )
        layer = Layer(cfg)
        layer.residual_connection.raw_weight.data.fill_(0.5)
        current = torch.full((2, dim), 3.0)
        previous = torch.full((2, dim), 2.0)

        result = layer._Layer__maybe_apply_residual_connection(current, previous)

        expected = previous + torch.tanh(torch.tensor(0.5)) * current
        torch.testing.assert_close(result, expected)

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
            layer.gate_model.model[0].model.weight_params.copy_(0.5 * torch.eye(dim))
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
        gate_weight_grad = layer.gate_model.model[0].model.weight_params.grad
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
            layer.gate_model.model[0].model.weight_params.copy_(0.5 * torch.eye(dim))
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
            requires_grad=True,
        )

        result = layer(LayerState(hidden=x))
        result.hidden.sum().backward()

        gate_weight_grad = layer.gate_model.model[0].model.weight_params.grad
        self.assertIsNotNone(gate_weight_grad)
        self.assertTrue(torch.any(gate_weight_grad.abs() > 0))

    def test_forward_backward_reaches_weighted_residual_parameters(self):
        dim = 3
        residual_options = [
            ResidualConnectionOptions.WEIGHTED_RESIDUAL,
            ResidualConnectionOptions.WEIGHTED_BLEND,
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

                self.assertIsNotNone(layer.residual_connection.raw_weight.grad)
                self.assertTrue(
                    torch.any(layer.residual_connection.raw_weight.grad.abs() > 0)
                )

    def test_maybe_apply_dropout(self):
        batch_size = 64
        dim = 128
        dropout_probabilities = [0.0, 0.5]
        training_modes = [True, False]

        for dropout_probability in dropout_probabilities:
            for training in training_modes:
                message = (
                    f"dropout_probability={dropout_probability}, "
                    f"training={training}"
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
                    result = layer._Layer__maybe_apply_dropout(x)

                    self.assertEqual(result.shape, x.shape)
                    if dropout_probability > 0.0 and training:
                        has_zeros = (result == 0.0).any()
                        not_identical = not torch.equal(result, x)
                        self.assertTrue(has_zeros or not_identical)
                    else:
                        self.assertTrue(torch.equal(result, x))

    def test_maybe_apply_layer_norm(self):
        batch_size = 4
        dim = 12
        positions = [
            LayerNormPositionOptions.DISABLED,
            LayerNormPositionOptions.DEFAULT,
            LayerNormPositionOptions.BEFORE,
            LayerNormPositionOptions.AFTER,
        ]
        methods = [
            ("before", LayerNormPositionOptions.BEFORE),
            ("default", LayerNormPositionOptions.DEFAULT),
            ("after", LayerNormPositionOptions.AFTER),
        ]
        for position in positions:
            for method_name, active_position in methods:
                message = f"position={position}, method={method_name}"
                with self.subTest(msg=message):
                    cfg = self.preset(
                        input_dim=dim,
                        output_dim=dim,
                        layer_norm_position=position,
                    )
                    layer = Layer(cfg)
                    x = torch.randn(batch_size, dim)
                    method = getattr(
                        layer, f"_Layer__maybe_apply_layer_norm_{method_name}"
                    )
                    result = method(x)

                    self.assertEqual(result.shape, x.shape)
                    if position == active_position:
                        expected = layer.layer_norm_module(x)
                        self.assertTrue(torch.equal(result, expected))
                    else:
                        self.assertTrue(torch.equal(result, x))

    def test_maybe_apply_gates(self):
        batch_size = 4
        dim = 8
        cfg_with_gate = self.preset(input_dim=dim, output_dim=dim)
        cfg_without_gate = self.preset(input_dim=dim, output_dim=dim)

        layer_with_gate = Layer(cfg_with_gate)
        layer_without_gate = Layer(cfg_without_gate)
        layer_without_gate.gate_model = None

        layers = [
            (layer_with_gate, True),
            (layer_without_gate, False),
        ]
        for layer, has_gate in layers:
            message = f"has_gate={has_gate}"
            with self.subTest(msg=message):
                x = torch.randn(batch_size, dim)
                result = layer._Layer__maybe_apply_gates(x)

                self.assertEqual(result.shape, x.shape)
                if has_gate:
                    gate_output = Layer.run_model_returning_hidden(
                        layer.gate_model.model, x
                    )
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
        layer.gate_model.model = ConstantGate(gate_values)
        x = torch.tensor([[2.0, -3.0, 4.0], [-1.0, 5.0, 0.5]])

        result = layer._Layer__maybe_apply_gates(x)

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
        self.assertIsNone(layer.gate_model)
        x = torch.tensor([[2.0, -3.0, 4.0], [-1.0, 5.0, 0.5]])

        result = layer._Layer__maybe_apply_gates(x)

        torch.testing.assert_close(result, x)

    def test_layer_gate_rejects_missing_model_at_forward(self):
        dim = 3
        layer = Layer(
            self.bare_config(
                input_dim=dim,
                output_dim=dim,
                gate_config=self.gate_stack_config(dim),
            )
        )
        layer.gate_model.model = None

        with self.assertRaisesRegex(ValueError, "LayerGate requires a gate model"):
            layer._Layer__maybe_apply_gates(torch.ones(2, dim))

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
                layer.gate_model.model = gate_model

                with self.assertRaisesRegex(error_type, message):
                    layer._Layer__maybe_apply_gates(current)

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
                layer.gate_model.model = ConstantGate(gate_values)

                result = layer._Layer__maybe_apply_gates(current)

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
                layer.gate_model.model = ConstantGate(gate_values)

                result = layer._Layer__maybe_apply_gates(current)

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
            apply_output_pipeline_flag=False,
            layer_config=StateAwareLayerConfig(
                input_dim=dim,
                output_dim=dim,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
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
        layer.gate_model.model = spy
        x = torch.randn(batch_size, dim)

        result = layer._Layer__maybe_apply_gates(x)

        self.assertIs(type(spy.received_state), LayerState)
        self.assertIs(spy.received_hidden, x)
        self.assertIsNone(spy.received_state.loss)
        self.assertIsNone(spy.received_state.halting_state)
        torch.testing.assert_close(result, torch.sigmoid(torch.ones_like(x)) * x)

    def test_run_model_returning_hidden(self):
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
                result = Layer.run_model_returning_hidden(layer, x)

                self.assertEqual(result.shape, (batch_size, dim))

    def test_forward_state_helpers_construct_plain_state_and_reject_masks(self):
        class StateSpy(nn.Module):
            def __init__(self):
                super().__init__()
                self.received_state = None

            def forward(self, state: LayerState) -> LayerState:
                self.received_state = state
                return state

        spy = StateSpy()
        x = torch.randn(2, 3)

        result = Layer.run_model_returning_state(spy, x)

        self.assertIs(result, spy.received_state)
        self.assertIs(type(spy.received_state), LayerState)
        self.assertIs(spy.received_state.hidden, x)

        with self.assertRaises(TypeError):
            Layer.run_model_returning_state(
                spy,
                x,
                key_padding_mask=torch.ones(2, 3, dtype=torch.bool),
            )
        with self.assertRaises(TypeError):
            Layer.run_model_returning_hidden(
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
            ResidualConnectionOptions.RESIDUAL,
            ResidualConnectionOptions.DISABLED,
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
                                            else ResidualConnectionOptions.DISABLED
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
            halting_dropout=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=2,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
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

    def test_maybe_apply_halting_returns_values_without_mutating_state(self):
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
        original_hidden = state.hidden.clone()
        candidate = torch.randn(batch_size, dim)

        hidden, halting_state, loss = layer._Layer__maybe_apply_halting(
            candidate,
            state.halting_state,
            state.loss,
        )

        torch.testing.assert_close(state.hidden, original_hidden)
        self.assertIs(state.loss, existing_loss)
        self.assertIsNone(state.halting_state)
        torch.testing.assert_close(hidden, candidate)
        self.assertIsNotNone(halting_state)
        self.assertIs(loss, existing_loss)

    def test_halting_last_layer_finalizes_and_accumulates_vector_loss(self):
        dim = 3
        cfg = self.bare_config(
            input_dim=dim,
            output_dim=dim,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        )
        layer = Layer(cfg)
        layer.mark_as_last_layer()
        with torch.no_grad():
            layer.model.weight_params.copy_(torch.eye(dim))
        fake_halting_state = FakeHaltingState(
            halt_mask=torch.tensor([False, False]),
        )
        fake_halting = FakeHaltingModel(fake_halting_state)
        layer.halting_model = fake_halting
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

        self.assertFalse(layer.last_layer_flag)

        self.assertTrue(result.halting_state.halt_mask.all().item())
        self.assertIsNotNone(result.loss)
        self.assertTrue(
            torch.allclose(result.hidden, result.halting_state.accumulated_hidden)
        )

    def test_halted_state_skip_bypasses_layer_pipeline(self):
        dim = 3
        layer = Layer(self.bare_config(input_dim=dim, output_dim=dim))
        layer.model = AddConstantModel(100.0)
        fake_halting_state = FakeHaltingState(halt_mask=torch.ones(2, dtype=torch.bool))
        layer.halting_model = FakeHaltingModel(fake_halting_state)
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
        self.assertIsNone(layer.gate_model)
        self.assertEqual(layer.halting_model.update_calls, 0)

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
