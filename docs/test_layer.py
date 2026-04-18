import torch
import unittest
import torch.nn as nn

from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.base.layer import Layer, LayerConfig, LayerStackConfig, LayerState
from emperor.linears.core.config import LinearLayerConfig
from emperor.base.enums import LastLayerBiasOptions
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions


class TestLayer(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 8,
        bias_flag: bool = True,
        activation: ActivationOptions = ActivationOptions.RELU,
        residual_flag: bool = False,
        dropout_probability: float = 0.2,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        gate_config: "LayerStackConfig | None" = None,
        gate_num_layers: int = 1,
        gate_activation: ActivationOptions = ActivationOptions.DISABLED,
        gate_residual_flag: bool = False,
        gate_dropout_probability: float = 0.0,
        gate_bias_flag: bool = True,
        halting_config: "StickBreakingConfig | None" = None,
        shared_halting_flag: bool = False,
    ) -> LayerConfig:

        if gate_config is None:
            gate_config = LayerStackConfig(
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim,
                num_layers=gate_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=gate_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=gate_residual_flag,
                    dropout_probability=gate_dropout_probability,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=output_dim,
                        output_dim=output_dim,
                        bias_flag=gate_bias_flag,
                    ),
                ),
            )

        if halting_config is None and input_dim == output_dim:
            halting_config = StickBreakingConfig(
                input_dim=output_dim,
                threshold=0.99,
                halting_dropout=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=LayerStackConfig(
                    input_dim=output_dim,
                    hidden_dim=output_dim,
                    output_dim=2,
                    num_layers=gate_num_layers,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_pipeline_flag=False,
                    layer_config=LayerConfig(
                        activation=gate_activation,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        residual_flag=gate_residual_flag,
                        dropout_probability=gate_dropout_probability,
                        halting_config=None,
                        shared_halting_flag=False,
                        gate_config=None,
                        layer_model_config=LinearLayerConfig(
                            input_dim=output_dim,
                            output_dim=output_dim,
                            bias_flag=True,
                        ),
                    ),
                ),
            )

        return LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            residual_flag=residual_flag,
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=gate_config,
            halting_config=halting_config,
            shared_halting_flag=shared_halting_flag,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
            ),
        )

    def test_init_stores_all_config_attributes(self):
        cfg = self.preset()
        layer = Layer(cfg)

        self.assertIsInstance(layer, Layer)
        self.assertEqual(layer.input_dim, cfg.input_dim)
        self.assertEqual(layer.output_dim, cfg.output_dim)
        self.assertEqual(layer.activation_function, cfg.activation)
        self.assertEqual(layer.residual_flag, cfg.residual_flag)
        self.assertEqual(layer.dropout_probability, cfg.dropout_probability)
        self.assertEqual(layer.layer_norm_position, cfg.layer_norm_position)
        self.assertEqual(layer.gate_config, cfg.gate_config)
        self.assertEqual(layer.halting_config, cfg.halting_config)
        self.assertEqual(layer.shared_halting_flag, cfg.shared_halting_flag or False)
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

    def test_mark_as_last_layer(self):
        layer = Layer(self.preset())

        self.assertFalse(layer.last_layer_flag)
        layer.mark_as_last_layer()
        self.assertTrue(layer.last_layer_flag)

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

    def test_build_gate_model(self):
        cfg = self.preset()
        layer = Layer(cfg)
        gate_configs = [cfg.gate_config, None]

        for gate_config in gate_configs:
            message = f"has_config={gate_config is not None}"
            with self.subTest(msg=message):
                layer.gate_config = gate_config
                result = layer._Layer__build_gate_model()

                if gate_config is not None:
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result, Layer)
                else:
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
                    self.assertTrue(layer.has_dropout)
                else:
                    self.assertIsNone(result)
                    self.assertFalse(layer.has_dropout)

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

    def test_has_activation_flag(self):
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
                self.assertEqual(layer.has_activation, expected)

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
                        residual_flag=input_dim == output_dim,
                    )
                    layer = Layer(cfg)
                    x = torch.randn(batch_size, input_dim)
                    state = LayerState(hidden=x)
                    output = layer(state)

                    self.assertEqual(output.hidden.shape, (batch_size, output_dim))

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
        residual_flags = [True, False]

        for residual_flag in residual_flags:
            message = f"residual_flag={residual_flag}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=dim,
                    output_dim=dim,
                    residual_flag=residual_flag,
                )
                layer = Layer(cfg)
                x = torch.randn(batch_size, dim)
                model_output = torch.randn(batch_size, dim)
                result = layer._Layer__maybe_apply_residual_connection(model_output, x)

                if residual_flag:
                    expected = model_output + x
                    self.assertTrue(torch.equal(result, expected))
                else:
                    self.assertTrue(torch.equal(result, model_output))

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
                    gate_output = Layer.forward_with_state(layer.gate_model, x)
                    expected = gate_output * x
                    self.assertTrue(torch.equal(result, expected))
                else:
                    self.assertTrue(torch.equal(result, x))

    def test_forward_with_state(self):
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
                result = Layer.forward_with_state(layer, x)

                self.assertEqual(result.shape, (batch_size, dim))

    def test_forward_output_shape_all_config_combinations(self):
        batch_size = 4
        input_dims = [8, 12]
        output_dims = [8, 16]
        activations = [ActivationOptions.RELU, ActivationOptions.DISABLED]
        residual_flags = [True, False]
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
                    for residual_flag in residual_flags:
                        for dropout in dropout_probabilities:
                            for layer_norm in layer_norm_positions:
                                message = (
                                    f"input_dim={input_dim}, "
                                    f"output_dim={output_dim}, "
                                    f"activation={activation}, "
                                    f"residual_flag={residual_flag}, "
                                    f"dropout={dropout}, "
                                    f"layer_norm={layer_norm}"
                                )
                                with self.subTest(msg=message):
                                    cfg = self.preset(
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        activation=activation,
                                        residual_flag=(
                                            residual_flag
                                            if input_dim == output_dim
                                            else False
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
                    residual_flag=False,
                    dropout_probability=0.0,
                    halting_config=None,
                    shared_halting_flag=False,
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
