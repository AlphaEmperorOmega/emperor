import torch
import unittest

from torch.nn import Sequential
from emperor.halting.config import HaltingConfig, StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.base.enums import LayerNormPositionOptions
from emperor.base.enums import ActivationOptions, LastLayerBiasOptions
from emperor.base.layer import (
    Layer,
    LayerConfig,
    LayerStack,
    LayerStackConfig,
    LayerState,
)
from emperor.linears.utils.config import LinearLayerConfig


class TestLayerStack(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 24,
        output_dim: int = 6,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = True,
        stack_dropout_probability: float = 0.2,
        shared_halting_flag: bool = False,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        gate_config: "LayerStackConfig | None" = None,
        halting_config: "StickBreakingConfig | None" = None,
    ) -> "LayerStackConfig":

        if gate_config is None:
            gate_config = LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=stack_activation,
                    layer_norm_position=layer_norm_position,
                    residual_flag=stack_residual_flag,
                    dropout_probability=stack_dropout_probability,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    model_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                        data_monitor=None,
                        parameter_monitor=None,
                    ),
                ),
            )

        if (
            halting_config is None
            and stack_num_layers > 1
            and input_dim == hidden_dim == output_dim
        ):
            halting_config = StickBreakingConfig(
                input_dim=output_dim,
                threshold=0.99,
                halting_dropout=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=LayerStackConfig(
                    input_dim=output_dim,
                    hidden_dim=output_dim,
                    output_dim=2,
                    num_layers=stack_num_layers,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_pipeline_flag=False,
                    layer_config=LayerConfig(
                        activation=ActivationOptions.DISABLED,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        residual_flag=stack_residual_flag,
                        dropout_probability=stack_dropout_probability,
                        halting_config=None,
                        shared_halting_flag=False,
                        gate_config=None,
                        model_config=LinearLayerConfig(
                            input_dim=output_dim,
                            output_dim=output_dim,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
            )

        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_flag=stack_residual_flag,
                dropout_probability=stack_dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                shared_halting_flag=shared_halting_flag,
                model_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=bias_flag,
                    data_monitor=None,
                    parameter_monitor=None,
                ),
            ),
        )

    def test_init_stores_all_config_attributes(self):
        cfg = self.preset()
        stack = LayerStack(cfg)

        self.assertIsInstance(stack, LayerStack)
        self.assertEqual(stack.input_dim, cfg.input_dim)
        self.assertEqual(stack.hidden_dim, cfg.hidden_dim)
        self.assertEqual(stack.output_dim, cfg.output_dim)
        self.assertEqual(stack.num_layers, cfg.num_layers)
        self.assertEqual(
            stack.apply_output_pipeline_flag, cfg.apply_output_pipeline_flag
        )
        self.assertEqual(stack.layer_config, cfg.layer_config)

        model = stack.build()
        layers = [model] if isinstance(model, Layer) else list(model)

        for i, layer in enumerate(layers):
            is_last_layer = i == len(layers) - 1
            with self.subTest(layer_index=i, is_last_layer=is_last_layer):
                self.assertIsInstance(layer, Layer)
                self.assertIsNotNone(layer.model)
                self.assertEqual(
                    layer.output_dim,
                    cfg.output_dim if is_last_layer else cfg.hidden_dim,
                )

                if is_last_layer and not cfg.apply_output_pipeline_flag:
                    self.assertEqual(
                        layer.activation_function, ActivationOptions.DISABLED
                    )
                    self.assertEqual(layer.dropout_probability, 0.0)
                    self.assertFalse(layer.residual_flag)
                else:
                    self.assertEqual(
                        layer.activation_function, cfg.layer_config.activation
                    )
                    self.assertEqual(
                        layer.dropout_probability, cfg.layer_config.dropout_probability
                    )

                if layer.gate_model is not None:
                    gate = layer.gate_model
                    gate_layers = [gate] if isinstance(gate, Layer) else list(gate)
                    for j, gate_layer in enumerate(gate_layers):
                        with self.subTest(gate_layer_index=j):
                            self.assertIsInstance(gate_layer, Layer)
                            self.assertIsNotNone(gate_layer.model)
                            self.assertIsNone(gate_layer.gate_config)
                            self.assertIsNone(gate_layer.halting_config)
                            self.assertFalse(gate_layer.shared_halting_flag)

    def test_build_returns_correct_type_for_num_layers(self):
        num_layers_options = [1, 2, 3, 4]
        for num_layers in num_layers_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(stack_num_layers=num_layers)
                model = LayerStack(cfg).build()

                if num_layers == 1:
                    self.assertIsInstance(model, Layer)
                    self.assertIsInstance(model.gate_model, Layer)
                else:
                    self.assertIsInstance(model, Sequential)
                    for layer in model:
                        self.assertIsInstance(layer.gate_model, Sequential)

    def test_layer_overrides_apply_correctly(self):
        cfg = self.preset(
            input_dim=8, output_dim=16, stack_dropout_probability=0.5
        ).layer_config
        overrides = LayerConfig(input_dim=12, output_dim=24)
        layer = Layer(cfg=cfg, overrides=overrides)

        self.assertEqual(layer.input_dim, 12)
        self.assertEqual(layer.output_dim, 24)
        self.assertEqual(layer.activation_function, ActivationOptions.RELU)
        self.assertEqual(layer.dropout_probability, 0.5)

    def test_stack_overrides_apply_correctly(self):
        cfg = self.preset(input_dim=8, hidden_dim=16, output_dim=4, stack_num_layers=3)
        overrides = LayerStackConfig(input_dim=12, hidden_dim=24, output_dim=6)
        stack = LayerStack(cfg, overrides)

        self.assertEqual(stack.input_dim, 12)
        self.assertEqual(stack.hidden_dim, 24)
        self.assertEqual(stack.output_dim, 6)
        self.assertEqual(stack.num_layers, 3)

    def test_last_layer_bias_option_applies_correctly(self):
        num_layers_options = [1, 2, 3]
        bias_options = [
            LastLayerBiasOptions.DEFAULT,
            LastLayerBiasOptions.DISABLED,
            LastLayerBiasOptions.ENABLED,
        ]
        bias_flags = [True, False]

        for num_layers in num_layers_options:
            for bias_option in bias_options:
                for bias_flag in bias_flags:
                    message = (
                        f"num_layers={num_layers}, "
                        f"bias_option={bias_option}, "
                        f"bias_flag={bias_flag}"
                    )
                    with self.subTest(msg=message):
                        cfg = self.preset(
                            stack_num_layers=num_layers,
                            bias_flag=bias_flag,
                            last_layer_bias_option=bias_option,
                        )
                        model = LayerStack(cfg).build()
                        layers = [model] if isinstance(model, Layer) else list(model)
                        last_layer = layers[-1]

                        match bias_option:
                            case LastLayerBiasOptions.DEFAULT:
                                if bias_flag:
                                    self.assertIsNotNone(last_layer.model.bias_params)
                                else:
                                    self.assertIsNone(last_layer.model.bias_params)
                            case LastLayerBiasOptions.DISABLED:
                                self.assertFalse(last_layer.model.bias_flag)
                            case LastLayerBiasOptions.ENABLED:
                                self.assertTrue(last_layer.model.bias_flag)

    def test_add_initial_layer(self):
        num_layers_options = [1, 2, 3, 4]
        input_dims = [8, 16]
        hidden_dim = 16
        output_dim = 6

        for num_layers in num_layers_options:
            for input_dim in input_dims:
                message = (
                    f"num_layers={num_layers}, "
                    f"input_dim={input_dim}, "
                    f"hidden_dim={hidden_dim}"
                )
                with self.subTest(msg=message):
                    cfg = self.preset(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        stack_num_layers=num_layers,
                    )
                    stack = LayerStack(cfg)
                    layers = []
                    adjustment = stack._LayerStack__add_initial_layer(layers)

                    should_add = input_dim != hidden_dim and num_layers > 1
                    if should_add:
                        self.assertEqual(len(layers), 1)
                        self.assertEqual(
                            adjustment, LayerStack.SEPARATE_INPUT_OUTPUT_DIM
                        )
                        self.assertEqual(layers[0].input_dim, input_dim)
                        self.assertEqual(layers[0].output_dim, hidden_dim)
                    else:
                        self.assertEqual(len(layers), 0)
                        self.assertEqual(adjustment, LayerStack.SHARED_INPUT_OUTPUT_DIM)

    def test_add_hidden_layers(self):
        num_layers_options = [1, 2, 3, 4]
        adjustments = [
            LayerStack.SHARED_INPUT_OUTPUT_DIM,
            LayerStack.SEPARATE_INPUT_OUTPUT_DIM,
        ]
        hidden_dim = 16

        for num_layers in num_layers_options:
            for adjustment in adjustments:
                message = f"num_layers={num_layers}, adjustment={adjustment}"
                with self.subTest(msg=message):
                    cfg = self.preset(
                        hidden_dim=hidden_dim,
                        stack_num_layers=num_layers,
                    )
                    stack = LayerStack(cfg)
                    layers = []
                    stack._LayerStack__add_hidden_layers(layers, adjustment)

                    expected_count = max(0, num_layers - adjustment)
                    self.assertEqual(len(layers), expected_count)

                    for layer in layers:
                        self.assertEqual(layer.input_dim, hidden_dim)
                        self.assertEqual(layer.output_dim, hidden_dim)

    def test_add_output_layer(self):
        num_layers_options = [1, 2, 3]
        output_dims = [6, 16]
        apply_pipeline_flags = [True, False]

        for num_layers in num_layers_options:
            for output_dim in output_dims:
                for apply_pipeline in apply_pipeline_flags:
                    message = (
                        f"num_layers={num_layers}, "
                        f"output_dim={output_dim}, "
                        f"apply_pipeline={apply_pipeline}"
                    )
                    with self.subTest(msg=message):
                        cfg = self.preset(
                            stack_num_layers=num_layers,
                            output_dim=output_dim,
                            apply_output_pipeline_flag=apply_pipeline,
                        )
                        stack = LayerStack(cfg)
                        layers = []
                        stack._LayerStack__add_output_layer(layers)

                        self.assertEqual(len(layers), 1)

                        layer = layers[0]
                        expected_input_dim = (
                            cfg.hidden_dim if num_layers > 1 else cfg.input_dim
                        )
                        self.assertIsInstance(layer, Layer)
                        self.assertEqual(layer.input_dim, expected_input_dim)
                        self.assertEqual(layer.output_dim, output_dim)
                        self.assertTrue(layer.last_layer_flag)

                        if apply_pipeline:
                            self.assertEqual(
                                layer.activation_function,
                                cfg.layer_config.activation,
                            )
                            self.assertEqual(
                                layer.dropout_probability,
                                cfg.layer_config.dropout_probability,
                            )
                        else:
                            self.assertEqual(
                                layer.activation_function, ActivationOptions.DISABLED
                            )
                            self.assertEqual(layer.dropout_probability, 0.0)
                            self.assertFalse(layer.residual_flag)

    def test_gate_config_rejects_nested_gates(self):
        gate_inner = LayerStackConfig(
            input_dim=6,
            hidden_dim=6,
            output_dim=6,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=6,
                output_dim=6,
                activation=ActivationOptions.DISABLED,
                residual_flag=False,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                shared_halting_flag=False,
                gate_config=None,
                model_config=LinearLayerConfig(bias_flag=True),
            ),
        )
        invalid_configs = [
            {"gate_config": gate_inner},
            {"halting_config": HaltingConfig()},
            {"shared_halting_flag": True},
        ]
        for invalid in invalid_configs:
            invalid_field = list(invalid.keys())[0]
            message = f"invalid_field={invalid_field}, value={invalid[invalid_field]}"
            with self.subTest(msg=message):
                gate_layer_config = LayerConfig(
                    input_dim=6,
                    output_dim=6,
                    activation=ActivationOptions.DISABLED,
                    residual_flag=False,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    model_config=LinearLayerConfig(bias_flag=True),
                    **invalid,
                )
                gate_config = LayerStackConfig(
                    input_dim=6,
                    hidden_dim=6,
                    output_dim=6,
                    num_layers=1,
                    last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                    apply_output_pipeline_flag=False,
                    layer_config=gate_layer_config,
                )
                cfg = self.preset(gate_config=gate_config)
                with self.assertRaises(ValueError):
                    LayerStack(cfg).build()

    def test_halting_rejects_single_layer_and_mismatched_dims(self):
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
                num_layers=2,
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
                    model_config=LinearLayerConfig(
                        input_dim=dim,
                        output_dim=dim,
                        bias_flag=True,
                        data_monitor=None,
                        parameter_monitor=None,
                    ),
                ),
            ),
        )

        invalid_cases = [
            {
                "stack_num_layers": 1,
                "input_dim": dim,
                "hidden_dim": dim,
                "output_dim": dim,
            },
            {
                "stack_num_layers": 2,
                "input_dim": dim,
                "hidden_dim": dim * 2,
                "output_dim": dim,
            },
            {
                "stack_num_layers": 2,
                "input_dim": dim,
                "hidden_dim": dim,
                "output_dim": dim * 2,
            },
        ]

        for case in invalid_cases:
            message = ", ".join(f"{k}={v}" for k, v in case.items())
            with self.subTest(msg=message):
                with self.assertRaises(ValueError):
                    LayerStack(self.preset(halting_config=halting_config, **case))

    def test_resolve_last_layer_bias_override(self):
        bias_options = [
            LastLayerBiasOptions.DEFAULT,
            LastLayerBiasOptions.DISABLED,
            LastLayerBiasOptions.ENABLED,
        ]
        bias_flags = [True, False]

        for bias_option in bias_options:
            for bias_flag in bias_flags:
                message = f"bias_option={bias_option}, bias_flag={bias_flag}"
                with self.subTest(msg=message):
                    cfg = self.preset(
                        bias_flag=bias_flag,
                        last_layer_bias_option=bias_option,
                    )
                    stack = LayerStack(cfg)
                    result = stack._LayerStack__resolve_last_layer_bias_override()

                    if bias_option == LastLayerBiasOptions.DEFAULT:
                        self.assertIsNone(result)
                    else:
                        self.assertIsInstance(result, LayerConfig)
                        self.assertIsNotNone(result.model_config)
                        if bias_option == LastLayerBiasOptions.DISABLED:
                            self.assertFalse(result.model_config.bias_flag)
                        elif bias_option == LastLayerBiasOptions.ENABLED:
                            self.assertTrue(result.model_config.bias_flag)

    def test_create_layer(self):
        dim_pairs = [(8, 8), (8, 16), (16, 8)]

        for input_dim, output_dim in dim_pairs:
            message = f"input_dim={input_dim}, output_dim={output_dim}"
            with self.subTest(msg=message):
                cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
                stack = LayerStack(cfg)
                layer = stack._LayerStack__create_layer(input_dim, output_dim)

                self.assertIsInstance(layer, Layer)
                self.assertEqual(layer.input_dim, input_dim)
                self.assertEqual(layer.output_dim, output_dim)

                if input_dim != output_dim:
                    self.assertFalse(layer.residual_flag)
                else:
                    self.assertEqual(
                        layer.residual_flag, cfg.layer_config.residual_flag
                    )

    def test_create_layer_with_overrides(self):
        cfg = self.preset(input_dim=8, output_dim=8)
        stack = LayerStack(cfg)
        overrides = LayerConfig(
            activation=ActivationOptions.DISABLED,
            dropout_probability=0.0,
        )
        layer = stack._LayerStack__create_layer(8, 16, overrides)

        self.assertIsInstance(layer, Layer)
        self.assertEqual(layer.input_dim, 8)
        self.assertEqual(layer.output_dim, 16)
        self.assertFalse(layer.residual_flag)
        self.assertEqual(layer.activation_function, ActivationOptions.DISABLED)
        self.assertEqual(layer.dropout_probability, 0.0)

    def test_build_forward_pass_output_shape(self):
        batch_size = 4
        num_layers_options = [1, 2, 3]
        input_dims = [6, 12]
        hidden_dims = [6, 24]
        output_dims = [6, 8]
        activations = [ActivationOptions.RELU, ActivationOptions.DISABLED]
        residual_flags = [True, False]
        dropout_probabilities = [0.0, 0.2]
        layer_norm_positions = [
            LayerNormPositionOptions.DISABLED,
            LayerNormPositionOptions.DEFAULT,
            LayerNormPositionOptions.BEFORE,
            LayerNormPositionOptions.AFTER,
        ]

        for num_layers in num_layers_options:
            for input_dim in input_dims:
                for hidden_dim in hidden_dims:
                    for output_dim in output_dims:
                        for activation in activations:
                            for residual_flag in residual_flags:
                                for dropout in dropout_probabilities:
                                    for layer_norm in layer_norm_positions:
                                        message = (
                                            f"num_layers={num_layers}, "
                                            f"input_dim={input_dim}, "
                                            f"output_dim={output_dim}, "
                                            f"activation={activation}, "
                                            f"residual_flag={residual_flag}, "
                                            f"dropout={dropout}, "
                                            f"layer_norm={layer_norm}"
                                        )
                                        with self.subTest(msg=message):
                                            cfg = self.preset(
                                                stack_num_layers=num_layers,
                                                input_dim=input_dim,
                                                hidden_dim=hidden_dim,
                                                output_dim=output_dim,
                                                stack_activation=activation,
                                                stack_residual_flag=residual_flag,
                                                stack_dropout_probability=dropout,
                                                layer_norm_position=layer_norm,
                                            )
                                            model = LayerStack(cfg).build()
                                            x = torch.randn(batch_size, input_dim)
                                            state = LayerState(hidden=x)
                                            output_state = model(state)
                                            expected_shape = (batch_size, output_dim)

                                            self.assertEqual(
                                                output_state.hidden.shape,
                                                expected_shape,
                                            )

                                            layers = (
                                                [model]
                                                if isinstance(model, Layer)
                                                else list(model)
                                            )
                                            for layer in layers:
                                                if layer.input_dim != layer.output_dim:
                                                    self.assertFalse(
                                                        layer.residual_flag
                                                    )
