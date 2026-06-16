from emperor.base.layer.residual import ResidualConnectionOptions
import torch
import unittest

from emperor.halting.config import HaltingConfig, StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.base.options import LayerNormPositionOptions
from emperor.base.options import ActivationOptions, LastLayerBiasOptions
from emperor.base.layer import (
    Layer,
    LayerConfig,
    LayerStack,
    LayerStackConfig,
    LayerState,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.linears.core.config import LinearLayerConfig


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
        stack_residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.DISABLED
        ),
        stack_dropout_probability: float = 0.2,
        shared_gate_config: "LayerStackConfig | GateConfig | None" = None,
        shared_halting_config: "StickBreakingConfig | None" = None,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        gate_enabled: bool = True,
        gate_config: "LayerStackConfig | GateConfig | None" = None,
        gate_option: LayerGateOptions | None = None,
        halting_config: "StickBreakingConfig | None" = None,
    ) -> "LayerStackConfig":

        if gate_enabled and gate_config is None and shared_gate_config is None:
            gate_config = LayerStackConfig(
                hidden_dim=hidden_dim,
                num_layers=stack_num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=stack_activation,
                    layer_norm_position=layer_norm_position,
                    residual_connection_option=stack_residual_connection_option,
                    dropout_probability=stack_dropout_probability,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=bias_flag,
                    ),
                ),
            )

        if (
            halting_config is None
            and shared_halting_config is None
            and stack_num_layers > 1
            and input_dim == hidden_dim == output_dim
        ):
            halting_config = StickBreakingConfig(
                threshold=0.99,
                halting_dropout=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=LayerStackConfig(
                    hidden_dim=output_dim,
                    output_dim=2,
                    num_layers=stack_num_layers,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_pipeline_flag=False,
                    layer_config=LayerConfig(
                        activation=ActivationOptions.DISABLED,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        residual_connection_option=stack_residual_connection_option,
                        dropout_probability=stack_dropout_probability,
                        halting_config=None,
                        gate_config=None,
                        layer_model_config=LinearLayerConfig(
                            bias_flag=True,
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
            shared_gate_config=self.layer_gate_config(shared_gate_config, gate_option),
            shared_halting_config=shared_halting_config,
            layer_config=LayerConfig(
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_connection_option=stack_residual_connection_option,
                dropout_probability=stack_dropout_probability,
                gate_config=self.layer_gate_config(gate_config, gate_option),
                halting_config=halting_config,
                layer_model_config=LinearLayerConfig(
                    bias_flag=bias_flag,
                ),
            ),
        )

    def layer_gate_config(
        self,
        model_config: "LayerStackConfig | GateConfig | None",
        option: LayerGateOptions | None,
    ) -> GateConfig | None:
        if isinstance(model_config, GateConfig):
            return model_config
        if model_config is None:
            return None
        if option is None:
            option = LayerGateOptions.MULTIPLIER
        return GateConfig(
            model_config=model_config,
            option=option,
            activation=ActivationOptions.SIGMOID,
        )

    def gate_stack_config(
        self,
        dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_layers: int = 1,
        apply_output_pipeline_flag: bool = False,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=dim,
            hidden_dim=hidden_dim if hidden_dim is not None else dim,
            output_dim=output_dim if output_dim is not None else dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                input_dim=dim,
                output_dim=output_dim if output_dim is not None else dim,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=dim,
                    output_dim=output_dim if output_dim is not None else dim,
                    bias_flag=True,
                ),
            ),
        )

    def halting_config(
        self,
        dim: int,
        threshold: float,
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
        self.assertEqual(stack.shared_gate_config, cfg.shared_gate_config)
        self.assertEqual(stack.shared_halting_config, cfg.shared_halting_config)
        self.assertEqual(stack.shared_memory_config, cfg.shared_memory_config)
        self.assertEqual(stack.layer_config, cfg.layer_config)

        model = stack
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
                    self.assertEqual(
                        layer.residual_connection_option,
                        ResidualConnectionOptions.DISABLED,
                    )
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

    def test_build_returns_correct_type_for_num_layers(self):
        num_layers_options = [1, 2, 3, 4]
        for num_layers in num_layers_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(
                    input_dim=8,
                    hidden_dim=8,
                    output_dim=8,
                    stack_num_layers=num_layers,
                )
                model = LayerStack(cfg)

                self.assertIsInstance(model, LayerStack)
                for layer in model:
                    self.assertIsInstance(layer.gate_model.model, LayerStack)

    def test_layer_overrides_apply_correctly(self):
        cfg = self.preset(
            input_dim=8,
            output_dim=16,
            stack_dropout_probability=0.5,
            gate_enabled=False,
        ).layer_config
        overrides = LayerConfig(
            input_dim=12,
            output_dim=24,
        )
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

    def test_validation_errors_for_base_stack_config_contract(self):
        required_fields = [
            "input_dim",
            "hidden_dim",
            "output_dim",
            "num_layers",
            "apply_output_pipeline_flag",
            "last_layer_bias_option",
            "layer_config",
        ]

        for field_name in required_fields:
            with self.subTest(field_name=field_name):
                cfg = self.preset(gate_enabled=False)
                setattr(cfg, field_name, None)

                with self.assertRaisesRegex(ValueError, field_name):
                    LayerStack(cfg)

        wrong_type_cases = [
            ("input_dim", "8", TypeError),
            ("hidden_dim", "8", TypeError),
            ("output_dim", "8", TypeError),
            ("num_layers", "2", TypeError),
            ("apply_output_pipeline_flag", "yes", TypeError),
            ("layer_config", object(), TypeError),
        ]
        for field_name, value, error_type in wrong_type_cases:
            with self.subTest(field_name=field_name):
                cfg = self.preset(gate_enabled=False)
                setattr(cfg, field_name, value)

                with self.assertRaisesRegex(error_type, field_name):
                    LayerStack(cfg)

        with self.assertRaisesRegex(ValueError, "num_layers"):
            LayerStack(self.preset(stack_num_layers=0, gate_enabled=False))

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
                        model = LayerStack(cfg)
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
                            self.assertEqual(
                                layer.residual_connection_option,
                                ResidualConnectionOptions.DISABLED,
                            )

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
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=True),
            ),
        )
        invalid_cases = [
            (
                "gate_config",
                {
                    "gate_config": GateConfig(
                        model_config=gate_inner,
                        option=LayerGateOptions.MULTIPLIER,
                    )
                },
                {},
            ),
            (
                "shared_gate_config",
                {},
                {
                    "shared_gate_config": GateConfig(
                        model_config=gate_inner,
                        option=LayerGateOptions.MULTIPLIER,
                    )
                },
            ),
            ("halting_config", {"halting_config": HaltingConfig()}, {}),
            (
                "shared_halting_config",
                {},
                {"shared_halting_config": self.halting_config(6, threshold=0.99)},
            ),
        ]
        for invalid_field, layer_invalid, stack_invalid in invalid_cases:
            message = f"invalid_field={invalid_field}"
            with self.subTest(msg=message):
                gate_layer_config = LayerConfig(
                    input_dim=6,
                    output_dim=6,
                    activation=ActivationOptions.DISABLED,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    layer_model_config=LinearLayerConfig(bias_flag=True),
                    **layer_invalid,
                )
                gate_config = LayerStackConfig(
                    input_dim=6,
                    hidden_dim=6,
                    output_dim=6,
                    num_layers=1,
                    last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                    apply_output_pipeline_flag=False,
                    layer_config=gate_layer_config,
                    **stack_invalid,
                )
                cfg = self.preset(gate_config=gate_config)
                with self.assertRaises(ValueError):
                    LayerStack(cfg)

    def test_shared_gate_reuses_one_module_across_stack(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_gate_config=self.gate_stack_config(dim),
        )
        model = LayerStack(cfg)
        layers = list(model)
        gated_layers = [layer for layer in layers if layer.gate_model is not None]
        gate_models = [layer.gate_model.model for layer in gated_layers]

        self.assertEqual(len(gated_layers), len(layers))
        self.assertTrue(all(gate_model is not None for gate_model in gate_models))
        shared_gate = gated_layers[0].gate_model
        self.assertEqual(shared_gate.gate_dim, dim)
        self.assertTrue(all(layer.gate_model is shared_gate for layer in layers))
        shared_gate_model = gate_models[0]
        self.assertTrue(
            all(gate_model is shared_gate_model for gate_model in gate_models)
        )
        for layer in gated_layers:
            self.assertIsNone(layer.cfg.gate_config)
            self.assertIs(layer.gate_config, cfg.shared_gate_config)

    def test_unshared_gate_builds_separate_modules_per_layer(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
        )
        model = LayerStack(cfg)
        layers = list(model)
        gated_layers = [layer for layer in layers if layer.gate_model is not None]
        gate_models = [layer.gate_model.model for layer in gated_layers]

        self.assertLess(len(gated_layers), len(layers))
        self.assertTrue(all(gate_model is not None for gate_model in gate_models))
        self.assertEqual(
            len({id(gate_model) for gate_model in gate_models}),
            len(gated_layers),
        )

    def test_stack_created_layers_inherit_gate_option(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            gate_option=LayerGateOptions.MULTIPLIER,
        )

        model = LayerStack(cfg)

        for index, layer in enumerate(model):
            with self.subTest(layer_index=index):
                if layer.input_dim != layer.output_dim:
                    self.assertIsNone(layer.gate_model)
                else:
                    self.assertIsNotNone(layer.gate_model)
                    self.assertEqual(
                        layer.gate_model.option, LayerGateOptions.MULTIPLIER
                    )

    def test_shared_gate_model_works_with_non_default_gate_option(self):
        dim = 4
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=2,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_gate_config=self.gate_stack_config(dim),
            gate_option=LayerGateOptions.MULTIPLIER,
        )
        model = LayerStack(cfg)
        for layer in model:
            with torch.no_grad():
                layer.model.weight_params.zero_()
                layer.model.weight_params[:dim, :dim].copy_(torch.eye(dim))
                layer.model.bias_params.zero_()
        gated_layers = [layer for layer in model if layer.gate_model is not None]
        shared_gate_model = gated_layers[0].gate_model.model
        with torch.no_grad():
            shared_gate_model[0].model.weight_params.zero_()
            shared_gate_model[0].model.bias_params.zero_()
        x = torch.ones(2, dim + 1)

        result = model(LayerState(hidden=x.clone()))

        self.assertTrue(
            all(layer.gate_model.model is shared_gate_model for layer in gated_layers)
        )
        self.assertEqual(result.hidden.shape, (2, dim))

    def test_shared_gate_rejects_invalid_config_type(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_gate_config=object(),
        )

        with self.assertRaises(TypeError):
            LayerStack(cfg)

    def test_shared_gate_rejects_per_layer_gate_config(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_gate_config=self.gate_stack_config(dim),
            gate_config=self.gate_stack_config(dim),
        )

        with self.assertRaises(ValueError):
            LayerStack(cfg)

    def test_shared_gate_rejects_hidden_output_mismatch(self):
        cfg = self.preset(
            input_dim=5,
            hidden_dim=8,
            output_dim=6,
            stack_num_layers=3,
            shared_gate_config=self.gate_stack_config(6),
        )

        with self.assertRaisesRegex(ValueError, "hidden_dim and output_dim"):
            LayerStack(cfg)

    def test_shared_gate_rejects_single_layer_hidden_output_mismatch(self):
        cfg = self.preset(
            input_dim=5,
            hidden_dim=8,
            output_dim=6,
            stack_num_layers=1,
            shared_gate_config=self.gate_stack_config(6),
        )

        with self.assertRaisesRegex(ValueError, "hidden_dim and output_dim"):
            LayerStack(cfg)

    def test_shared_gate_stack_forward_pass(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_gate_config=self.gate_stack_config(dim),
        )
        model = LayerStack(cfg)
        model.eval()

        state = model(LayerState(hidden=torch.randn(batch_size, dim + 1)))
        gated_layers = [layer for layer in model if layer.gate_model is not None]
        shared_gate_model = gated_layers[0].gate_model.model

        self.assertEqual(state.hidden.shape, (batch_size, dim))
        self.assertTrue(
            all(layer.gate_model.model is shared_gate_model for layer in gated_layers)
        )

    def test_shared_gate_receives_shared_gradients(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_gate_config=self.gate_stack_config(dim),
        )
        model = LayerStack(cfg)
        layers = list(model)
        gated_layers = [layer for layer in layers if layer.gate_model is not None]
        shared_gate_model = gated_layers[0].gate_model.model
        before = [
            parameter.detach().clone()
            for parameter in shared_gate_model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        state = model(LayerState(hidden=torch.randn(batch_size, dim + 1)))
        optimizer.zero_grad()
        state.hidden.sum().backward()
        optimizer.step()

        self.assertTrue(
            all(layer.gate_model.model is shared_gate_model for layer in gated_layers)
        )
        after = [
            parameter.detach()
            for parameter in shared_gate_model.parameters()
            if parameter.requires_grad
        ]
        changed = any(
            not torch.equal(before_parameter, after_parameter)
            for before_parameter, after_parameter in zip(before, after)
        )
        self.assertTrue(changed)

    def test_unshared_stack_layers_receive_gradients(self):
        batch_size = 4
        cfg = self.preset(
            input_dim=4,
            hidden_dim=5,
            output_dim=6,
            stack_num_layers=3,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            gate_enabled=False,
            halting_config=None,
            shared_halting_config=None,
        )
        model = LayerStack(cfg)
        hidden = torch.tensor(
            [
                [1.0, 0.5, -1.0, 2.0],
                [0.0, 1.0, 2.0, -0.5],
                [2.0, -1.0, 0.5, 1.0],
                [1.5, 0.0, -0.5, 0.5],
            ],
            requires_grad=True,
        )

        state = model(LayerState(hidden=hidden))
        state.hidden.sum().backward()

        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.any(hidden.grad.abs() > 0))
        for index, layer in enumerate(model):
            with self.subTest(layer_index=index):
                gradients = [
                    parameter.grad
                    for parameter in layer.model.parameters()
                    if parameter.requires_grad
                ]
                nonzero_gradients = [
                    gradient
                    for gradient in gradients
                    if gradient is not None and torch.any(gradient.abs() > 0)
                ]
                self.assertTrue(len(nonzero_gradients) > 0)

    def test_output_pipeline_flag_false_does_not_disable_gates(self):
        dim = 8
        per_layer_cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=3,
            apply_output_pipeline_flag=False,
        )
        shared_cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=3,
            apply_output_pipeline_flag=False,
            shared_gate_config=self.gate_stack_config(dim),
        )
        no_gate_cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=3,
            apply_output_pipeline_flag=False,
            gate_enabled=False,
        )

        per_layer_model = LayerStack(per_layer_cfg)
        shared_model = LayerStack(shared_cfg)
        no_gate_model = LayerStack(no_gate_cfg)

        self.assertIsNotNone(per_layer_model[-1].gate_model)
        self.assertIsNotNone(shared_model[-1].gate_model)
        self.assertIsNotNone(shared_model[0].gate_model)
        self.assertIs(shared_model[0].gate_model, shared_model[1].gate_model)
        self.assertIs(shared_model[-1].gate_model, shared_model[1].gate_model)
        self.assertIsNone(no_gate_model[-1].gate_model)

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

    def test_shared_halting_rejects_per_layer_halting_config(self):
        dim = 8
        shared_halting_config = self.halting_config(dim, threshold=0.99)
        per_layer_halting_config = self.halting_config(dim, threshold=0.99)
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_halting_config=shared_halting_config,
            halting_config=per_layer_halting_config,
        )

        with self.assertRaises(ValueError):
            LayerStack(cfg)

    def test_shared_halting_rejects_invalid_config_type(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_halting_config=object(),
        )

        with self.assertRaises(TypeError):
            LayerStack(cfg)

    def test_shared_halting_rejects_single_layer_and_mismatched_dims(self):
        dim = 8
        shared_halting_config = self.halting_config(dim, threshold=0.99)
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
                cfg = self.preset(
                    shared_halting_config=shared_halting_config,
                    **case,
                )
                with self.assertRaises(ValueError):
                    LayerStack(cfg)

    def test_shared_halting_reuses_one_module_across_stack(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_halting_config=self.halting_config(dim, threshold=0.99),
        )
        model = LayerStack(cfg)
        layers = [model] if isinstance(model, Layer) else list(model)
        halting_models = [layer.halting_model for layer in layers]

        self.assertTrue(all(halting_model is not None for halting_model in halting_models))
        first_halting_model = halting_models[0]
        self.assertTrue(
            all(halting_model is first_halting_model for halting_model in halting_models)
        )
        for layer in layers:
            self.assertIsNone(layer.cfg.halting_config)
            self.assertIsNone(layer.halting_config)

    def test_unshared_halting_builds_separate_modules_per_layer(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            halting_config=self.halting_config(dim, threshold=0.99),
        )
        model = LayerStack(cfg)
        layers = [model] if isinstance(model, Layer) else list(model)
        halting_models = [layer.halting_model for layer in layers]

        self.assertTrue(all(halting_model is not None for halting_model in halting_models))
        self.assertEqual(len({id(halting_model) for halting_model in halting_models}), len(layers))

    def test_shared_halting_stack_forward_pass(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_halting_config=self.halting_config(dim, threshold=1.0),
        )
        model = LayerStack(cfg)
        model.eval()

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)

        self.assertEqual(state.hidden.shape, (batch_size, dim))
        self.assertIsNotNone(state.halting_state)
        self.assertIsNotNone(state.loss)

    def test_shared_halting_remains_shared_after_training_step(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_halting_config=self.halting_config(dim, threshold=1.0),
        )
        model = LayerStack(cfg)
        model.eval()
        layers = [model] if isinstance(model, Layer) else list(model)
        shared_halting_model = layers[0].halting_model
        before = [
            parameter.detach().clone()
            for parameter in shared_halting_model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)
        loss = state.hidden.sum() + state.loss.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertTrue(
            all(layer.halting_model is shared_halting_model for layer in layers)
        )
        after = [
            parameter.detach()
            for parameter in shared_halting_model.parameters()
            if parameter.requires_grad
        ]
        changed = any(
            not torch.equal(before_parameter, after_parameter)
            for before_parameter, after_parameter in zip(before, after)
        )
        self.assertTrue(changed)

    def test_halting_stack_finalizes_early_and_skips_remaining_layers(self):
        batch_size = 4
        dim = 8
        num_layers = 3
        halting_config = self.halting_config(dim, threshold=1e-9)
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=num_layers,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            halting_config=halting_config,
        )
        model = LayerStack(cfg)
        model.eval()

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)

        self.assertTrue(state.halting_state.halt_mask.all().item())
        self.assertEqual(state.halting_state.step_count, 0)
        self.assertIsNotNone(state.loss)

    def test_halting_stack_finalizes_at_last_layer_when_not_all_halted(self):
        batch_size = 4
        dim = 8
        num_layers = 10
        halting_config = self.halting_config(dim, threshold=1.0)
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=num_layers,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            halting_config=halting_config,
        )
        model = LayerStack(cfg)
        model.eval()

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)

        self.assertFalse(state.halting_state.halt_mask.all().item())
        self.assertEqual(state.halting_state.step_count, num_layers - 1)
        self.assertIsNotNone(state.loss)

    def test_halting_stack_with_ten_layers_finalizes_between_first_and_last_layer(self):
        batch_size = 4
        dim = 8
        num_layers = 10
        halting_config = self.halting_config(dim, threshold=0.7)
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=num_layers,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            halting_config=halting_config,
        )
        model = LayerStack(cfg)
        model.eval()

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)

        self.assertTrue(state.halting_state.halt_mask.all().item())
        self.assertGreater(state.halting_state.step_count, 0)
        self.assertLess(state.halting_state.step_count, num_layers - 1)
        self.assertIsNotNone(state.loss)

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
                        self.assertIsNotNone(result.layer_model_config)
                        if bias_option == LastLayerBiasOptions.DISABLED:
                            self.assertFalse(result.layer_model_config.bias_flag)
                        elif bias_option == LastLayerBiasOptions.ENABLED:
                            self.assertTrue(result.layer_model_config.bias_flag)

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
                    self.assertEqual(
                        layer.residual_connection_option,
                        ResidualConnectionOptions.DISABLED,
                    )
                else:
                    self.assertEqual(
                        layer.residual_connection_option,
                        cfg.layer_config.residual_connection_option,
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
        self.assertEqual(
            layer.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(layer.activation_function, ActivationOptions.DISABLED)
        self.assertEqual(layer.dropout_probability, 0.0)

    def test_build_forward_pass_output_shape(self):
        batch_size = 4
        num_layers_options = [1, 2, 3]
        input_dims = [6, 12]
        hidden_dims = [6, 24]
        output_dims = [6, 8]
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

        for num_layers in num_layers_options:
            for input_dim in input_dims:
                for hidden_dim in hidden_dims:
                    for output_dim in output_dims:
                        for activation in activations:
                            for residual_option in residual_options:
                                for dropout in dropout_probabilities:
                                    for layer_norm in layer_norm_positions:
                                        message = (
                                            f"num_layers={num_layers}, "
                                            f"input_dim={input_dim}, "
                                            f"output_dim={output_dim}, "
                                            f"activation={activation}, "
                                            f"residual_connection_option={residual_option}, "
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
                                                stack_residual_connection_option=residual_option,
                                                stack_dropout_probability=dropout,
                                                layer_norm_position=layer_norm,
                                            )
                                            model = LayerStack(cfg)
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
                                                    self.assertEqual(
                                                        layer.residual_connection_option,
                                                        ResidualConnectionOptions.DISABLED,
                                                    )
