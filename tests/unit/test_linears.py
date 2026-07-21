import json
import subprocess
import sys
import unittest

import torch

import emperor.linears as linears
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    DualModelDynamicWeightConfig,
    DynamicDepthOptions,
    GeneratorDynamicBiasConfig,
    MaskDimensionOptions,
    StandardDynamicDiagonalConfig,
    WeightDecayScheduleOptions,
    WeightInformedScoreAxisMaskConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.halting import HaltingHiddenStateModeOptions, StickBreakingConfig
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayer, LinearLayerConfig, LinearOptions


def make_layer_stack_config(
    input_dim: int = 16,
    output_dim: int | None = None,
    hidden_dim: int | None = None,
    num_layers: int = 2,
) -> LayerStackConfig:
    output_dim = input_dim if output_dim is None else output_dim
    hidden_dim = max(input_dim, output_dim) if hidden_dim is None else hidden_dim
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=True,
            ),
        ),
    )


def make_weight_config(input_dim: int, output_dim: int) -> DualModelDynamicWeightConfig:
    return DualModelDynamicWeightConfig(
        generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
        normalization_option=WeightNormalizationOptions.L2_SCALE,
        normalization_position_option=WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
        decay_schedule=WeightDecayScheduleOptions.DISABLED,
        decay_rate=0.0,
        decay_warmup_batches=0,
        model_config=make_layer_stack_config(input_dim, output_dim),
    )


def make_bias_config(
    input_dim: int, output_dim: int, bias_flag: bool = True
) -> GeneratorDynamicBiasConfig:
    return GeneratorDynamicBiasConfig(
        decay_schedule=WeightDecayScheduleOptions.DISABLED,
        decay_rate=0.0,
        decay_warmup_batches=0,
        model_config=make_layer_stack_config(input_dim, output_dim),
    )


def make_diagonal_config(
    input_dim: int, output_dim: int
) -> StandardDynamicDiagonalConfig:
    return StandardDynamicDiagonalConfig(
        model_config=make_layer_stack_config(input_dim, output_dim),
    )


def make_mask_config(
    input_dim: int,
    output_dim: int,
    mask_threshold: float = 0.0,
) -> WeightInformedScoreAxisMaskConfig:
    return WeightInformedScoreAxisMaskConfig(
        mask_threshold=mask_threshold,
        mask_surrogate_scale=10.0,
        mask_floor=0.0,
        mask_dimension_option=MaskDimensionOptions.COLUMN,
        model_config=make_layer_stack_config(input_dim, output_dim),
    )


def assert_module_has_nonzero_grads(
    test_case: unittest.TestCase,
    module,
    name: str,
) -> None:
    params = [p for p in module.parameters() if p.requires_grad]
    test_case.assertTrue(params, f"{name} has no trainable parameters")
    nonzero_grads = [
        p.grad for p in params if p.grad is not None and torch.any(p.grad.abs() > 0)
    ]
    test_case.assertTrue(
        nonzero_grads,
        f"{name} did not receive a nonzero gradient",
    )


class TestLinearOptions(unittest.TestCase):
    def test_public_enum_values_are_stable(self):
        self.assertEqual(LinearOptions.LINEAR.value, 0)
        self.assertEqual(LinearOptions.ADAPTIVE.value, 1)

    def test_public_interface_is_exact_lazy_and_resolves_declared_owners(self):
        script = """\
import json
import sys

import emperor.linears as linears

private_modules = (
    "emperor.linears._config",
    "emperor.linears._layer",
    "emperor.linears._monitoring",
    "emperor.linears._monitoring.callback",
    "emperor.linears._options",
)
before = {name: name in sys.modules for name in private_modules}
print(
    json.dumps(
        {
            "all": linears.__all__,
            "before": before,
        }
    )
)
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(
            tuple(payload["all"]),
            (
                "LinearLayer",
                "LinearAbstract",
                "LinearLayerConfig",
                "LinearOptions",
                "LinearMonitorCallback",
            ),
        )
        self.assertEqual(
            payload["before"],
            {
                "emperor.linears._config": False,
                "emperor.linears._layer": False,
                "emperor.linears._monitoring": False,
                "emperor.linears._monitoring.callback": False,
                "emperor.linears._options": False,
            },
        )
        owners = {name: getattr(linears, name).__module__ for name in linears.__all__}
        self.assertEqual(
            owners,
            {
                "LinearLayer": "emperor.linears._layer",
                "LinearAbstract": "emperor.linears._layer",
                "LinearLayerConfig": "emperor.linears._config",
                "LinearOptions": "emperor.linears._options",
                "LinearMonitorCallback": ("emperor.linears._monitoring.callback"),
            },
        )
        self.assertTrue(
            all(
                getattr(linears, name) is getattr(linears, name)
                for name in linears.__all__
            )
        )

    def test_unknown_public_export_raises_the_standard_attribute_error(self):
        with self.assertRaisesRegex(
            AttributeError,
            "module 'emperor.linears' has no attribute 'MissingLinear'",
        ) as raised:
            _ = linears.MissingLinear

        self.assertIsInstance(raised.exception.__cause__, KeyError)


class TestLinearLayer(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
    ) -> LinearLayerConfig:
        return LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
        )

    def test_init_with_different_configation_options(self):
        bias_options = [True, False]
        for bias_flag in bias_options:
            message = f"Test failed for the inputs: {bias_flag}"
            with self.subTest(i=message):
                c = self.preset(bias_flag=bias_flag)
                m = LinearLayer(c)

                expected_weight_shape = (c.input_dim, c.output_dim)
                expected_bias_shape = (c.output_dim,)
                self.assertEqual(m.input_dim, c.input_dim)
                self.assertEqual(m.output_dim, c.output_dim)
                self.assertEqual(m.bias_flag, bias_flag)
                self.assertIsInstance(m.weight_params, torch.Tensor)
                self.assertEqual(m.weight_params.shape, expected_weight_shape)
                if bias_flag:
                    self.assertIsInstance(m.bias_params, torch.Tensor)
                    self.assertEqual(m.bias_params.shape, expected_bias_shape)
                else:
                    self.assertIsNone(m.bias_params)

    def test_seeded_initialization_is_reproducible_and_bias_starts_at_zero(self):
        config = self.preset(input_dim=3, output_dim=5, bias_flag=True)

        torch.manual_seed(721)
        first = LinearLayer(config)
        torch.manual_seed(721)
        second = LinearLayer(config)

        torch.testing.assert_close(
            first.weight_params,
            second.weight_params,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(torch.isfinite(first.weight_params).all())
        self.assertGreater(first.weight_params.norm().item(), 0.0)
        torch.testing.assert_close(
            first.bias_params,
            torch.zeros(5),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            second.bias_params,
            torch.zeros(5),
            rtol=0.0,
            atol=0.0,
        )

    def test_state_dict_topology_and_strict_roundtrip(self):
        for bias_flag, expected_keys in (
            (True, ("weight_params", "bias_params")),
            (False, ("weight_params",)),
        ):
            with self.subTest(bias_flag=bias_flag):
                cfg = self.preset(bias_flag=bias_flag)
                model = LinearLayer(cfg)
                with torch.no_grad():
                    model.weight_params.copy_(
                        torch.arange(
                            cfg.input_dim * cfg.output_dim,
                            dtype=model.weight_params.dtype,
                        ).reshape(cfg.input_dim, cfg.output_dim)
                        / 10.0
                    )
                    if model.bias_params is not None:
                        model.bias_params.copy_(
                            torch.arange(
                                cfg.output_dim,
                                dtype=model.bias_params.dtype,
                            )
                            / 4.0
                        )
                inputs = torch.arange(
                    2 * cfg.input_dim,
                    dtype=model.weight_params.dtype,
                ).reshape(2, cfg.input_dim)
                expected = model(inputs).detach().clone()
                state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }

                self.assertEqual(tuple(state), expected_keys)

                restored = LinearLayer(cfg)
                incompatible = restored.load_state_dict(state, strict=True)
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                torch.testing.assert_close(
                    restored(inputs),
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_forward(self):
        batch_size = 5
        bias_options = [True, False]
        input_params = output_params = [4, 8, 16]

        for input_dim in input_params:
            for output_dim in output_params:
                for bias_flag in bias_options:
                    message = (
                        f"input={input_dim}, output={output_dim}, bias={bias_flag}"
                    )
                    with self.subTest(i=message):
                        c = self.preset()
                        overrides = LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                        )
                        m = LinearLayer(c, overrides)

                        input_batch = torch.randn(batch_size, overrides.input_dim)
                        output = m.forward(input_batch)
                        expected_output_shape = (batch_size, overrides.output_dim)
                        self.assertEqual(output.shape, expected_output_shape)

    def test_gradients_flow_through_linear_layer(self):
        batch_size = 5
        input_params = [4, 8]
        output_params = [3, 6]
        for input_dim in input_params:
            for output_dim in output_params:
                for bias_flag in [True, False]:
                    with self.subTest(
                        input_dim=input_dim, output_dim=output_dim, bias_flag=bias_flag
                    ):
                        c = self.preset()
                        overrides = LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                        )
                        m = LinearLayer(c, overrides)

                        input_batch = torch.randn(
                            batch_size, overrides.input_dim, requires_grad=True
                        )
                        output = m.forward(input_batch)
                        output.sum().backward()

                        self.assertIsNotNone(m.weight_params.grad)
                        self.assertEqual(
                            m.weight_params.grad.shape, m.weight_params.shape
                        )

                        if bias_flag:
                            self.assertIsNotNone(m.bias_params.grad)
                            self.assertEqual(
                                m.bias_params.grad.shape, m.bias_params.shape
                            )
                        else:
                            self.assertIsNone(m.bias_params)

    def test_backward_and_optimizer_step_match_hand_calculated_values(self):
        model = LinearLayer(self.preset(input_dim=2, output_dim=2, bias_flag=True))
        with torch.no_grad():
            model.weight_params.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ]
                )
            )
            model.bias_params.copy_(torch.tensor([0.5, -0.5]))
        inputs = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            requires_grad=True,
        )
        output_coefficients = torch.tensor(
            [
                [1.0, -1.0],
                [2.0, 0.5],
            ]
        )

        loss = (model(inputs) * output_coefficients).sum()
        loss.backward()

        torch.testing.assert_close(
            model.weight_params.grad,
            torch.tensor(
                [
                    [7.0, 0.5],
                    [10.0, 0.0],
                ]
            ),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.bias_params.grad,
            torch.tensor([3.0, -0.5]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            inputs.grad,
            torch.tensor(
                [
                    [-1.0, -1.0],
                    [3.0, 8.0],
                ]
            ),
            rtol=0.0,
            atol=0.0,
        )
        for gradient in (
            model.weight_params.grad,
            model.bias_params.grad,
            inputs.grad,
        ):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0.0)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()

        torch.testing.assert_close(
            model.weight_params,
            torch.tensor(
                [
                    [0.3, 1.95],
                    [2.0, 4.0],
                ]
            ),
        )
        torch.testing.assert_close(
            model.bias_params,
            torch.tensor([0.2, -0.45]),
        )

    def test_output_matches_torch_linear(self):
        for bias_flag in [True, False]:
            with self.subTest(bias_flag=bias_flag):
                c = self.preset(input_dim=4, output_dim=3, bias_flag=bias_flag)
                m = LinearLayer(c)

                ref = torch.nn.Linear(4, 3, bias=bias_flag)
                with torch.no_grad():
                    ref.weight.copy_(m.weight_params.T)
                    if bias_flag:
                        ref.bias.copy_(m.bias_params)

                input_batch = torch.randn(2, 4)
                torch.testing.assert_close(m.forward(input_batch), ref(input_batch))

    def test_forward_matches_hand_calculated_affine_values_and_metadata(self):
        model = LinearLayer(
            self.preset(input_dim=2, output_dim=3, bias_flag=True)
        ).double()
        with torch.no_grad():
            model.weight_params.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0, -1.0],
                        [0.5, -2.0, 3.0],
                    ],
                    dtype=torch.float64,
                )
            )
            model.bias_params.copy_(
                torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64)
            )
        inputs = torch.tensor(
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ],
            dtype=torch.float64,
        ).T
        expected = torch.tensor(
            [
                [2.25, -2.5, 6.0],
                [5.25, -2.5, 10.0],
                [8.25, -2.5, 14.0],
            ],
            dtype=torch.float64,
        )

        self.assertFalse(inputs.is_contiguous())
        model.train()
        output = model(inputs)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (3, 3))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
        for sample_index in range(len(inputs)):
            torch.testing.assert_close(
                model(inputs[sample_index : sample_index + 1])[0],
                expected[sample_index],
                rtol=0.0,
                atol=0.0,
            )

        model.eval()
        torch.testing.assert_close(
            model(inputs),
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_disabling_bias_preserves_the_exact_matrix_product_baseline(self):
        without_bias = LinearLayer(
            self.preset(input_dim=2, output_dim=2, bias_flag=False)
        )
        with_bias = LinearLayer(self.preset(input_dim=2, output_dim=2, bias_flag=True))
        weights = torch.tensor(
            [
                [2.0, -1.0],
                [0.5, 3.0],
            ]
        )
        bias = torch.tensor([0.25, -0.75])
        with torch.no_grad():
            without_bias.weight_params.copy_(weights)
            with_bias.weight_params.copy_(weights)
            with_bias.bias_params.copy_(bias)
        inputs = torch.tensor(
            [
                [1.0, 2.0],
                [-3.0, 4.0],
            ]
        )
        expected_without_bias = torch.tensor(
            [
                [3.0, 5.0],
                [-4.0, 15.0],
            ]
        )

        baseline = without_bias(inputs)
        biased = with_bias(inputs)

        torch.testing.assert_close(
            baseline,
            expected_without_bias,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            biased - baseline,
            bias.expand_as(biased),
            rtol=0.0,
            atol=0.0,
        )

    def test_deterministic_output(self):
        c = self.preset(input_dim=4, output_dim=3, bias_flag=True)
        m = LinearLayer(c)

        input_batch = torch.randn(2, 4)
        output_1 = m.forward(input_batch)
        output_2 = m.forward(input_batch)
        torch.testing.assert_close(output_1, output_2)

    def test_forward_raises_on_1d_input(self):
        c = self.preset()
        m = LinearLayer(c)
        with self.assertRaises(ValueError):
            m.forward(torch.randn(c.input_dim))

    def test_forward_supports_leading_dimensions(self):
        c = self.preset()
        m = LinearLayer(c)
        input_batch = torch.randn(2, 3, c.input_dim)

        output = m.forward(input_batch)

        expected = torch.nn.functional.linear(
            input_batch,
            m.weight_params.T,
            m.bias_params,
        )
        self.assertEqual(output.shape, (2, 3, c.output_dim))
        torch.testing.assert_close(output, expected)

    def test_forward_raises_on_wrong_final_dimension(self):
        c = self.preset()
        m = LinearLayer(c)
        with self.assertRaises(ValueError):
            m.forward(torch.randn(2, 3, c.input_dim + 1))

    def test_init_raises_on_missing_input_dim(self):
        with self.assertRaises(ValueError) as raised:
            LinearLayer(LinearLayerConfig(output_dim=4, bias_flag=True))
        self.assertEqual(
            str(raised.exception),
            "input_dim is required for LinearLayerConfig, received None",
        )

    def test_init_raises_on_missing_output_dim(self):
        with self.assertRaises(ValueError) as raised:
            LinearLayer(LinearLayerConfig(input_dim=4, bias_flag=True))
        self.assertEqual(
            str(raised.exception),
            "output_dim is required for LinearLayerConfig, received None",
        )

    def test_init_rejects_missing_bias_and_wrong_field_types_exactly(self):
        with self.assertRaises(ValueError) as missing_bias:
            LinearLayer(LinearLayerConfig(input_dim=2, output_dim=3))
        self.assertEqual(
            str(missing_bias.exception),
            "bias_flag is required for LinearLayerConfig, received None",
        )

        invalid_types = (
            (
                LinearLayerConfig(
                    input_dim="2",
                    output_dim=3,
                    bias_flag=True,
                ),
                "input_dim must be int for LinearLayerConfig, got str",
            ),
            (
                LinearLayerConfig(
                    input_dim=2,
                    output_dim=3.0,
                    bias_flag=True,
                ),
                "output_dim must be int for LinearLayerConfig, got float",
            ),
            (
                LinearLayerConfig(
                    input_dim=2,
                    output_dim=3,
                    bias_flag=1,
                ),
                "bias_flag must be bool for LinearLayerConfig, got int",
            ),
        )
        for config, expected_message in invalid_types:
            with self.subTest(expected_message=expected_message):
                with self.assertRaises(TypeError) as raised:
                    LinearLayer(config)
                self.assertEqual(str(raised.exception), expected_message)

    def test_init_rejects_boolean_dimensions_before_parameter_allocation(self):
        for field_name in ("input_dim", "output_dim"):
            values = {
                "input_dim": 2,
                "output_dim": 3,
                "bias_flag": True,
            }
            values[field_name] = True
            with self.subTest(field_name=field_name):
                with self.assertRaises(TypeError) as raised:
                    LinearLayer(LinearLayerConfig(**values))
                self.assertEqual(
                    str(raised.exception),
                    f"{field_name} must be int for LinearLayerConfig, got bool",
                )

    def test_validation_failure_does_not_consume_parameter_rng(self):
        torch.manual_seed(913)
        expected_next_values = torch.rand(4)

        torch.manual_seed(913)
        with self.assertRaises(ValueError):
            LinearLayer(
                LinearLayerConfig(
                    input_dim=2,
                    output_dim=None,
                    bias_flag=True,
                )
            )
        actual_next_values = torch.rand(4)

        torch.testing.assert_close(
            actual_next_values,
            expected_next_values,
            rtol=0.0,
            atol=0.0,
        )

    def test_init_raises_on_zero_or_negative_dims(self):
        invalid_cases = [
            (
                "zero_input_dim",
                {"input_dim": 0, "output_dim": 4},
                "input_dim must be greater than 0, received 0",
            ),
            (
                "zero_output_dim",
                {"input_dim": 4, "output_dim": 0},
                "output_dim must be greater than 0, received 0",
            ),
            (
                "negative_input_dim",
                {"input_dim": -1, "output_dim": 4},
                "input_dim must be greater than 0, received -1",
            ),
            (
                "negative_output_dim",
                {"input_dim": 4, "output_dim": -1},
                "output_dim must be greater than 0, received -1",
            ),
        ]
        for case, kwargs, expected_message in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError) as raised:
                    LinearLayer(LinearLayerConfig(bias_flag=True, **kwargs))
                self.assertEqual(str(raised.exception), expected_message)

    def test_overrides_take_precedence_over_base_cfg(self):
        base = LinearLayerConfig(input_dim=4, output_dim=4, bias_flag=True)
        overrides = LinearLayerConfig(input_dim=8, output_dim=2, bias_flag=False)
        m = LinearLayer(base, overrides)

        self.assertEqual(m.input_dim, 8)
        self.assertEqual(m.output_dim, 2)
        self.assertEqual(m.bias_flag, False)
        self.assertIsNone(m.bias_params)
        self.assertIsNot(m.cfg, base)
        self.assertEqual(base.input_dim, 4)
        self.assertEqual(base.output_dim, 4)
        self.assertIs(base.bias_flag, True)

    def test_partial_overrides_keep_unset_fields_from_base_cfg(self):
        base = LinearLayerConfig(input_dim=4, output_dim=6, bias_flag=True)
        overrides = LinearLayerConfig(input_dim=8)
        m = LinearLayer(base, overrides)

        self.assertEqual(m.input_dim, 8)
        self.assertEqual(m.output_dim, 6)
        self.assertEqual(m.bias_flag, True)

    def test_config_build_returns_linear_layer(self):
        cfg = self.preset(input_dim=5, output_dim=3, bias_flag=True)
        m = cfg.build()

        self.assertIsInstance(m, LinearLayer)
        self.assertEqual(m.input_dim, cfg.input_dim)
        self.assertEqual(m.output_dim, cfg.output_dim)
        self.assertEqual(m.bias_flag, cfg.bias_flag)

    def test_config_build_applies_overrides(self):
        cfg = self.preset(input_dim=5, output_dim=3, bias_flag=True)
        overrides = LinearLayerConfig(input_dim=7, output_dim=2, bias_flag=False)
        m = cfg.build(overrides)

        self.assertIsInstance(m, LinearLayer)
        self.assertEqual(m.input_dim, overrides.input_dim)
        self.assertEqual(m.output_dim, overrides.output_dim)
        self.assertEqual(m.bias_flag, overrides.bias_flag)
        self.assertIsNone(m.bias_params)


class TestLinearLayerStack(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 24,
        output_dim: int = 6,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = (
            LayerNormPositionOptions.DISABLED
        ),
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.RESIDUAL
        ),
        stack_dropout_probability: float = 0.2,
        shared_halting_config: "StickBreakingConfig | None" = None,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        gate_config: "GateConfig | None" = None,
        use_gate_config: bool = True,
    ) -> LayerStackConfig:
        if use_gate_config and gate_config is None:
            gate_config = GateConfig(
                model_config=LayerStackConfig(
                    hidden_dim=hidden_dim,
                    num_layers=stack_num_layers,
                    last_layer_bias_option=last_layer_bias_option,
                    apply_output_pipeline_flag=apply_output_pipeline_flag,
                    layer_config=LayerConfig(
                        activation=stack_activation,
                        layer_norm_position=layer_norm_position,
                        residual_config=None
                        if stack_residual_connection_option is None
                        else ResidualConfig(option=stack_residual_connection_option),
                        dropout_probability=stack_dropout_probability,
                        halting_config=None,
                        gate_config=None,
                        layer_model_config=LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                        ),
                    ),
                ),
                option=LayerGateOptions.MULTIPLIER,
            )

        halting_config = None
        if (
            shared_halting_config is None
            and stack_num_layers > 1
            and input_dim == hidden_dim == output_dim
        ):
            halting_config = StickBreakingConfig(
                threshold=0.99,
                dropout_probability=0.0,
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
                        residual_config=None
                        if stack_residual_connection_option is None
                        else ResidualConfig(option=stack_residual_connection_option),
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
            shared_halting_config=shared_halting_config,
            layer_config=LayerConfig(
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_config=None
                if stack_residual_connection_option is None
                else ResidualConfig(option=stack_residual_connection_option),
                dropout_probability=stack_dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                layer_model_config=LinearLayerConfig(
                    bias_flag=bias_flag,
                ),
            ),
        )

    def test_stack_layers_contain_linear_layer(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(stack_num_layers=num_layers)
                m = LayerStack(cfg)

                layers = [m] if isinstance(m, Layer) else list(m)
                for i, layer in enumerate(layers):
                    with self.subTest(layer_index=i):
                        self.assertIsInstance(layer.model, LinearLayer)

    def test_stack_can_build_plain_layers_without_gate_config(self):
        cfg = self.preset(
            input_dim=4,
            hidden_dim=5,
            output_dim=3,
            stack_num_layers=2,
            use_gate_config=False,
        )
        self.assertIsNone(cfg.layer_config.gate_config)

        m = LayerStack(cfg)
        input_batch = torch.randn(2, cfg.input_dim)
        output = Layer.run_model_returning_hidden(m, input_batch)
        self.assertEqual(output.shape, (2, cfg.output_dim))

        for layer in list(m):
            self.assertIsNone(layer.gate_model)

    def test_gradients_flow_through_linear_layer_stack(self):
        num_layer_options = [1, 2, 3]
        for num_layers in num_layer_options:
            with self.subTest(num_layers=num_layers):
                batch_size = 2
                input_dim = 8
                output_dim = 4
                cfg = self.preset(
                    stack_num_layers=num_layers,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                m = LayerStack(cfg)

                input_batch = torch.randn(batch_size, input_dim, requires_grad=True)
                output = Layer.run_model_returning_hidden(m, input_batch)
                output.sum().backward()

                grads = [p.grad for p in m.parameters() if p.requires_grad]
                non_none_grads = [g for g in grads if g is not None]
                self.assertTrue(len(non_none_grads) > 0)


class TestAdaptiveLinearLayer(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        weight_config=None,
        bias_config=None,
        diagonal_config=None,
        mask_config=None,
    ) -> AdaptiveLinearLayerConfig:
        return AdaptiveLinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=weight_config,
                bias_config=bias_config,
                diagonal_config=diagonal_config,
                mask_config=mask_config,
                model_config=make_layer_stack_config(input_dim, output_dim),
            ),
        )

    def test_init_with_all_sub_configs_disabled(self):
        bias_options = [True, False]
        for bias_flag in bias_options:
            with self.subTest(bias_flag=bias_flag):
                cfg = self.preset(bias_flag=bias_flag)
                m = AdaptiveLinearLayer(cfg)

                self.assertEqual(m.input_dim, cfg.input_dim)
                self.assertEqual(m.output_dim, cfg.output_dim)
                self.assertEqual(m.bias_flag, bias_flag)
                self.assertEqual(m.weight_params.shape, (cfg.input_dim, cfg.output_dim))
                self.assertFalse(m.has_adaptive_augmentation)
                self.assertIsNone(m.adaptive_behaviour)
                if bias_flag:
                    self.assertEqual(m.bias_params.shape, (cfg.output_dim,))
                else:
                    self.assertIsNone(m.bias_params)

    def test_state_dict_topology_and_strict_roundtrip_for_adaptive_augmentation(self):
        disabled = AdaptiveLinearLayer(self.preset(input_dim=4, output_dim=3))
        self.assertEqual(
            tuple(disabled.state_dict()),
            ("weight_params", "bias_params"),
        )

        active = AdaptiveLinearLayer(
            self.preset(
                input_dim=4,
                output_dim=3,
                weight_config=make_weight_config(4, 3),
            )
        )
        state = active.state_dict()
        self.assertEqual(
            tuple(state),
            (
                "weight_params",
                "bias_params",
                "adaptive_behaviour.weight_model.scale",
                "adaptive_behaviour.weight_model.clamp_limit",
                "adaptive_behaviour.weight_model.decay_step",
                "adaptive_behaviour.weight_model.warmup_step",
                "adaptive_behaviour.weight_model.input_model.model.layers.0.model.weight_params",
                "adaptive_behaviour.weight_model.input_model.model.layers.0.model.bias_params",
                "adaptive_behaviour.weight_model.input_model.model.layers.1.model.weight_params",
                "adaptive_behaviour.weight_model.input_model.model.layers.1.model.bias_params",
                "adaptive_behaviour.weight_model.output_model.model.layers.0.model.weight_params",
                "adaptive_behaviour.weight_model.output_model.model.layers.0.model.bias_params",
                "adaptive_behaviour.weight_model.output_model.model.layers.1.model.weight_params",
                "adaptive_behaviour.weight_model.output_model.model.layers.1.model.bias_params",
            ),
        )

        restored = AdaptiveLinearLayer(
            self.preset(
                input_dim=4,
                output_dim=3,
                weight_config=make_weight_config(4, 3),
            )
        )
        incompatible = restored.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        for name, tensor in state.items():
            torch.testing.assert_close(restored.state_dict()[name], tensor)

    def test_init_raises_on_missing_adaptive_augmentation_config(self):
        cfg = AdaptiveLinearLayerConfig(
            input_dim=4,
            output_dim=3,
            bias_flag=True,
            adaptive_augmentation_config=None,
        )
        with self.assertRaises(ValueError):
            AdaptiveLinearLayer(cfg)

    def test_validation_failure_does_not_consume_parameter_rng(self):
        cfg = AdaptiveLinearLayerConfig(
            input_dim=4,
            output_dim=3,
            bias_flag=True,
            adaptive_augmentation_config=None,
        )
        torch.manual_seed(913)
        expected_next_values = torch.randn(4)

        torch.manual_seed(913)
        with self.assertRaises(ValueError):
            AdaptiveLinearLayer(cfg)
        actual_next_values = torch.randn(4)

        torch.testing.assert_close(actual_next_values, expected_next_values)

    def test_init_raises_when_dynamic_bias_is_configured_without_layer_bias(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=3,
            bias_flag=False,
            bias_config=make_bias_config(4, 3),
        )

        with self.assertRaises(ValueError):
            AdaptiveLinearLayer(cfg)

    def test_output_matches_linear_layer_when_all_sub_configs_disabled(self):
        input_dim = 4
        output_dim = 3
        batch_size = 2
        adaptive_cfg = self.preset(
            input_dim=input_dim, output_dim=output_dim, bias_flag=True
        )
        adaptive_layer = AdaptiveLinearLayer(adaptive_cfg)

        linear_cfg = LinearLayerConfig(
            input_dim=input_dim, output_dim=output_dim, bias_flag=True
        )
        linear_layer = LinearLayer(linear_cfg)

        with torch.no_grad():
            linear_layer.weight_params.copy_(adaptive_layer.weight_params)
            linear_layer.bias_params.copy_(adaptive_layer.bias_params)

        input_batch = torch.randn(batch_size, input_dim)
        torch.testing.assert_close(
            adaptive_layer.forward(input_batch),
            linear_layer.forward(input_batch),
        )

    def test_adaptive_generator_configs_preserve_requested_dimensions(self):
        cases = [
            ("weight", make_weight_config(5, 3).model_config),
            ("bias", make_bias_config(5, 3).model_config),
            ("diagonal", make_diagonal_config(5, 3).model_config),
            ("mask", make_mask_config(5, 3).model_config),
        ]

        for name, model_config in cases:
            with self.subTest(name=name):
                self.assertEqual(model_config.input_dim, 5)
                self.assertEqual(model_config.output_dim, 3)
                self.assertEqual(model_config.num_layers, 2)
                self.assertEqual(model_config.hidden_dim, 5)

    def test_per_sample_weight_callback_matches_manual_matrix_product(self):
        cfg = self.preset(input_dim=2, output_dim=3, bias_flag=True)
        m = AdaptiveLinearLayer(cfg)
        input_batch = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        weights = torch.tensor(
            [
                [
                    [1.0, 0.0, 2.0],
                    [0.0, 1.0, 3.0],
                ],
                [
                    [2.0, 1.0, 0.0],
                    [1.0, 0.0, -1.0],
                ],
            ]
        )
        bias = torch.tensor(
            [
                [0.5, -0.5, 1.0],
                [1.0, 2.0, -1.0],
            ]
        )

        expected = (
            torch.stack(
                [
                    input_batch[0].matmul(weights[0]),
                    input_batch[1].matmul(weights[1]),
                ]
            )
            + bias
        )
        output = m._compute_affine_transformation_callback(
            weights,
            bias,
            input_batch,
        )

        torch.testing.assert_close(output, expected)

    def test_active_adaptive_components_change_output_and_receive_gradients(self):
        input_dim = 4
        output_dim = 3
        input_batch = torch.tensor(
            [
                [0.1, -0.2, 0.3, -0.4],
                [0.5, 0.6, -0.7, -0.8],
            ],
            requires_grad=True,
        )
        cases = [
            (
                "weight",
                {"weight_config": make_weight_config(input_dim, output_dim)},
                "weight_model",
            ),
            (
                "bias",
                {"bias_config": make_bias_config(input_dim, output_dim)},
                "bias_model",
            ),
            (
                "diagonal",
                {"diagonal_config": make_diagonal_config(input_dim, output_dim)},
                "diagonal_model",
            ),
            (
                "mask",
                {
                    "mask_config": make_mask_config(
                        input_dim,
                        output_dim,
                        mask_threshold=0.0,
                    )
                },
                "mask_model",
            ),
        ]

        for seed, (name, kwargs, component_name) in enumerate(cases):
            with self.subTest(name=name):
                torch.manual_seed(seed)
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=True,
                    **kwargs,
                )
                m = AdaptiveLinearLayer(cfg)
                batch = input_batch.detach().clone().requires_grad_(True)
                baseline = m._compute_affine_transformation_callback(
                    m.weight_params,
                    m.bias_params,
                    batch,
                ).detach()

                output = m.forward(batch)
                self.assertFalse(torch.allclose(output.detach(), baseline, atol=1e-6))
                output.sum().backward()

                component = getattr(m.adaptive_behaviour, component_name)
                assert_module_has_nonzero_grads(self, component, name)

    def test_init_with_different_sub_config_combinations(self):
        input_dim = 12
        output_dim = 6
        bias_options = [True, False]
        weight_options = [None, make_weight_config]
        bias_config_options = [None, make_bias_config]
        diagonal_options = [None, make_diagonal_config]
        mask_options = [None, make_mask_config]

        for bias_flag in bias_options:
            for weight_builder in weight_options:
                for bias_builder in bias_config_options:
                    for diagonal_builder in diagonal_options:
                        for mask_builder in mask_options:
                            message = (
                                f"bias_flag={bias_flag}, "
                                f"weight={'set' if weight_builder else 'none'}, "
                                f"bias_config={'set' if bias_builder else 'none'}, "
                                f"diagonal={'set' if diagonal_builder else 'none'}, "
                                f"mask={'set' if mask_builder else 'none'}"
                            )
                            with self.subTest(message=message):
                                weight_config = (
                                    weight_builder(input_dim, output_dim)
                                    if weight_builder
                                    else None
                                )
                                bias_config = (
                                    bias_builder(input_dim, output_dim)
                                    if bias_builder and bias_flag
                                    else None
                                )
                                diagonal_config = (
                                    diagonal_builder(input_dim, output_dim)
                                    if diagonal_builder
                                    else None
                                )
                                mask_config = (
                                    mask_builder(input_dim, output_dim)
                                    if mask_builder
                                    else None
                                )
                                cfg = self.preset(
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    bias_flag=bias_flag,
                                    weight_config=weight_config,
                                    bias_config=bias_config,
                                    diagonal_config=diagonal_config,
                                    mask_config=mask_config,
                                )
                                m = AdaptiveLinearLayer(cfg)
                                expected_has_adaptive_augmentation = any(
                                    config is not None
                                    for config in (
                                        weight_config,
                                        bias_config,
                                        diagonal_config,
                                        mask_config,
                                    )
                                )

                                self.assertEqual(m.input_dim, cfg.input_dim)
                                self.assertEqual(m.output_dim, cfg.output_dim)
                                self.assertEqual(
                                    m.has_adaptive_augmentation,
                                    expected_has_adaptive_augmentation,
                                )
                                self.assertIsInstance(m.weight_params, torch.Tensor)
                                if expected_has_adaptive_augmentation:
                                    self.assertIsNotNone(m.adaptive_behaviour)
                                else:
                                    self.assertIsNone(m.adaptive_behaviour)
                                if bias_flag:
                                    self.assertIsInstance(m.bias_params, torch.Tensor)
                                else:
                                    self.assertIsNone(m.bias_params)

    def test_forward(self):
        batch_size = 2
        bias_options = [True, False]
        input_params = output_params = [8, 16]
        weight_options = [None, make_weight_config]
        bias_config_options = [None, make_bias_config]
        diagonal_options = [None, make_diagonal_config]
        mask_options = [None, make_mask_config]

        for bias_flag in bias_options:
            for input_dim in input_params:
                for output_dim in output_params:
                    for weight_builder in weight_options:
                        for bias_builder in bias_config_options:
                            for diagonal_builder in diagonal_options:
                                for mask_builder in mask_options:
                                    message = (
                                        f"bias_flag={bias_flag}, "
                                        f"input={input_dim}, output={output_dim}, "
                                        f"weight={bool(weight_builder)}, "
                                        f"bias={bool(bias_builder)}, "
                                        f"diagonal={bool(diagonal_builder)}, "
                                        f"mask={bool(mask_builder)}"
                                    )
                                    with self.subTest(message=message):
                                        weight_config = (
                                            weight_builder(input_dim, output_dim)
                                            if weight_builder
                                            else None
                                        )
                                        bias_config = (
                                            bias_builder(input_dim, output_dim)
                                            if bias_builder and bias_flag
                                            else None
                                        )
                                        diagonal_config = (
                                            diagonal_builder(input_dim, output_dim)
                                            if diagonal_builder
                                            else None
                                        )
                                        mask_config = (
                                            mask_builder(input_dim, output_dim)
                                            if mask_builder
                                            else None
                                        )
                                        cfg = self.preset(
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            bias_flag=bias_flag,
                                            weight_config=weight_config,
                                            bias_config=bias_config,
                                            diagonal_config=diagonal_config,
                                            mask_config=mask_config,
                                        )

                                        m = AdaptiveLinearLayer(cfg)
                                        input_batch = torch.randn(batch_size, input_dim)
                                        output = m.forward(input_batch)
                                        self.assertEqual(
                                            output.shape, (batch_size, output_dim)
                                        )

    def test_config_build_returns_adaptive_linear_layer(self):
        cfg = self.preset(input_dim=5, output_dim=3, bias_flag=True)
        m = cfg.build()

        self.assertIsInstance(m, AdaptiveLinearLayer)
        self.assertEqual(m.input_dim, cfg.input_dim)
        self.assertEqual(m.output_dim, cfg.output_dim)
        self.assertEqual(m.bias_flag, cfg.bias_flag)

    def test_config_build_applies_overrides(self):
        cfg = self.preset(input_dim=5, output_dim=3, bias_flag=True)
        overrides = AdaptiveLinearLayerConfig(
            input_dim=7,
            output_dim=2,
            bias_flag=False,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=None,
                bias_config=None,
                diagonal_config=None,
                mask_config=None,
                model_config=make_layer_stack_config(7, 2),
            ),
        )
        m = cfg.build(overrides)

        self.assertIsInstance(m, AdaptiveLinearLayer)
        self.assertEqual(m.input_dim, overrides.input_dim)
        self.assertEqual(m.output_dim, overrides.output_dim)
        self.assertEqual(m.bias_flag, overrides.bias_flag)
        self.assertIsNone(m.bias_params)

    def test_gradients_flow_through_adaptive_linear_layer(self):
        cfg = self.preset(input_dim=4, output_dim=3, bias_flag=True)
        m = AdaptiveLinearLayer(cfg)

        input_batch = torch.randn(2, cfg.input_dim, requires_grad=True)
        output = m.forward(input_batch)
        output.sum().backward()

        grads = [p.grad for p in m.parameters() if p.requires_grad]
        non_none_grads = [g for g in grads if g is not None]
        self.assertTrue(len(non_none_grads) > 0)


class TestLinearLayerAdaptiveStack(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 4,
        bias_flag: bool = True,
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_connection_option: ResidualConnectionOptions | None = None,
        stack_dropout_probability: float = 0.0,
        layer_norm_position: LayerNormPositionOptions = (
            LayerNormPositionOptions.DISABLED
        ),
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        weight_config=None,
        bias_config=None,
        diagonal_config=None,
        mask_config=None,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=True,
            layer_config=LayerConfig(
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_config=None
                if stack_residual_connection_option is None
                else ResidualConfig(option=stack_residual_connection_option),
                dropout_probability=stack_dropout_probability,
                gate_config=None,
                halting_config=None,
                layer_model_config=AdaptiveLinearLayerConfig(
                    bias_flag=bias_flag,
                    adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                        weight_config=weight_config,
                        bias_config=bias_config,
                        diagonal_config=diagonal_config,
                        mask_config=mask_config,
                        model_config=make_layer_stack_config(input_dim, output_dim),
                    ),
                ),
            ),
        )

    def test_stack_layers_contain_adaptive_linear_layer(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(stack_num_layers=num_layers)
                m = LayerStack(cfg)

                self.assertIsInstance(m, LayerStack)

                layers = list(m)
                for i, layer in enumerate(layers):
                    with self.subTest(layer_index=i):
                        self.assertIsInstance(layer.model, AdaptiveLinearLayer)

    def test_forward_with_different_sub_config_combinations(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        stack_num_layers = 2
        bias_options = [True, False]
        weight_options = [None, make_weight_config]
        bias_config_options = [None, make_bias_config]
        diagonal_options = [None, make_diagonal_config]
        mask_options = [None, make_mask_config]

        for bias_flag in bias_options:
            for weight_builder in weight_options:
                for bias_builder in bias_config_options:
                    for diagonal_builder in diagonal_options:
                        for mask_builder in mask_options:
                            message = (
                                f"bias_flag={bias_flag}, "
                                f"weight={'set' if weight_builder else 'none'}, "
                                f"bias_config={'set' if bias_builder else 'none'}, "
                                f"diagonal={'set' if diagonal_builder else 'none'}, "
                                f"mask={'set' if mask_builder else 'none'}"
                            )
                            with self.subTest(message=message):
                                weight_config = (
                                    weight_builder(input_dim, output_dim)
                                    if weight_builder
                                    else None
                                )
                                bias_config = (
                                    bias_builder(input_dim, output_dim)
                                    if bias_builder and bias_flag
                                    else None
                                )
                                diagonal_config = (
                                    diagonal_builder(input_dim, output_dim)
                                    if diagonal_builder
                                    else None
                                )
                                mask_config = (
                                    mask_builder(input_dim, output_dim)
                                    if mask_builder
                                    else None
                                )
                                cfg = self.preset(
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    bias_flag=bias_flag,
                                    stack_num_layers=stack_num_layers,
                                    weight_config=weight_config,
                                    bias_config=bias_config,
                                    diagonal_config=diagonal_config,
                                    mask_config=mask_config,
                                )
                                m = LayerStack(cfg)

                                input_batch = torch.randn(batch_size, input_dim)
                                output = Layer.run_model_returning_hidden(
                                    m, input_batch
                                )
                                self.assertEqual(output.shape, (batch_size, output_dim))

    def test_gradients_flow_through_adaptive_linear_layer_stack(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        num_layer_options = [1, 2, 3]
        bias_options = [True, False]
        weight_options = [None, make_weight_config]
        bias_config_options = [None, make_bias_config]
        diagonal_options = [None, make_diagonal_config]
        mask_options = [None, make_mask_config]

        for num_layers in num_layer_options:
            for bias_flag in bias_options:
                for weight_builder in weight_options:
                    for bias_builder in bias_config_options:
                        for diagonal_builder in diagonal_options:
                            for mask_builder in mask_options:
                                message = (
                                    f"num_layers={num_layers}, "
                                    f"bias_flag={bias_flag}, "
                                    f"weight={bool(weight_builder)}, "
                                    f"bias={bool(bias_builder)}, "
                                    f"diagonal={bool(diagonal_builder)}, "
                                    f"mask={bool(mask_builder)}"
                                )
                                with self.subTest(message=message):
                                    weight_config = (
                                        weight_builder(input_dim, output_dim)
                                        if weight_builder
                                        else None
                                    )
                                    bias_config = (
                                        bias_builder(input_dim, output_dim)
                                        if bias_builder and bias_flag
                                        else None
                                    )
                                    diagonal_config = (
                                        diagonal_builder(input_dim, output_dim)
                                        if diagonal_builder
                                        else None
                                    )
                                    mask_config = (
                                        mask_builder(input_dim, output_dim)
                                        if mask_builder
                                        else None
                                    )
                                    cfg = self.preset(
                                        stack_num_layers=num_layers,
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        bias_flag=bias_flag,
                                        weight_config=weight_config,
                                        bias_config=bias_config,
                                        diagonal_config=diagonal_config,
                                        mask_config=mask_config,
                                    )
                                    m = LayerStack(cfg)

                                    input_batch = torch.randn(
                                        batch_size, input_dim, requires_grad=True
                                    )
                                    output = Layer.run_model_returning_hidden(
                                        m, input_batch
                                    )
                                    output.sum().backward()

                                    grads = [
                                        p.grad
                                        for p in m.parameters()
                                        if p.requires_grad
                                    ]
                                    non_none_grads = [g for g in grads if g is not None]
                                    self.assertTrue(len(non_none_grads) > 0)

    def test_active_adaptive_generator_components_receive_gradients(self):
        input_dim = hidden_dim = output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            stack_num_layers=2,
            bias_flag=True,
            weight_config=make_weight_config(input_dim, output_dim),
            bias_config=make_bias_config(input_dim, output_dim),
            diagonal_config=make_diagonal_config(input_dim, output_dim),
            mask_config=make_mask_config(
                input_dim,
                output_dim,
                mask_threshold=0.0,
            ),
        )
        m = LayerStack(cfg)

        input_batch = torch.tensor(
            [
                [0.1, -0.2, 0.3, -0.4],
                [0.5, 0.6, -0.7, -0.8],
            ],
            requires_grad=True,
        )
        output = Layer.run_model_returning_hidden(m, input_batch)
        output.sum().backward()

        for layer_index, layer in enumerate(list(m)):
            adaptive = layer.model.adaptive_behaviour
            for component_name in (
                "weight_model",
                "bias_model",
                "diagonal_model",
                "mask_model",
            ):
                with self.subTest(layer_index=layer_index, component=component_name):
                    component = getattr(adaptive, component_name)
                    assert_module_has_nonzero_grads(self, component, component_name)
