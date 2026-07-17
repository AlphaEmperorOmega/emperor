from __future__ import annotations

import unittest
from collections.abc import Callable
from dataclasses import replace

import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdditiveDynamicBiasConfig,
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    LayeredWeightedBankDynamicWeightConfig,
    MaskDimensionOptions,
    PerAxisScoreMaskConfig,
    SingleModelDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TopSliceAxisMaskConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._augmentation import (
    AdaptiveParameterAugmentation,
)
from emperor.augmentations.adaptive_parameters._biases.validation import (
    DynamicBiasValidator,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveGeneratorValidatorBase,
    AdaptiveParameterAugmentationValidator,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayer,
    DepthMappingLayerConfig,
    DepthMappingLayerStack,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig


def linear_stack_config(
    input_dim: int = 2,
    output_dim: int = 3,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=True,
            ),
        ),
    )


def depth_mapping_config(
    model_config: LayerStackConfig | None = None,
) -> DepthMappingHandlerConfig:
    return DepthMappingHandlerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=True,
        generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
        model_config=model_config or linear_stack_config(),
    )


class AdaptiveParameterValidationMutationContractTests(unittest.TestCase):
    def assert_exact_error(
        self,
        error_type: type[Exception],
        message: str,
        action: Callable[[], object],
    ) -> None:
        with self.assertRaises(error_type) as raised:
            action()
        self.assertEqual(str(raised.exception), message)

    def test_runtime_tensor_validation_reports_every_exact_failure(self) -> None:
        model = AdaptiveParameterAugmentation(
            AdaptiveParameterAugmentationConfig(input_dim=2, output_dim=3)
        )

        def callback(weights, bias, inputs):
            return inputs @ weights + bias

        weights = torch.ones(2, 3)
        bias = torch.ones(3)
        inputs = torch.ones(2, 2)
        cases = (
            (
                TypeError,
                "affine_transform_callback must be callable, received int.",
                lambda: model(7, weights, bias, inputs),
            ),
            (
                TypeError,
                "affine_transform_callback must be callable, received NoneType.",
                lambda: model(None, weights, bias, inputs),
            ),
            (
                TypeError,
                "input must be a Tensor, received list.",
                lambda: model(callback, weights, bias, [[1.0, 2.0]]),
            ),
            (
                ValueError,
                "AdaptiveParameterAugmentation expects a 2D input tensor "
                "(batch_size, input_dim), received a 3D tensor with shape "
                "(2, 1, 2).",
                lambda: model(
                    callback,
                    weights,
                    bias,
                    torch.ones(2, 1, 2),
                ),
            ),
            (
                ValueError,
                "AdaptiveParameterAugmentation input feature dimension must match "
                "input_dim, received input_dim=2 and input shape (2, 1).",
                lambda: model(callback, weights, bias, torch.ones(2, 1)),
            ),
            (
                TypeError,
                "weight_params must be a Tensor, received list.",
                lambda: model(callback, [[1.0]], bias, inputs),
            ),
            (
                ValueError,
                "weight_params must be a 2D tensor (input_dim, output_dim) or a "
                "3D tensor (batch_size, input_dim, output_dim), received a 4D "
                "tensor with shape (1, 1, 2, 3).",
                lambda: model(
                    callback,
                    torch.ones(1, 1, 2, 3),
                    bias,
                    inputs,
                ),
            ),
            (
                ValueError,
                "weight_params trailing dimensions must match "
                "(input_dim, output_dim), expected (2, 3), received shape (2, 2).",
                lambda: model(callback, torch.ones(2, 2), bias, inputs),
            ),
            (
                ValueError,
                "weight_params batch dimension must match input batch dimension, "
                "received weight_params shape (3, 2, 3) and input shape (2, 2).",
                lambda: model(
                    callback,
                    torch.ones(3, 2, 3),
                    bias,
                    inputs,
                ),
            ),
            (
                TypeError,
                "bias_params must be a Tensor when provided, received list.",
                lambda: model(callback, weights, [1.0], inputs),
            ),
            (
                ValueError,
                "bias_params must be a 1D tensor (output_dim) or a 2D tensor "
                "(batch_size, output_dim), received a 3D tensor with shape "
                "(1, 1, 3).",
                lambda: model(
                    callback,
                    weights,
                    torch.ones(1, 1, 3),
                    inputs,
                ),
            ),
            (
                ValueError,
                "bias_params feature dimension must match output_dim, received "
                "output_dim=3 and bias_params shape (2,).",
                lambda: model(callback, weights, torch.ones(2), inputs),
            ),
            (
                ValueError,
                "bias_params batch dimension must match input batch dimension, "
                "received bias_params shape (3, 3) and input shape (2, 2).",
                lambda: model(callback, weights, torch.ones(3, 3), inputs),
            ),
        )

        for error_type, message, action in cases:
            with self.subTest(message=message):
                self.assert_exact_error(error_type, message, action)

        rectangular_model = AdaptiveParameterAugmentation(
            AdaptiveParameterAugmentationConfig(input_dim=2, output_dim=5)
        )
        rectangular_inputs = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        rectangular_weights = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        rectangular_bias = torch.arange(15, dtype=torch.float32).reshape(3, 5)

        actual = rectangular_model(
            callback,
            rectangular_weights,
            rectangular_bias,
            rectangular_inputs,
        )

        self.assertTrue(
            torch.equal(
                actual,
                rectangular_inputs @ rectangular_weights + rectangular_bias,
            )
        )

    def test_adaptive_augmentation_dimension_and_model_errors_are_exact(
        self,
    ) -> None:
        for field_name, value in (
            ("input_dim", 0),
            ("output_dim", -1),
            ("output_dim", 0),
        ):
            with self.subTest(field_name=field_name, value=value):
                dimensions = {"input_dim": 2, "output_dim": 3}
                dimensions[field_name] = value
                config = AdaptiveParameterAugmentationConfig(**dimensions)
                self.assert_exact_error(
                    ValueError,
                    f"{field_name} must be a positive integer, received {value!r}.",
                    lambda config=config: AdaptiveParameterAugmentation(config),
                )

        model = AdaptiveParameterAugmentation(
            AdaptiveParameterAugmentationConfig(input_dim=2, output_dim=3)
        )
        for field_name, value in (
            ("input_dim", None),
            ("input_dim", 1.5),
            ("output_dim", None),
            ("output_dim", 1.5),
        ):
            with self.subTest(field_name=field_name, value=value):
                original = getattr(model, field_name)
                setattr(model, field_name, value)
                try:
                    self.assert_exact_error(
                        ValueError,
                        f"{field_name} must be a positive integer, received {value!r}.",
                        lambda: (
                            AdaptiveParameterAugmentationValidator._validate_dimensions(
                                model
                            )
                        ),
                    )
                finally:
                    setattr(model, field_name, original)

        unit_model = AdaptiveParameterAugmentation(
            AdaptiveParameterAugmentationConfig(input_dim=1, output_dim=1)
        )
        unit_inputs = torch.tensor([[2.0], [-3.0]])
        unit_weights = torch.tensor([[4.0]])
        unit_bias = torch.tensor([0.5])
        self.assertTrue(
            torch.equal(
                unit_model(
                    lambda weights, bias, inputs: inputs @ weights + bias,
                    unit_weights,
                    unit_bias,
                    unit_inputs,
                ),
                torch.tensor([[8.5], [-11.5]]),
            )
        )

        wrong_model_config = LinearLayerConfig(
            input_dim=2,
            output_dim=3,
            bias_flag=True,
        )
        self.assert_exact_error(
            TypeError,
            "model_config must be a LayerStackConfig when provided, got "
            "LinearLayerConfig.",
            lambda: AdaptiveParameterAugmentation(
                AdaptiveParameterAugmentationConfig(
                    input_dim=2,
                    output_dim=3,
                    model_config=wrong_model_config,
                )
            ),
        )

    def test_each_sub_config_type_and_missing_model_error_is_exact(self) -> None:
        model = AdaptiveParameterAugmentation(
            AdaptiveParameterAugmentationConfig(input_dim=2, output_dim=3)
        )
        wrong_config = LinearLayerConfig(
            input_dim=2,
            output_dim=3,
            bias_flag=True,
        )
        expected_types = (
            ("weight_config", "DynamicWeightConfig"),
            ("diagonal_config", "DynamicDiagonalConfig"),
            ("bias_config", "DynamicBiasConfig"),
            ("mask_config", "AxisMaskConfig"),
        )
        for field_name, expected_type in expected_types:
            with self.subTest(field_name=field_name, failure="type"):
                setattr(model, field_name, wrong_config)
                self.assert_exact_error(
                    TypeError,
                    f"{field_name} must be a {expected_type} instance, got "
                    "LinearLayerConfig.",
                    lambda: (
                        AdaptiveParameterAugmentationValidator._validate_sub_configs(
                            model
                        )
                    ),
                )
                setattr(model, field_name, None)

        invalid_nested_model = SingleModelDynamicWeightConfig(
            input_dim=2,
            output_dim=2,
            generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            model_config=wrong_config,
            normalization_option=WeightNormalizationOptions.DISABLED,
            normalization_position_option=(WeightNormalizationPositionOptions.DISABLED),
        )
        self.assert_exact_error(
            TypeError,
            "weight_config.model_config must be a LayerStackConfig when provided, "
            "got LinearLayerConfig.",
            lambda: AdaptiveParameterAugmentation(
                AdaptiveParameterAugmentationConfig(
                    input_dim=2,
                    output_dim=2,
                    weight_config=invalid_nested_model,
                )
            ),
        )

        missing_model_configs = (
            ("weight_config", SingleModelDynamicWeightConfig()),
            ("diagonal_config", StandardDynamicDiagonalConfig()),
            ("bias_config", AdditiveDynamicBiasConfig()),
            ("mask_config", PerAxisScoreMaskConfig()),
        )
        for field_name, sub_config in missing_model_configs:
            with self.subTest(field_name=field_name, failure="missing model"):
                config = AdaptiveParameterAugmentationConfig(
                    input_dim=2,
                    output_dim=3,
                    **{field_name: sub_config},
                )
                self.assert_exact_error(
                    ValueError,
                    f"{type(sub_config).__name__} requires a model_config but none "
                    "was provided on the sub-config or the parent "
                    "AdaptiveParameterAugmentationConfig.",
                    lambda config=config: AdaptiveParameterAugmentation(config),
                )

    def test_decay_validation_reports_exact_rate_and_warmup_contracts(self) -> None:
        cases = (
            (
                WeightDecayScheduleOptions.EXPONENTIAL,
                None,
                0,
                "decay_rate is required for AdditiveDynamicBiasConfig, received None",
            ),
            (
                WeightDecayScheduleOptions.EXPONENTIAL,
                0.0,
                0,
                "decay_rate must be greater than 0.0 when decay_schedule is "
                "EXPONENTIAL, received 0.0.",
            ),
            (
                WeightDecayScheduleOptions.LINEAR,
                1.0,
                0,
                "decay_rate must be less than 1.0 for LINEAR, received 1.0.",
            ),
            (
                WeightDecayScheduleOptions.MULTIPLICATIVE,
                1.0,
                0,
                "decay_rate must be less than 1.0 for MULTIPLICATIVE, received 1.0.",
            ),
            (
                WeightDecayScheduleOptions.EXPONENTIAL,
                0.1,
                -1,
                "decay_warmup_batches must be >= 0, received -1.",
            ),
        )
        for schedule, rate, warmup, message in cases:
            with self.subTest(schedule=schedule, rate=rate, warmup=warmup):
                config = AdditiveDynamicBiasConfig(
                    input_dim=2,
                    output_dim=3,
                    decay_schedule=schedule,
                    decay_rate=rate,
                    decay_warmup_batches=warmup,
                    model_config=linear_stack_config(),
                )
                self.assert_exact_error(ValueError, message, config.build)

        config = AdditiveDynamicBiasConfig(
            input_dim=2,
            output_dim=3,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=None,
            decay_warmup_batches=0,
            model_config=linear_stack_config(),
        )
        self.assert_exact_error(
            ValueError,
            "decay_rate must be greater than 0.0 when decay_schedule is "
            "EXPONENTIAL, received None.",
            lambda: AdaptiveGeneratorValidatorBase.validate_decay_parameters(config),
        )

        weight_config = SingleModelDynamicWeightConfig(
            input_dim=2,
            output_dim=2,
            generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=-0.1,
            decay_warmup_batches=0,
            model_config=linear_stack_config(2, 2),
            normalization_option=WeightNormalizationOptions.L2_SCALE,
            normalization_position_option=(
                WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
            ),
        )
        self.assert_exact_error(
            ValueError,
            "decay_rate must be greater than 0.0 when decay_schedule is "
            "EXPONENTIAL, received -0.1.",
            weight_config.build,
        )

    def test_bank_factor_and_required_bias_errors_are_exact(self) -> None:
        for factor, message in (
            (
                None,
                "WeightedBankDynamicBias requires bank_expansion_factor to be a "
                "BankExpansionFactorOptions value, received None.",
            ),
            (
                BankExpansionFactorOptions.DISABLED,
                "WeightedBankDynamicBias requires bank_expansion_factor > 0, "
                "received BankExpansionFactorOptions.DISABLED. Use FACTOR_OF_ONE, "
                "FACTOR_OF_TWO, FACTOR_OF_THREE, or FACTOR_OF_FOUR.",
            ),
        ):
            with self.subTest(owner="bias", factor=factor):
                config = WeightedBankDynamicBiasConfig(
                    input_dim=2,
                    output_dim=3,
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    model_config=linear_stack_config(),
                    bank_expansion_factor=factor,
                )
                self.assert_exact_error(ValueError, message, config.build)

        for factor, message in (
            (
                None,
                "LayeredWeightedBankDynamicWeight requires bank_expansion_factor "
                "to be a BankExpansionFactorOptions value, received None.",
            ),
            (
                BankExpansionFactorOptions.DISABLED,
                "LayeredWeightedBankDynamicWeight requires bank_expansion_factor "
                "> 0, received BankExpansionFactorOptions.DISABLED. Use "
                "FACTOR_OF_ONE, FACTOR_OF_TWO, FACTOR_OF_THREE, or FACTOR_OF_FOUR.",
            ),
        ):
            with self.subTest(owner="weight", factor=factor):
                config = LayeredWeightedBankDynamicWeightConfig(
                    input_dim=2,
                    output_dim=3,
                    generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    model_config=linear_stack_config(),
                    bank_expansion_factor=factor,
                )
                self.assert_exact_error(ValueError, message, config.build)

        self.assert_exact_error(
            ValueError,
            "bias_params must not be None. Provide a valid bias tensor for this "
            "dynamic bias strategy.",
            lambda: DynamicBiasValidator.ensure_parameters_exist(None),
        )

    def test_mask_validation_boundaries_have_exact_errors(self) -> None:
        base = {
            "input_dim": 2,
            "output_dim": 3,
            "mask_threshold": 0.5,
            "mask_surrogate_scale": 5.0,
            "mask_floor": 0.0,
            "mask_dimension_option": MaskDimensionOptions.ROW,
            "model_config": linear_stack_config(),
        }
        cases = (
            (
                {"mask_threshold": -0.1},
                "mask_threshold must be between 0.0 and 1.0 inclusive, received -0.1.",
            ),
            (
                {"mask_threshold": 1.1},
                "mask_threshold must be between 0.0 and 1.0 inclusive, received 1.1.",
            ),
            (
                {"mask_surrogate_scale": -0.1},
                "mask_surrogate_scale must be greater than or equal to 0.0, "
                "received -0.1.",
            ),
            (
                {"mask_floor": -0.1},
                "mask_floor must be between 0.0 inclusive and 1.0 exclusive, "
                "received -0.1.",
            ),
            (
                {"mask_floor": 1.0},
                "mask_floor must be between 0.0 inclusive and 1.0 exclusive, "
                "received 1.0.",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                config = PerAxisScoreMaskConfig(**(base | overrides))
                self.assert_exact_error(ValueError, message, config.build)

        inclusive_threshold = PerAxisScoreMaskConfig(
            **(base | {"mask_threshold": 1.0})
        ).build()
        self.assertEqual(inclusive_threshold.mask_threshold, 1.0)

        for width in (0.0, -0.1):
            with self.subTest(mask_transition_width=width):
                config = TopSliceAxisMaskConfig(
                    **base,
                    mask_transition_width=width,
                )
                self.assert_exact_error(
                    ValueError,
                    "mask_transition_width must be greater than 0.0, "
                    f"received {width!r}.",
                    config.build,
                )


if __name__ == "__main__":
    unittest.main()
