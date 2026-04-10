import torch
import unittest
import torch.nn as nn

from emperor.base.utils import Module
from emperor.config import ModelConfig
from emperor.linears.core.presets import LinearPresets
from emperor.augmentations.adaptive_parameters.model import AdaptiveParameterAugmentation
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    DynamicWeightOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from emperor.augmentations.adaptive_parameters.utils.handlers.weight import WeightHandlerAbstract
from emperor.augmentations.adaptive_parameters.utils.handlers.diagonal import DiagonalHandlerAbstract
from emperor.augmentations.adaptive_parameters.utils.handlers.bias import BiasHandlerAbstract
from emperor.augmentations.adaptive_parameters.utils.handlers.memory import MemoryHandlerAbstract


class TestAdaptiveParameterAugmentation(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.config = None
        self.model = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.config = (
            LinearPresets.adaptive_linear_layer_preset(return_model_config_flag=True)
            if config is None
            else config
        )
        self.batch_size = self.config.batch_size
        self.input_dim = self.config.input_dim
        self.output_dim = self.config.output_dim

    def _make_affine_callback(self):
        def callback(weights, bias, X):
            if weights.dim() == 3:
                output = torch.einsum("ij,ijk->ik", X, weights)
            else:
                output = torch.matmul(X, weights)
            if bias is not None:
                output = output + bias
            return output

        return callback

    def _make_weight_and_bias_params(self):
        weight_shape = (self.input_dim, self.output_dim)
        weight_params = Module()._init_parameter_bank(weight_shape)
        bias_shape = (self.output_dim,)
        bias_params = Module()._init_parameter_bank(bias_shape, nn.init.zeros_)
        return weight_params, bias_params

    def test_init_all_disabled(self):
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.override_config
        model = AdaptiveParameterAugmentation(cfg)
        self.assertIsNone(model.generator_model)
        self.assertIsNone(model.diagonal_model)
        self.assertIsNone(model.memory_model)
        self.assertIsNone(model.bias_model)

    def test_init_with_sub_models_enabled(self):
        generator_cases = [
            (
                "generator_model",
                WeightHandlerAbstract,
                {"generator_depth": depth},
            )
            for depth in DynamicDepthOptions
            if depth != DynamicDepthOptions.DISABLED
        ]
        diagonal_cases = [
            (
                "diagonal_model",
                DiagonalHandlerAbstract,
                {"diagonal_option": option},
            )
            for option in DynamicDiagonalOptions
            if option != DynamicDiagonalOptions.DISABLED
        ]
        bias_cases = [
            (
                "bias_model",
                BiasHandlerAbstract,
                {"bias_option": option},
            )
            for option in DynamicBiasOptions
            if option != DynamicBiasOptions.DISABLED
        ]
        memory_cases = [
            (
                "memory_model",
                MemoryHandlerAbstract,
                {
                    "memory_option": option,
                    "memory_size_option": LinearMemorySizeOptions.SMALL,
                    "memory_position_option": LinearMemoryPositionOptions.BEFORE_AFFINE,
                },
            )
            for option in LinearMemoryOptions
            if option != LinearMemoryOptions.DISABLED
        ]
        cases = generator_cases + diagonal_cases + bias_cases + memory_cases
        for attr, expected_type, preset_kwargs in cases:
            message = f"{attr} with {preset_kwargs}"
            with self.subTest(message):
                cfg = LinearPresets.adaptive_linear_layer_preset(**preset_kwargs)
                cfg = cfg.override_config
                model = AdaptiveParameterAugmentation(cfg)
                sub_model = getattr(model, attr)
                self.assertIsNotNone(sub_model)
                self.assertIsInstance(sub_model, expected_type)

    def test_forward_all_disabled(self):
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.override_config
        model = AdaptiveParameterAugmentation(cfg)

        weight_params, bias_params = self._make_weight_and_bias_params()
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        callback = self._make_affine_callback()

        output = model.compute_adaptive_parameters(
            callback, weight_params, bias_params, input_tensor
        )
        expected = callback(weight_params, bias_params, input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        self.assertTrue(
            torch.allclose(
                output.round(decimals=6),
                expected.round(decimals=6),
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_forward_all_disabled_without_bias(self):
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.override_config
        model = AdaptiveParameterAugmentation(cfg)

        weight_shape = (self.input_dim, self.output_dim)
        weight_params = Module()._init_parameter_bank(weight_shape)
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        callback = self._make_affine_callback()

        output = model.compute_adaptive_parameters(
            callback, weight_params, None, input_tensor
        )
        expected = callback(weight_params, None, input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        self.assertTrue(
            torch.allclose(
                output.round(decimals=6),
                expected.round(decimals=6),
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_forward_with_generator(self):
        for depth in DynamicDepthOptions:
            if depth == DynamicDepthOptions.DISABLED:
                continue
            with self.subTest(f"generator_depth={depth}"):
                cfg = LinearPresets.adaptive_linear_layer_preset(
                    generator_depth=depth,
                )
                cfg = cfg.override_config
                model = AdaptiveParameterAugmentation(cfg)

                weight_params, bias_params = self._make_weight_and_bias_params()
                input_tensor = torch.randn(self.batch_size, self.input_dim)
                callback = self._make_affine_callback()

                output = model.compute_adaptive_parameters(
                    callback, weight_params, bias_params, input_tensor
                )
                expected_shape = (self.batch_size, self.output_dim)
                self.assertEqual(output.shape, expected_shape)
                self.assertIsInstance(output, torch.Tensor)

    def test_forward_with_weight_option(self):
        for option in DynamicWeightOptions:
            if option == DynamicWeightOptions.DISABLED:
                continue
            with self.subTest(f"weight_option={option}"):
                cfg = LinearPresets.adaptive_linear_layer_preset(
                    weight_option=option,
                )
                cfg = cfg.override_config
                model = AdaptiveParameterAugmentation(cfg)

                weight_params, bias_params = self._make_weight_and_bias_params()
                input_tensor = torch.randn(self.batch_size, self.input_dim)
                callback = self._make_affine_callback()

                output = model.compute_adaptive_parameters(
                    callback, weight_params, bias_params, input_tensor
                )
                expected_shape = (self.batch_size, self.output_dim)
                self.assertEqual(output.shape, expected_shape)
                self.assertIsInstance(output, torch.Tensor)

    def test_forward_with_diagonal(self):
        for option in DynamicDiagonalOptions:
            if option == DynamicDiagonalOptions.DISABLED:
                continue
            with self.subTest(f"diagonal_option={option}"):
                cfg = LinearPresets.adaptive_linear_layer_preset(
                    diagonal_option=option,
                )
                cfg = cfg.override_config
                model = AdaptiveParameterAugmentation(cfg)

                weight_params, bias_params = self._make_weight_and_bias_params()
                input_tensor = torch.randn(self.batch_size, self.input_dim)
                callback = self._make_affine_callback()

                output = model.compute_adaptive_parameters(
                    callback, weight_params, bias_params, input_tensor
                )
                expected_shape = (self.batch_size, self.output_dim)
                self.assertEqual(output.shape, expected_shape)
                self.assertIsInstance(output, torch.Tensor)

    def test_forward_with_bias(self):
        for option in DynamicBiasOptions:
            if option == DynamicBiasOptions.DISABLED:
                continue
            with self.subTest(f"bias_option={option}"):
                cfg = LinearPresets.adaptive_linear_layer_preset(
                    bias_option=option,
                )
                cfg = cfg.override_config
                model = AdaptiveParameterAugmentation(cfg)

                weight_params, bias_params = self._make_weight_and_bias_params()
                input_tensor = torch.randn(self.batch_size, self.input_dim)
                callback = self._make_affine_callback()

                output = model.compute_adaptive_parameters(
                    callback, weight_params, bias_params, input_tensor
                )
                expected_shape = (self.batch_size, self.output_dim)
                self.assertEqual(output.shape, expected_shape)
                self.assertIsInstance(output, torch.Tensor)

    def test_forward_with_memory_before_affine(self):
        for memory_option in LinearMemoryOptions:
            if memory_option == LinearMemoryOptions.DISABLED:
                continue
            for size_option in LinearMemorySizeOptions:
                if size_option == LinearMemorySizeOptions.DISABLED:
                    continue
                message = f"memory_option={memory_option}, size_option={size_option}"
                with self.subTest(message):
                    cfg = LinearPresets.adaptive_linear_layer_preset(
                        memory_option=memory_option,
                        memory_size_option=size_option,
                        memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
                    )
                    cfg = cfg.override_config
                    model = AdaptiveParameterAugmentation(cfg)

                    weight_params, bias_params = self._make_weight_and_bias_params()
                    input_tensor = torch.randn(self.batch_size, self.input_dim)
                    callback = self._make_affine_callback()

                    output = model.compute_adaptive_parameters(
                        callback, weight_params, bias_params, input_tensor
                    )
                    expected_shape = (self.batch_size, self.output_dim)
                    self.assertEqual(output.shape, expected_shape)
                    self.assertIsInstance(output, torch.Tensor)

    def test_forward_with_memory_after_affine(self):
        for memory_option in LinearMemoryOptions:
            if memory_option == LinearMemoryOptions.DISABLED:
                continue
            for size_option in LinearMemorySizeOptions:
                if size_option == LinearMemorySizeOptions.DISABLED:
                    continue
                message = f"memory_option={memory_option}, size_option={size_option}"
                with self.subTest(message):
                    cfg = LinearPresets.adaptive_linear_layer_preset(
                        memory_option=memory_option,
                        memory_size_option=size_option,
                        memory_position_option=LinearMemoryPositionOptions.AFTER_AFFINE,
                    )
                    cfg = cfg.override_config
                    model = AdaptiveParameterAugmentation(cfg)

                    weight_params, bias_params = self._make_weight_and_bias_params()
                    input_tensor = torch.randn(self.batch_size, self.input_dim)
                    callback = self._make_affine_callback()

                    output = model.compute_adaptive_parameters(
                        callback, weight_params, bias_params, input_tensor
                    )
                    self.assertEqual(output.shape, (self.batch_size, self.output_dim))
                    self.assertIsInstance(output, torch.Tensor)

    def test_forward_full_pipeline(self):
        for depth in DynamicDepthOptions:
            if depth == DynamicDepthOptions.DISABLED:
                continue
            for diagonal in DynamicDiagonalOptions:
                if diagonal == DynamicDiagonalOptions.DISABLED:
                    continue
                for bias in DynamicBiasOptions:
                    if bias == DynamicBiasOptions.DISABLED:
                        continue
                    for memory_option in LinearMemoryOptions:
                        for position_option in LinearMemoryPositionOptions:
                            memory_kwargs = {}
                            if memory_option != LinearMemoryOptions.DISABLED:
                                memory_kwargs = {
                                    "memory_option": memory_option,
                                    "memory_size_option": LinearMemorySizeOptions.SMALL,
                                    "memory_position_option": position_option,
                                }
                            message = (
                                f"depth={depth}, diagonal={diagonal}, "
                                f"bias={bias}, memory={memory_option}, "
                                f"position={position_option}"
                            )
                            with self.subTest(message):
                                cfg = LinearPresets.adaptive_linear_layer_preset(
                                    generator_depth=depth,
                                    diagonal_option=diagonal,
                                    bias_option=bias,
                                    **memory_kwargs,
                                )
                                cfg = cfg.override_config
                                model = AdaptiveParameterAugmentation(cfg)

                                weight_params, bias_params = (
                                    self._make_weight_and_bias_params()
                                )
                                input_tensor = torch.randn(
                                    self.batch_size, self.input_dim
                                )
                                callback = self._make_affine_callback()

                                output = model.compute_adaptive_parameters(
                                    callback,
                                    weight_params,
                                    bias_params,
                                    input_tensor,
                                )
                                self.assertEqual(
                                    output.shape,
                                    (self.batch_size, self.output_dim),
                                )
                                self.assertIsInstance(output, torch.Tensor)
