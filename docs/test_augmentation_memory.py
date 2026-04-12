import torch
import unittest

from dataclasses import asdict
from emperor.config import ModelConfig
from emperor.linears.core.presets import LinearPresets
from emperor.augmentations.adaptive_parameters.core.factory import DynamicMemoryFactory
from emperor.augmentations.adaptive_parameters.options import (
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from emperor.augmentations.adaptive_parameters.core.handlers.memory import (
    MemoryFusionHandler,
    MemoryHandlerAbstract,
    WeightedMemoryHandler,
)


class TestLinearsAugmentationMemory(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.num_heads = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = (
            LinearPresets.adaptive_linear_layer_preset(return_model_config_flag=True)
            if config is None
            else config
        )

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim


class TestMemoryFusionHandler(TestLinearsAugmentationMemory):
    def test_forward(self):
        for position_option in LinearMemoryPositionOptions:
            for size_option in LinearMemorySizeOptions:
                message = f"Test failed for memory position option: {position_option} and memory size option: {size_option}."
                with self.subTest(message=message):
                    dim = self.cfg.output_dim
                    if position_option == LinearMemoryPositionOptions.BEFORE_AFFINE:
                        dim = self.cfg.input_dim
                    cfg = LinearPresets.adaptive_linear_layer_preset(
                        memory_position_option=position_option,
                        memory_size_option=size_option,
                    )
                    cfg = cfg.override_config
                    if size_option == LinearMemorySizeOptions.DISABLED:
                        with self.assertRaises(ValueError):
                            model = MemoryFusionHandler(cfg)
                    else:
                        input_tensor = torch.ones(self.batch_size, dim)
                        model = MemoryFusionHandler(cfg)
                        output = model(input_tensor)
                        expected_weight_shape = (self.batch_size, dim)
                        self.assertEqual(output.shape, expected_weight_shape)
                        self.assertIsInstance(output, torch.Tensor)


class TestWeightedMemoryHandler(TestLinearsAugmentationMemory):
    def test_forward(self):
        for size_option in LinearMemorySizeOptions:
            for position_option in LinearMemoryPositionOptions:
                message = f"Test failed for memory position option: {position_option}."
                with self.subTest(message=message):
                    dim = self.cfg.output_dim
                    if position_option == LinearMemoryPositionOptions.BEFORE_AFFINE:
                        dim = self.cfg.input_dim
                    cfg = LinearPresets.adaptive_linear_layer_preset(
                        input_dim=dim,
                        memory_size_option=size_option,
                        memory_position_option=position_option,
                    )
                    cfg = cfg.override_config
                    if size_option == LinearMemorySizeOptions.DISABLED:
                        with self.assertRaises(ValueError):
                            model = WeightedMemoryHandler(cfg)
                    else:
                        input_tensor = torch.randn(self.batch_size, dim)
                        model = WeightedMemoryHandler(cfg)
                        output = model(input_tensor)
                        expected_weight_shape = (self.batch_size, dim)
                        self.assertEqual(output.shape, expected_weight_shape)
                        self.assertIsInstance(output, torch.Tensor)


class TestDynamicMemoryFactory(TestLinearsAugmentationMemory):
    def test_build(self):
        for memory_option in LinearMemoryOptions:
            for position_option in LinearMemoryPositionOptions:
                for size_option in LinearMemorySizeOptions:
                    message = f"Test failed for `memory option`: {memory_option}, `position option`: {position_option}, and `size option`: {size_option}."
                    with self.subTest(message):
                        dim = self.cfg.output_dim
                        if position_option == LinearMemoryPositionOptions.BEFORE_AFFINE:
                            dim = self.cfg.input_dim
                        cfg = LinearPresets.adaptive_linear_layer_preset(
                            batch_size=2,
                            input_dim=dim,
                            memory_option=memory_option,
                            memory_position_option=position_option,
                            memory_size_option=size_option,
                        )
                        cfg = cfg.override_config
                        if (
                            memory_option == LinearMemoryOptions.DISABLED
                            or size_option == LinearMemorySizeOptions.DISABLED
                        ):
                            with self.assertRaises(ValueError):
                                DynamicMemoryFactory(cfg).build()
                        else:
                            input_tensor = torch.randn(self.batch_size, dim)
                            handler = DynamicMemoryFactory(cfg).build()
                            output = handler(input_tensor)
                            expected_weight_shape = (self.batch_size, dim)
                            self.assertEqual(output.shape, expected_weight_shape)
                            self.assertIsInstance(handler, MemoryHandlerAbstract)
                            self.assertIsInstance(output, torch.Tensor)
