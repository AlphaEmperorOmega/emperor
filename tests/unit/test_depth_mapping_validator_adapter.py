import unittest

import torch

from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingLayer,
    DepthMappingLayerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    DepthMappingValidator,
)


class TestDepthMappingValidatorAdapter(unittest.TestCase):
    def test_depth_mapping_modules_declare_the_shared_validator_adapter(self):
        module_types = (DepthMappingLayer, DepthMappingLayerStack)

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, DepthMappingValidator)

    def test_layer_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(DepthMappingValidator):
            @staticmethod
            def validate_required_fields(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayer(DepthMappingLayer):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayer(DepthMappingLayerConfig())

    def test_stack_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(DepthMappingValidator):
            @staticmethod
            def validate_input_is_2d(input_batch, input_dim=None):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingStack(DepthMappingLayerStack):
            VALIDATOR = RejectingValidator

        model = RejectingStack.__new__(RejectingStack)
        torch.nn.Module.__init__(model)
        model.input_dim = 3

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()
