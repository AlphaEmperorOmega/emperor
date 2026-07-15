import unittest

import torch
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core._validator import (
    AdaptiveParameterAugmentationValidator,
)
from emperor.augmentations.adaptive_parameters.model import (
    AdaptiveParameterAugmentation,
)


class TestAdaptiveParameterAugmentationValidatorAdapter(unittest.TestCase):
    def test_module_declares_its_validator_adapter(self):
        self.assertIs(
            AdaptiveParameterAugmentation.VALIDATOR,
            AdaptiveParameterAugmentationValidator,
        )

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(AdaptiveParameterAugmentationValidator):
            @staticmethod
            def _validate_dimensions(model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingAugmentation(AdaptiveParameterAugmentation):
            VALIDATOR = TrackingValidator

        cfg = AdaptiveParameterAugmentationConfig(input_dim=3, output_dim=4)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingAugmentation(cfg)

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(AdaptiveParameterAugmentationValidator):
            @staticmethod
            def validate_input_batch(model, input_batch):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingAugmentation(AdaptiveParameterAugmentation):
            VALIDATOR = RejectingValidator

        model = RejectingAugmentation.__new__(RejectingAugmentation)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(
                lambda weights, bias, input_batch: input_batch,
                torch.ones(3, 4),
                None,
                torch.ones(1, 3),
            )


if __name__ == "__main__":
    unittest.main()
