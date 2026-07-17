import unittest

import torch
from torch import nn

from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveGeneratorValidatorBase,
)
from emperor.layers import Layer


class TestAdaptiveGeneratorValidatorBase(unittest.TestCase):
    def test_generator_model_orchestration_dispatches_through_subclass(self):
        validated_layers = []

        class TrackingValidator(AdaptiveGeneratorValidatorBase):
            @staticmethod
            def _validate_generator_layer(generator_layer):
                validated_layers.append(generator_layer)

        layer = Layer.__new__(Layer)
        torch.nn.Module.__init__(layer)

        TrackingValidator.validate_generator_model(layer)

        self.assertEqual(validated_layers, [layer])

    def test_batched_weight_orchestration_dispatches_through_subclass(self):
        validated_weights = []

        class TrackingValidator(AdaptiveGeneratorValidatorBase):
            @staticmethod
            def validate_weight_params(model, weight_params):
                validated_weights.append(weight_params)

        weight_params = torch.ones(2, 3)
        TrackingValidator.validate_batched_weight_params(
            object(), weight_params, torch.ones(4, 2)
        )

        self.assertEqual(validated_weights, [weight_params])

    def test_invalid_generator_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            TypeError,
            r"Expected model_config.build\(\.\.\.\) to return a Layer, "
            r"Sequential, or LayerStack, received object",
        ):
            AdaptiveGeneratorValidatorBase.validate_generator_model(object())

    def test_generator_sequence_rejects_items_that_are_not_layers(self):
        sequence = nn.Sequential(nn.Identity())

        with self.assertRaisesRegex(
            TypeError,
            "^Expected each generator sequence item to be a Layer, "
            "received Identity\\.$",
        ):
            AdaptiveGeneratorValidatorBase.validate_generator_model(sequence)

    def test_generator_layer_rejects_non_linear_inner_model(self):
        layer = Layer.__new__(Layer)
        nn.Module.__init__(layer)
        layer.model = nn.Identity()

        with self.assertRaisesRegex(
            TypeError,
            "^Expected each generator Layer to wrap a LinearLayer, "
            "received Identity\\.$",
        ):
            AdaptiveGeneratorValidatorBase.validate_generator_model(layer)


if __name__ == "__main__":
    unittest.main()
