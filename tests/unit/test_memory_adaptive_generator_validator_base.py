import unittest

import torch
from emperor.base.layer.layer import Layer
from emperor.memory.core._validator import AdaptiveGeneratorValidatorBase


class TestMemoryAdaptiveGeneratorValidatorBase(unittest.TestCase):
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

    def test_test_time_training_orchestration_dispatches_through_subclass(self):
        validated_models = []

        class TrackingValidator(AdaptiveGeneratorValidatorBase):
            @classmethod
            def validate_generator_model(cls, generator_model):
                validated_models.append(generator_model)

        generator_model = torch.nn.Linear(2, 2)

        TrackingValidator.validate_test_time_training_generator_model(generator_model)

        self.assertEqual(validated_models, [generator_model])

    def test_invalid_generator_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            TypeError,
            r"Expected model_config.build\(\.\.\.\) to return a Layer, "
            r"Sequential, or LayerStack, received object",
        ):
            AdaptiveGeneratorValidatorBase.validate_generator_model(object())


if __name__ == "__main__":
    unittest.main()
