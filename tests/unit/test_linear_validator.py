import unittest

import torch
from emperor.linears.core._validator import LinearValidator
from emperor.linears.core.config import LinearLayerConfig
from emperor.linears.core.layers import (
    AdaptiveLinearLayer,
    LinearAbstract,
    LinearLayer,
)


class TestLinearValidatorAdapter(unittest.TestCase):
    def test_linear_modules_share_the_abstract_owner_adapter(self):
        for module_type in (LinearAbstract, LinearLayer, AdaptiveLinearLayer):
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, LinearValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        validated_models = []

        class TrackingLinearValidator(LinearValidator):
            @staticmethod
            def _validate_adaptive_bias_consistency(model):
                validated_models.append(model)

        class TrackingLinearLayer(LinearLayer):
            VALIDATOR = TrackingLinearValidator

        model = TrackingLinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=3, bias_flag=True)
        )

        self.assertEqual(validated_models, [model])

    def test_forward_dispatches_through_substituted_validator(self):
        class RejectingLinearValidator(LinearValidator):
            @staticmethod
            def validate_input_tensor(X, input_dim):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingLinearLayer(LinearLayer):
            VALIDATOR = RejectingLinearValidator

        model = RejectingLinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=3, bias_flag=True)
        )

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model(torch.randn(1, 2))

    def test_plain_linear_runtime_error_contract_is_preserved(self):
        model = LinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=3, bias_flag=True)
        )

        with self.assertRaisesRegex(
            ValueError,
            "Input final dimension must be 2, got 4",
        ):
            model(torch.randn(1, 4))


if __name__ == "__main__":
    unittest.main()
