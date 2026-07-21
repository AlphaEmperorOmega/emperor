import unittest

import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveLinearValidator,
)
from emperor.linears import (
    LinearAbstract,
    LinearLayer,
    LinearLayerConfig,
)
from emperor.linears._validation import LinearValidator


class TestLinearValidatorAdapter(unittest.TestCase):
    def test_generic_linear_modules_share_the_abstract_owner_adapter(self):
        for module_type in (LinearAbstract, LinearLayer):
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, LinearValidator)

    def test_adaptive_linear_uses_the_adaptive_owner_adapter(self):
        self.assertIs(AdaptiveLinearLayer.VALIDATOR, AdaptiveLinearValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        validated_models = []

        class TrackingLinearValidator(LinearValidator):
            @classmethod
            def validate(cls, model):
                validated_models.append(model)
                super().validate(model)

        class TrackingLinearLayer(LinearLayer):
            VALIDATOR = TrackingLinearValidator

        model = TrackingLinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=3, bias_flag=True)
        )

        self.assertEqual(validated_models, [model])

    def test_adaptive_construction_dispatches_through_substituted_validator(self):
        validated_models = []

        class TrackingAdaptiveLinearValidator(AdaptiveLinearValidator):
            @classmethod
            def validate(cls, model):
                validated_models.append(model)
                super().validate(model)

        class TrackingAdaptiveLinearLayer(AdaptiveLinearLayer):
            VALIDATOR = TrackingAdaptiveLinearValidator

        model = TrackingAdaptiveLinearLayer(
            AdaptiveLinearLayerConfig(
                input_dim=2,
                output_dim=3,
                bias_flag=True,
                adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(),
            )
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

    def test_runtime_errors_report_the_actual_rank_and_final_dimension(self):
        model = LinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=3, bias_flag=True)
        )

        with self.assertRaises(ValueError) as rank_error:
            model(torch.tensor([1.0, 2.0]))
        self.assertEqual(
            str(rank_error.exception),
            "Input must have shape (..., input_dim), got 1D tensor with shape "
            "torch.Size([2])",
        )

        with self.assertRaises(ValueError) as dimension_error:
            model(torch.ones(2, 3, 4))
        self.assertEqual(
            str(dimension_error.exception),
            "Input final dimension must be 2, got 4 for tensor with shape "
            "torch.Size([2, 3, 4])",
        )

    def test_runtime_rejects_non_tensor_input_with_a_clear_type_error(self):
        model = LinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=3, bias_flag=True)
        )

        with self.assertRaises(TypeError) as raised:
            model([[1.0, 2.0]])

        self.assertEqual(
            str(raised.exception),
            "Input must be a Tensor, got list",
        )


if __name__ == "__main__":
    unittest.main()
