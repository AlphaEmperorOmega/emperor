import unittest

import torch

from emperor.config import ConfigBase
from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import RecurrentLayerConfig, RowLayout
from emperor.transformer import FeedForward, FeedForwardConfig
from emperor.transformer._validation import FeedForwardValidator
from support.layers import linear_stack_config


class TestFeedForwardValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(FeedForward.VALIDATOR, FeedForwardValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(FeedForwardValidator):
            @staticmethod
            def _validate_stack_config_type(stack_config):
                raise RuntimeError("substituted construction validator was called")

        class TrackingFeedForward(FeedForward):
            VALIDATOR = TrackingValidator

        cfg = FeedForwardConfig(
            input_dim=2,
            output_dim=3,
            stack_config=ConfigBase(),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingFeedForward(cfg)

    def test_stack_config_type_error_contract_is_preserved(self):
        cfg = FeedForwardConfig(
            input_dim=2,
            output_dim=3,
            stack_config=ConfigBase(),
        )

        with self.assertRaisesRegex(
            TypeError,
            "FeedForward.stack_config must be a LayerStackConfig, "
            "MixtureOfExpertsModelConfig, or RecurrentLayerConfig, got ConfigBase",
        ):
            FeedForward(cfg)

    def test_mixture_of_experts_nested_stack_is_validated_before_construction(self):
        for invalid_stack_config in (None, ConfigBase()):
            with self.subTest(type=type(invalid_stack_config).__name__):
                cfg = FeedForwardConfig(
                    input_dim=2,
                    output_dim=2,
                    stack_config=MixtureOfExpertsModelConfig(
                        stack_config=invalid_stack_config
                    ),
                )

                with self.assertRaisesRegex(
                    TypeError,
                    "FeedForward cannot mirror stack_config of type "
                    f"{type(invalid_stack_config).__name__}",
                ):
                    FeedForward(cfg)

    def test_recurrent_nested_block_is_validated_before_construction(self):
        for invalid_block_config in (None, ConfigBase()):
            with self.subTest(type=type(invalid_block_config).__name__):
                cfg = FeedForwardConfig(
                    input_dim=2,
                    output_dim=2,
                    stack_config=RecurrentLayerConfig(
                        block_config=invalid_block_config
                    ),
                )

                with self.assertRaisesRegex(
                    TypeError,
                    "FeedForward cannot mirror stack_config of type "
                    f"{type(invalid_block_config).__name__}",
                ):
                    FeedForward(cfg)

    def test_forward_dispatches_through_substituted_validator(self):
        class TrackingValidator(FeedForwardValidator):
            @staticmethod
            def validate_forward_inputs(flattened_input, row_layout):
                raise RuntimeError("substituted runtime validator was called")

        class TrackingFeedForward(FeedForward):
            VALIDATOR = TrackingValidator

        model = TrackingFeedForward(
            FeedForwardConfig(
                input_dim=2,
                output_dim=2,
                stack_config=linear_stack_config(2),
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(torch.randn(3, 2))

    def test_row_layout_row_count_error_contract_is_preserved(self):
        model = FeedForward(
            FeedForwardConfig(
                input_dim=2,
                output_dim=2,
                stack_config=linear_stack_config(2),
            )
        )
        row_layout = RowLayout.rows(
            2,
            context_sharing_restricted=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "row_layout row_count=2 does not match feed-forward row count 3",
        ):
            model(torch.randn(3, 2), row_layout=row_layout)


if __name__ == "__main__":
    unittest.main()
