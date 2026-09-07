import unittest

import torch

from emperor.config import ConfigBase
from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import (
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentLayerConfig,
)
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
            "MixtureOfExpertsModelConfig, or RecurrentCompositionConfig, got "
            "ConfigBase",
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

        hierarchical_reasoning_model_config = HierarchicalReasoningModelRecurrentConfig(
            high_block_config=None,
            low_block_config=linear_stack_config(2),
        )
        with self.assertRaisesRegex(
            TypeError,
            "FeedForward cannot mirror stack_config of type NoneType",
        ):
            FeedForward(
                FeedForwardConfig(
                    input_dim=2,
                    output_dim=2,
                    stack_config=hierarchical_reasoning_model_config,
                )
            )

    def test_forward_dispatches_through_substituted_validator(self):
        class TrackingValidator(FeedForwardValidator):
            @staticmethod
            def validate_forward_inputs(model, input_batch):
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

    def test_grouped_input_checks_actual_sequence_length_before_flattening(self):
        from emperor.augmentations.adaptive_parameters import (
            AdaptiveParameterGroupingScopeOptions,
        )
        from support.adaptive_grouping import bias_linear, grouping_value

        stack = linear_stack_config(2)
        stack.layer_config.layer_model_config = bias_linear(
            grouping_value(
                AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                2,
                sequence_length=4,
            )
        ).cfg
        model = FeedForward(
            FeedForwardConfig(input_dim=2, output_dim=2, stack_config=stack)
        )
        with self.assertRaisesRegex(ValueError, "actual sequence length"):
            model(torch.randn(4, 2, 2))


if __name__ == "__main__":
    unittest.main()
