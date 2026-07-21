import unittest

from emperor.config import ConfigBase
from emperor.transformer import FeedForward, FeedForwardConfig
from emperor.transformer._validation import FeedForwardValidator


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


if __name__ == "__main__":
    unittest.main()
