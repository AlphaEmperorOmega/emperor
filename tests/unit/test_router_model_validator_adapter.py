import unittest
from unittest.mock import patch

import torch

from emperor.config import ConfigBase
from emperor.sampler import RouterConfig, RouterModel
from emperor.sampler._validation import RouterModelValidator


def make_config(**overrides) -> RouterConfig:
    values = {
        "input_dim": 3,
        "num_experts": 4,
        "noisy_topk_flag": False,
        "model_config": ConfigBase(),
    }
    values.update(overrides)
    return RouterConfig(**values)


class TestRouterModelValidatorAdapter(unittest.TestCase):
    def test_custom_builder_is_not_executed_speculatively(self):
        config = make_config()
        with patch.object(
            ConfigBase,
            "build",
            side_effect=RuntimeError("custom construction validation"),
        ) as build:
            RouterModelValidator.validate_config(config)
            build.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "custom construction validation"):
                RouterModel(config)
            build.assert_called_once()

    def test_module_exposes_validator_adapter(self):
        self.assertIs(RouterModel.VALIDATOR, RouterModelValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(RouterModelValidator):
            @staticmethod
            def _validate_model_config(model_config):
                raise RuntimeError("substituted construction validator was called")

        class TrackingRouterModel(RouterModel):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingRouterModel(make_config())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(RouterModelValidator):
            @staticmethod
            def validate_forward_inputs(model, input_batch):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingRouterModel(RouterModel):
            VALIDATOR = RejectingValidator

        model = RejectingRouterModel.__new__(RejectingRouterModel)
        torch.nn.Module.__init__(model)
        model.input_dim = 3

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model.compute_logit_scores(torch.ones(1, 3))

    def test_boolean_dimension_type_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            TypeError,
            "input_dim must be int for RouterConfig, got bool",
        ):
            RouterModel(make_config(input_dim=True))


if __name__ == "__main__":
    unittest.main()
