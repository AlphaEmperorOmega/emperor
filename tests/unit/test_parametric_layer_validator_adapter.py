import unittest

import torch

from emperor.parametric import (
    AdaptiveRouterOptions,
    ParametricLayer,
    ParametricLayerConfig,
)
from emperor.parametric._validation import ParametricLayerValidator


def make_config(**overrides) -> ParametricLayerConfig:
    values = {
        "input_dim": 3,
        "output_dim": 4,
        "weight_mixture_config": object(),
        "bias_mixture_config": None,
        "routing_initialization_mode": AdaptiveRouterOptions.INDEPENDENT_ROUTER,
        "router_config": object(),
        "sampler_config": object(),
        "adaptive_augmentation_config": object(),
    }
    values.update(overrides)
    return ParametricLayerConfig(**values)


class TestParametricLayerValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(ParametricLayer.VALIDATOR, ParametricLayerValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(ParametricLayerValidator):
            @staticmethod
            def _validate_weight_mixture_config(config):
                raise RuntimeError("substituted construction validator was called")

        class TrackingParametricLayer(ParametricLayer):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingParametricLayer(make_config())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(ParametricLayerValidator):
            @staticmethod
            def validate_forward_inputs(input_batch, expected_input_dim):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingParametricLayer(ParametricLayer):
            VALIDATOR = RejectingValidator

        model = RejectingParametricLayer.__new__(RejectingParametricLayer)
        torch.nn.Module.__init__(model)
        model.input_dim = 3

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model(torch.ones(1, 3))

    def test_weight_mixture_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            TypeError,
            "weight_mixture_config must be a weight mixture config, got object",
        ):
            ParametricLayer(make_config())


if __name__ == "__main__":
    unittest.main()
