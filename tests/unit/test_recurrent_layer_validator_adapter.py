import unittest

import torch

from emperor.config import ConfigBase
from emperor.layers import (
    LayerNormPositionOptions,
    LayerState,
    RecurrentLayerConfig,
)
from emperor.layers._recurrent import RecurrentLayer
from emperor.layers._validation import RecurrentLayerValidator


def make_config(**overrides) -> RecurrentLayerConfig:
    values = {
        "input_dim": 3,
        "output_dim": 3,
        "max_steps": 1,
        "recurrent_layer_norm_position": LayerNormPositionOptions.DISABLED,
        "block_config": ConfigBase(),
        "gate_config": None,
        "residual_config": None,
        "halting_config": None,
        "memory_config": None,
    }
    values.update(overrides)
    return RecurrentLayerConfig(**values)


class TestRecurrentLayerValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(RecurrentLayer.VALIDATOR, RecurrentLayerValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(RecurrentLayerValidator):
            @staticmethod
            def _validate_integer_field(field_name, value):
                raise RuntimeError("substituted construction validator was called")

        class TrackingRecurrentLayer(RecurrentLayer):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingRecurrentLayer(make_config())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(RecurrentLayerValidator):
            @classmethod
            def validate_state(cls, state, expected_feature_dim):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingRecurrentLayer(RecurrentLayer):
            VALIDATOR = RejectingValidator

        model = RejectingRecurrentLayer.__new__(RejectingRecurrentLayer)
        torch.nn.Module.__init__(model)
        model.input_dim = 3

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model(LayerState(hidden=torch.ones(1, 3)))

    def test_integer_field_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            TypeError,
            "input_dim must be int for RecurrentLayerConfig, got float",
        ):
            RecurrentLayer(make_config(input_dim=3.0))


if __name__ == "__main__":
    unittest.main()
