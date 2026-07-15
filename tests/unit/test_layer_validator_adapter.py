import unittest

from emperor.base.config import ConfigBase
from emperor.base.layer._validator import LayerValidator
from emperor.base.layer.config import LayerConfig
from emperor.base.layer.layer import Layer
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions


def make_config(**overrides) -> LayerConfig:
    values = {
        "input_dim": 3,
        "output_dim": 3,
        "activation": ActivationOptions.DISABLED,
        "residual_connection_option": ResidualConnectionOptions.DISABLED,
        "dropout_probability": 0.0,
        "layer_norm_position": LayerNormPositionOptions.DISABLED,
        "gate_config": None,
        "halting_config": None,
        "memory_config": None,
        "layer_model_config": ConfigBase(),
    }
    values.update(overrides)
    return LayerConfig(**values)


class TestLayerValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(Layer.VALIDATOR, LayerValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(LayerValidator):
            @staticmethod
            def _validate_dropout_probability(dropout_probability):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayer(Layer):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayer(make_config())

    def test_gate_validation_uses_replaceable_collaborator(self):
        class TrackingGateValidator:
            @classmethod
            def validate_layer_gate_config(cls, gate_config, owner_name=None):
                raise RuntimeError("substituted gate validator was called")

        class TrackingValidator(LayerValidator):
            GATE_VALIDATOR = TrackingGateValidator

        class TrackingLayer(Layer):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted gate validator was called",
        ):
            TrackingLayer(make_config())

    def test_dropout_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "dropout_probability must be between 0.0 and 1.0, received 1.1",
        ):
            Layer(make_config(dropout_probability=1.1))


if __name__ == "__main__":
    unittest.main()
