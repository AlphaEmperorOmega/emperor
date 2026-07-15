import unittest

from emperor.base.layer._validator import LayerStackValidator
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.layer.stack import LayerStack
from emperor.base.options import LastLayerBiasOptions


def make_config(**overrides) -> LayerStackConfig:
    values = {
        "input_dim": 3,
        "hidden_dim": 3,
        "output_dim": 3,
        "num_layers": 1,
        "apply_output_pipeline_flag": False,
        "last_layer_bias_option": LastLayerBiasOptions.DEFAULT,
        "shared_gate_config": None,
        "shared_halting_config": None,
        "shared_memory_config": None,
        "layer_config": LayerConfig(),
    }
    values.update(overrides)
    return LayerStackConfig(**values)


class TestLayerStackValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(LayerStack.VALIDATOR, LayerStackValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(LayerStackValidator):
            @staticmethod
            def _validate_num_layers(num_layers):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayerStack(LayerStack):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayerStack(make_config())

    def test_dimension_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "num_layers must be greater than 0, received 0",
        ):
            LayerStack(make_config(num_layers=0))


if __name__ == "__main__":
    unittest.main()
