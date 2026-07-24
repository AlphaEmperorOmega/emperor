import unittest

from emperor.layers import (
    ActivationOptions,
    AttentionResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.layers._validation import LayerStackValidator
from emperor.linears import LinearLayerConfig


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


def attention_residual_layer_config() -> LayerConfig:
    return LayerConfig(
        activation=ActivationOptions.DISABLED,
        residual_config=ResidualConfig(
            option=ResidualConnectionOptions.ATTENTION_RESIDUAL,
            attention_config=AttentionResidualConfig(
                block_size=1,
                rms_norm_epsilon=1e-6,
            ),
        ),
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=LinearLayerConfig(bias_flag=True),
    )


class TestLayerStackValidatorAdapter(unittest.TestCase):
    def test_attention_residual_requires_the_final_output_pipeline(self):
        with self.assertRaisesRegex(
            ValueError,
            "apply_output_pipeline_flag must be True",
        ):
            LayerStack(
                make_config(
                    num_layers=2,
                    apply_output_pipeline_flag=False,
                    layer_config=attention_residual_layer_config(),
                )
            )

    def test_attention_residual_requires_one_stable_stack_dimension(self):
        with self.assertRaisesRegex(
            ValueError,
            "input_dim, hidden_dim, and output_dim must all be equal",
        ):
            LayerStack(
                make_config(
                    input_dim=2,
                    hidden_dim=3,
                    output_dim=3,
                    apply_output_pipeline_flag=True,
                    layer_config=attention_residual_layer_config(),
                )
            )

    def test_module_exposes_validator_adapter(self):
        self.assertIs(LayerStack.VALIDATOR, LayerStackValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(LayerStackValidator):
            @classmethod
            def _validate_gate_config(cls, cfg):
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
