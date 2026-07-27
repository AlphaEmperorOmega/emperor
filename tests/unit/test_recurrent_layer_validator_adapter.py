import unittest

import torch

from emperor.config import ConfigBase
from emperor.layers import (
    LayerNormPositionOptions,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
)
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
    RecurrentLayerValidator,
    TinyRecursiveModelRecurrentValidator,
)


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

    def test_halting_output_rejects_shape_dtype_and_device_drift(self):
        candidate_hidden = torch.ones(2, 3)
        cases = (
            (torch.ones(1, 3), "preserve the candidate hidden shape"),
            (
                torch.ones(2, 3, dtype=torch.float64),
                "preserve the candidate hidden dtype",
            ),
            (
                torch.empty(2, 3, device="meta"),
                "preserve the candidate hidden device",
            ),
        )

        for output_hidden, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    RecurrentLayerValidator.validate_halting_output(
                        output_hidden,
                        candidate_hidden,
                    )

    def test_transition_output_rejects_type_dtype_and_device_drift(self):
        transition_input = torch.ones(2, 3)
        cases = (
            (object(), TypeError, "must return LayerState"),
            (
                LayerState(hidden=torch.ones(2, 3, dtype=torch.float64)),
                ValueError,
                "preserve hidden dtype",
            ),
            (
                LayerState(hidden=torch.empty(2, 3, device="meta")),
                ValueError,
                "preserve hidden device",
            ),
        )

        for output_state, exception, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception, message):
                    RecurrentLayerValidator.validate_transition_output(
                        output_state,
                        transition_input,
                        None,
                        expected_feature_dim=3,
                    )

    def test_variant_hidden_validation_wrappers_preserve_owner_context(self):
        cases = (
            (
                TinyRecursiveModelRecurrentValidator,
                "TinyRecursiveModelRecurrent",
            ),
            (
                HierarchicalReasoningModelRecurrentValidator,
                "HierarchicalReasoningModelRecurrent",
            ),
        )

        for validator, owner_name in cases:
            with self.subTest(owner_name=owner_name):
                with self.assertRaisesRegex(
                    ValueError,
                    f"last dimension must be 3 for {owner_name}",
                ):
                    validator.validate_hidden(
                        torch.ones(2, 4),
                        3,
                        field_name="hidden",
                    )


if __name__ == "__main__":
    unittest.main()
