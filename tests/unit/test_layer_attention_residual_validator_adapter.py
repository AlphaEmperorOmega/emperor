import unittest

import torch

from emperor.layers._composition.residual.config import AttentionResidualConfig
from emperor.layers._composition.residual.validation import (
    ResidualConnectionValidator,
)
from emperor.layers._composition.residual.variants.attention import (
    AttentionResidual,
    AttentionResidualState,
)


class TestAttentionResidualValidatorAdapter(unittest.TestCase):
    def test_component_and_state_expose_validator_adapter(self):
        self.assertIs(AttentionResidual.VALIDATOR, ResidualConnectionValidator)
        self.assertIs(AttentionResidualState.VALIDATOR, ResidualConnectionValidator)

    def test_successful_validations_are_check_only(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
        initial_source = torch.ones(1, 2)
        current = torch.full((1, 2), 2.0)
        state = residual.new_state(initial_source)
        validator = residual.VALIDATOR

        results = (
            validator.validate_positive_integer(2, name="residual_dim"),
            validator.validate_finite_positive_number(
                1e-6,
                name="rms_norm_epsilon",
            ),
            validator.validate_source(initial_source, residual_dim=2),
            validator.validate_attention_state(state, block_size=1),
            validator.validate_attention_forward_inputs(
                current,
                state,
                residual_dim=2,
                block_size=1,
            ),
        )

        self.assertTupleEqual(results, (None, None, None, None, None))

    def test_construction_dispatches_through_substituted_validator(self):
        class RejectingValidator(ResidualConnectionValidator):
            @staticmethod
            def validate_positive_integer(value, *, name):
                raise RuntimeError("substituted construction validator was called")

        class RejectingAttentionResidual(AttentionResidual):
            VALIDATOR = RejectingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            RejectingAttentionResidual(AttentionResidualConfig(residual_dim=2))

    def test_state_construction_dispatches_through_substituted_validator(self):
        class RejectingValidator(ResidualConnectionValidator):
            @staticmethod
            def validate_positive_integer(value, *, name):
                raise RuntimeError("substituted state validator was called")

        class RejectingState(AttentionResidualState):
            VALIDATOR = RejectingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted state validator was called",
        ):
            RejectingState(torch.ones(1, 2), block_size=1)

    def test_forward_dispatches_through_substituted_validator_before_mutation(self):
        class RejectingValidator(ResidualConnectionValidator):
            @classmethod
            def validate_attention_forward_inputs(
                cls,
                current,
                state,
                *,
                residual_dim,
                block_size,
            ):
                raise RuntimeError("substituted forward validator was called")

        class RejectingAttentionResidual(AttentionResidual):
            VALIDATOR = RejectingValidator

        residual = RejectingAttentionResidual(AttentionResidualConfig(residual_dim=2))
        state = residual.new_state(torch.ones(1, 2))

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted forward validator was called",
        ):
            residual(
                torch.full((1, 2), 2.0),
                torch.full((1, 2), 2.0),
                residual_state=state,
            )

        self.assertEqual(len(state.sources), 1)


if __name__ == "__main__":
    unittest.main()
