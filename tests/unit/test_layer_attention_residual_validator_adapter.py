import unittest

import torch

from emperor.layers import LayerState
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

    def test_successful_validations_preserve_checked_state_identity(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
        initial_source = torch.ones(1, 2)
        current = torch.full((1, 2), 2.0)
        state = residual.new_state(initial_source)
        validator = residual.VALIDATOR

        check_only_results = (
            validator.validate_positive_integer(2, name="residual_dim"),
            validator.validate_finite_positive_number(
                1e-6,
                name="rms_norm_epsilon",
            ),
            validator.validate_source(initial_source, residual_dim=2),
        )

        self.assertTupleEqual(check_only_results, (None, None, None))
        self.assertIs(
            validator.validate_created_attention_state(state, block_size=1),
            state,
        )
        self.assertIs(
            validator.validate_attention_state(state, block_size=1),
            state,
        )
        self.assertIs(
            validator.validate_attention_forward_inputs(
                current,
                state,
                residual_dim=2,
                block_size=1,
            ),
            state,
        )

    def test_created_state_validation_preserves_missing_state_error_contract(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "^AttentionResidual failed to create forward-local residual state\\.$",
        ):
            ResidualConnectionValidator.validate_created_attention_state(
                None,
                block_size=1,
            )

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

    def test_lifecycle_state_creation_uses_the_residual_validator_adapter(self):
        class RejectingValidator(ResidualConnectionValidator):
            @staticmethod
            def validate_source(source, *, residual_dim):
                raise RuntimeError("substituted lifecycle validator was called")

        class RejectingAttentionResidual(AttentionResidual):
            VALIDATOR = RejectingValidator

        residual = RejectingAttentionResidual(AttentionResidualConfig(residual_dim=2))

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted lifecycle validator was called",
        ):
            residual.new_state(torch.ones(1, 2))

    def test_state_application_dispatches_created_state_validation(self):
        class RejectingValidator(ResidualConnectionValidator):
            @classmethod
            def validate_created_attention_state(cls, state, *, block_size):
                raise RuntimeError("substituted created-state validator was called")

        class RejectingAttentionResidual(AttentionResidual):
            VALIDATOR = RejectingValidator

        residual = RejectingAttentionResidual(AttentionResidualConfig(residual_dim=2))
        layer_state = LayerState(hidden=torch.ones(1, 2))

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted created-state validator was called",
        ):
            residual.apply_to_layer_state(layer_state, torch.ones(1, 2))

        self.assertIsNone(layer_state.residual_state)

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
