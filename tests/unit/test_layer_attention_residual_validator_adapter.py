import unittest

import torch

from emperor.layers import LayerState
from emperor.layers._composition.residual.config import AttentionResidualConfig
from emperor.layers._composition.residual.validation import (
    AttentionResidualValidator,
)
from emperor.layers._composition.residual.variants.attention import (
    AttentionResidual,
    AttentionResidualState,
)

from emperor.linears import LinearLayerConfig


class TestAttentionResidualValidatorAdapter(unittest.TestCase):
    def test_component_and_state_expose_validator_adapter(self):
        self.assertIs(AttentionResidual.VALIDATOR, AttentionResidualValidator)
        self.assertIs(AttentionResidualState.VALIDATOR, AttentionResidualValidator)

    def test_successful_validations_return_none_without_changing_state(self):
        residual = AttentionResidual(
            AttentionResidualConfig(block_size=2, rms_norm_epsilon=1e-6, residual_dim=3)
        )
        initial_source = torch.ones(1, 3)
        current = torch.full((1, 3), 2.0)
        state = residual.new_state(initial_source)
        state.append(current)
        original_sources = state.sources
        validator = residual.VALIDATOR

        check_only_results = (
            validator.validate_positive_integer(3, name="residual_dim"),
            validator.validate_finite_positive_number(
                1e-6,
                name="rms_norm_epsilon",
            ),
            validator.validate_source(initial_source, residual_dim=3),
            validator.validate_created_attention_state(residual, state),
            validator.validate_attention_state(residual, state),
            validator.validate_attention_forward_inputs(residual, current, state),
        )

        self.assertTupleEqual(check_only_results, (None,) * 6)
        self.assertIs(state.initial_source, initial_source)
        self.assertEqual(state.block_size, 2)
        self.assertEqual(len(state.sources), len(original_sources))
        for source, original_source in zip(
            state.sources, original_sources, strict=True
        ):
            self.assertIs(source, original_source)

    def test_created_state_validation_preserves_missing_state_error_contract(self):
        residual = AttentionResidualConfig(
            block_size=1, rms_norm_epsilon=1e-6, residual_dim=2
        ).build()
        with self.assertRaisesRegex(
            RuntimeError,
            "^AttentionResidual failed to create forward-local residual state\\.$",
        ):
            AttentionResidualValidator.validate_created_attention_state(
                residual,
                None,
            )

    def test_construction_dispatches_through_substituted_validator(self):
        class RejectingValidator(AttentionResidualValidator):
            @staticmethod
            def validate_positive_integer(value, *, name):
                raise RuntimeError("substituted construction validator was called")

        class RejectingAttentionResidual(AttentionResidual):
            VALIDATOR = RejectingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            RejectingAttentionResidual(
                AttentionResidualConfig(
                    block_size=1, rms_norm_epsilon=1e-6, residual_dim=2
                )
            )



    def test_state_construction_dispatches_through_substituted_validator(self):
        class RejectingValidator(AttentionResidualValidator):
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
        class RejectingValidator(AttentionResidualValidator):
            @staticmethod
            def validate_source(source, *, residual_dim):
                raise RuntimeError("substituted lifecycle validator was called")

        class RejectingAttentionResidual(AttentionResidual):
            VALIDATOR = RejectingValidator

        residual = RejectingAttentionResidual(
            AttentionResidualConfig(block_size=1, rms_norm_epsilon=1e-6, residual_dim=2)
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted lifecycle validator was called",
        ):
            residual.new_state(torch.ones(1, 2))

    def test_state_application_dispatches_created_state_validation(self):
        validated_connections = []

        class RejectingValidator(AttentionResidualValidator):
            @classmethod
            def validate_created_attention_state(cls, connection, state):
                validated_connections.append(connection)
                raise RuntimeError("substituted created-state validator was called")

        class RejectingAttentionResidual(AttentionResidual):
            VALIDATOR = RejectingValidator

        residual = RejectingAttentionResidual(
            AttentionResidualConfig(block_size=1, rms_norm_epsilon=1e-6, residual_dim=2)
        )
        layer_state = LayerState(hidden=torch.ones(1, 2))

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted created-state validator was called",
        ):
            residual.apply_to_layer_state(layer_state, torch.ones(1, 2))

        self.assertIsNone(layer_state.residual_state)
        self.assertEqual(len(validated_connections), 1)
        self.assertIs(validated_connections[0], residual)

    def test_forward_dispatches_through_substituted_validator_before_mutation(self):
        validated_connections = []

        class RejectingValidator(AttentionResidualValidator):
            @classmethod
            def validate_attention_forward_inputs(
                cls,
                connection,
                current,
                state,
            ):
                validated_connections.append(connection)
                raise RuntimeError("substituted forward validator was called")

        class RejectingAttentionResidual(AttentionResidual):
            VALIDATOR = RejectingValidator

        residual = RejectingAttentionResidual(
            AttentionResidualConfig(block_size=1, rms_norm_epsilon=1e-6, residual_dim=2)
        )
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
        self.assertEqual(len(validated_connections), 1)
        self.assertIs(validated_connections[0], residual)


    def test_generated_query_validation_dispatches_before_history_mutation(self):
        class RejectingValidator(AttentionResidualValidator):
            @staticmethod
            def validate_query_model_output(query, current):
                raise RuntimeError("substituted query validator was called")

        class RejectingAttentionResidual(AttentionResidual):
            VALIDATOR = RejectingValidator

        residual = RejectingAttentionResidual(
            AttentionResidualConfig(
                block_size=1,
                rms_norm_epsilon=1e-6,
                residual_dim=2,
                model_config=LinearLayerConfig(bias_flag=False),
            )
        )
        initial = torch.zeros(1, 2)
        state = residual.new_state(initial)
        with self.assertRaisesRegex(
            RuntimeError, "substituted query validator was called"
        ):
            residual(torch.ones_like(initial), initial, residual_state=state)
        self.assertEqual(len(state.sources), 1)
        self.assertIs(state.sources[0], initial)



if __name__ == "__main__":
    unittest.main()
