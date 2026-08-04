import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from emperor.halting import (
    HaltingHiddenStateModeOptions,
    HaltingUsageTracker,
    SoftHalting,
    StickBreaking,
    StickBreakingConfig,
)
from emperor.halting._validation import StickBreakingValidator
from emperor.halting._variants.soft import _SoftPreparedStep
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig


def gate_config(input_dim: int = 2) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=input_dim,
        output_dim=2,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def stick_config(**overrides) -> StickBreakingConfig:
    values = {
        "input_dim": 2,
        "threshold": 0.99,
        "dropout_probability": None,
        "hidden_state_mode": HaltingHiddenStateModeOptions.RAW,
        "halting_gate_config": gate_config(),
    }
    values.update(overrides)
    return StickBreakingConfig(**values)


class TestHaltingEdgeContracts(unittest.TestCase):
    def test_tracking_fallbacks_clamp_continuation_and_default_to_one(self) -> None:
        tracker = HaltingUsageTracker().double()
        continuation_state = SimpleNamespace(
            halt_mask=None,
            valid_mask=torch.tensor([True, False, True]),
            continuation_probability=torch.tensor([-0.5, 0.25, 1.5]),
            accumulated_halt_probabilities=torch.tensor([0.2, 0.9, 0.8]),
        )
        empty_state = SimpleNamespace(halt_mask=None)

        tracker.begin_forward()
        tracker.record_step(continuation_state)
        tracker.record_step(empty_state)
        tracker.record_final(None, continuation_state)

        self.assertEqual(tracker.last_survival.dtype, torch.float64)
        torch.testing.assert_close(
            tracker.last_survival,
            torch.tensor([5.0 / 12.0, 1.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            tracker.last_step_count,
            torch.tensor(2.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            tracker.last_accumulated_halt_prob_mean,
            torch.tensor(0.5, dtype=torch.float64),
        )
        torch.testing.assert_close(
            tracker.last_remaining_mass_mean,
            torch.tensor(0.5, dtype=torch.float64),
        )
        self.assertEqual(tracker._survival_stage, [])

    def test_ponder_loss_uses_only_compatible_broadcast_selected_entries(
        self,
    ) -> None:
        cases = (
            (
                "one-dimensional mask",
                torch.tensor([True, False]),
                torch.tensor([[2.0, 4.0], [100.0, 200.0]]),
                3.0,
            ),
            (
                "two-dimensional mask",
                torch.tensor([[True], [False]]),
                torch.arange(1.0, 9.0).reshape(2, 2, 2),
                2.5,
            ),
            (
                "all-false compatible mask",
                torch.tensor([False, False]),
                torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
                0.0,
            ),
            (
                "incompatible mask uses unmasked mean",
                torch.tensor([[True, False], [False, True]]),
                torch.arange(1.0, 7.0).reshape(2, 3),
                3.5,
            ),
        )

        for name, valid_mask, ponder_loss, expected in cases:
            with self.subTest(name=name):
                tracker = HaltingUsageTracker()
                state = SimpleNamespace(
                    halt_mask=torch.zeros_like(valid_mask, dtype=torch.bool),
                    valid_mask=valid_mask,
                )

                tracker.begin_forward()
                tracker.record_final(ponder_loss, state)

                torch.testing.assert_close(
                    tracker.last_ponder_loss,
                    torch.tensor(expected),
                )
                self.assertEqual(tracker._survival_stage, [])

    def test_validate_config_matches_model_threshold_boundaries(self) -> None:
        boundary = stick_config(
            input_dim=1,
            threshold=1.0,
            dropout_probability=1.0,
            halting_gate_config=gate_config(1),
        )

        self.assertIsNone(StickBreakingValidator.validate_config(boundary))

        for threshold, exception_type in (
            (True, TypeError),
            (0.0, ValueError),
        ):
            with self.subTest(threshold=threshold):
                invalid = stick_config(threshold=threshold)
                with self.assertRaises(exception_type) as config_error:
                    StickBreakingValidator.validate_config(invalid)
                with self.assertRaises(exception_type) as model_error:
                    StickBreaking(invalid)

                self.assertEqual(
                    str(config_error.exception),
                    str(model_error.exception),
                )

    def test_optional_soft_gate_diagnostics_preserve_none_and_mask_values(
        self,
    ) -> None:
        prepared = _SoftPreparedStep(
            accumulated_hidden=torch.zeros(2, 3),
            continuation_probability=torch.ones(2),
            halt_mask=torch.zeros(2, dtype=torch.bool),
            valid_mask=torch.ones(2, dtype=torch.bool),
            step_count=torch.zeros(2),
            log_continuation=torch.zeros(2),
            accumulated_ponder_cost=torch.zeros(2),
            halt_probability=torch.zeros(2),
            gate_input=None,
            gate_logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            advanced_mask=torch.ones(2, dtype=torch.bool),
        )
        gate_mask = torch.tensor([True, False])

        masked_logits = SoftHalting._SoftHalting__mask_gate_diagnostics(
            prepared,
            gate_mask,
        )

        self.assertIsNone(masked_logits.gate_input)
        torch.testing.assert_close(
            masked_logits.gate_logits,
            torch.tensor([[1.0, 2.0], [0.0, 0.0]]),
        )
        torch.testing.assert_close(
            prepared.gate_logits,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

        input_only = replace(
            prepared,
            gate_input=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            gate_logits=None,
        )
        masked_input = SoftHalting._SoftHalting__mask_gate_diagnostics(
            input_only,
            gate_mask,
        )

        torch.testing.assert_close(
            masked_input.gate_input,
            torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
        )
        self.assertIsNone(masked_input.gate_logits)


if __name__ == "__main__":
    unittest.main()
