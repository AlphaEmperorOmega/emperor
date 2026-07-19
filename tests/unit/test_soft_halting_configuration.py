import unittest

import torch
from emperor.halting import (
    HaltingHiddenStateModeOptions,
    HaltingStateBase,
    SoftHalting,
    SoftHaltingConfig,
    SoftHaltingState,
)
from torch import nn


class SoftHaltingConfigurationTests(unittest.TestCase):
    def test_default_gate_uses_the_canonical_two_projection_shape(self) -> None:
        model = SoftHalting(
            SoftHaltingConfig(
                input_dim=3,
                threshold=None,
                dropout_probability=0.25,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=None,
            )
        )

        self.assertEqual(model.threshold, 0.999)
        self.assertIsInstance(model._gate, nn.Sequential)
        self.assertEqual(
            tuple(type(module) for module in model._gate),
            (nn.Linear, nn.GELU, nn.Dropout, nn.Linear),
        )
        self.assertEqual(model._gate[2].p, 0.25)
        self.assertEqual(model._gate[-1].out_features, 2)
        self.assertEqual(model._gate[-1].weight.count_nonzero().item(), 0)

    def test_raw_hidden_belongs_to_the_soft_state_only(self) -> None:
        self.assertNotIn("raw_hidden", HaltingStateBase.__dataclass_fields__)
        self.assertIn("raw_hidden", SoftHaltingState.__dataclass_fields__)

    def test_owner_step_helpers_gather_rows_and_freeze_rejected_updates(self) -> None:
        model = SoftHalting(
            SoftHaltingConfig(
                input_dim=3,
                threshold=0.9,
                dropout_probability=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=None,
            )
        ).eval()
        raw_hidden = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        initial_step = model.prepare_owner_step(None, raw_hidden)
        previous_state = model.complete_owner_step(
            initial_step,
            raw_hidden + 1.0,
        )
        later_step = model.prepare_owner_step(
            previous_state,
            previous_state.raw_hidden,
        )

        gathered = model.gather_owner_step_rows(
            later_step,
            torch.tensor([1]),
            previous_state=None,
        )
        restricted = model.restrict_owner_step_updates(
            later_step,
            torch.tensor([True, False]),
        )
        next_state = model.complete_owner_step(
            restricted,
            restricted.raw_hidden + 2.0,
        )

        self.assertEqual(tuple(gathered.raw_hidden.shape), (1, 3))
        self.assertTrue(
            torch.equal(restricted.update_mask, torch.tensor([True, False]))
        )
        self.assertFalse(restricted.computation.computation_mask[1].item())
        torch.testing.assert_close(
            next_state.raw_hidden[1],
            previous_state.raw_hidden[1],
        )
        torch.testing.assert_close(
            next_state.output_hidden[1],
            previous_state.output_hidden[1],
        )


if __name__ == "__main__":
    unittest.main()
