import unittest
from dataclasses import fields

import torch

from emperor.halting import HaltingBase, HaltingStateBase


class HaltingBaseLifecycleTests(unittest.TestCase):
    def test_base_state_contains_only_shared_lifecycle_fields(self) -> None:
        field_names = {field.name for field in fields(HaltingStateBase)}

        self.assertNotIn("raw_hidden", field_names)
        self.assertNotIn("stop_requested", field_names)
        self.assertNotIn("finalized", field_names)
        self.assertIn("output_hidden", field_names)
        self.assertIn("continuation_probability", field_names)

    def test_base_state_is_constructible_without_field_arguments(self) -> None:
        state = HaltingStateBase()
        hidden = torch.ones(1, 2)
        position_values = torch.ones(1)
        position_mask = torch.ones(1, dtype=torch.bool)

        state.output_hidden = hidden
        state.accumulated_hidden = hidden
        state.continuation_probability = position_values
        state.halt_mask = position_mask
        state.valid_mask = position_mask
        state.advanced_mask = position_mask
        state.step_indices = position_values

        self.assertIs(state.output_hidden, hidden)
        self.assertIs(state.continuation_probability, position_values)

    def test_base_requires_both_official_lifecycle_methods(self) -> None:
        model = HaltingBase()
        hidden = torch.ones(1, 2)

        with self.assertRaises(NotImplementedError):
            model.update_halting_state(None, hidden)
        with self.assertRaises(NotImplementedError):
            model.finalize_weighted_accumulation(None, hidden)


if __name__ == "__main__":
    unittest.main()
