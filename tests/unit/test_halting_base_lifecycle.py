import unittest
from dataclasses import fields
from types import SimpleNamespace

import torch
from emperor.halting import HaltingBase, HaltingStateBase
from torch import Tensor


class _LifecycleHalting(HaltingBase[SimpleNamespace]):
    def __init__(self) -> None:
        super().__init__()
        self.input_dim = 2

    def update_halting_state(
        self,
        previous_state: SimpleNamespace | None,
        model_hidden_state: Tensor,
    ) -> tuple[SimpleNamespace, Tensor]:
        leading_shape = model_hidden_state.shape[:-1]
        if previous_state is None:
            state = SimpleNamespace(
                halt_mask=torch.zeros(
                    leading_shape,
                    dtype=torch.bool,
                    device=model_hidden_state.device,
                ),
                log_continuation=model_hidden_state.new_zeros(leading_shape),
                accumulated_hidden=model_hidden_state.clone(),
                output_hidden=model_hidden_state.clone(),
                accumulated_halt_probabilities=model_hidden_state.new_zeros(
                    leading_shape
                ),
                accumulated_ponder_cost=model_hidden_state.new_zeros(leading_shape),
            )
        else:
            state = SimpleNamespace(
                halt_mask=previous_state.halt_mask.clone(),
                log_continuation=previous_state.log_continuation.clone(),
                accumulated_hidden=model_hidden_state.clone(),
                output_hidden=model_hidden_state.clone(),
                accumulated_halt_probabilities=(
                    previous_state.accumulated_halt_probabilities.clone()
                ),
                accumulated_ponder_cost=(
                    previous_state.accumulated_ponder_cost.clone()
                ),
            )
        return state, model_hidden_state

    def finalize_weighted_accumulation(
        self,
        state: SimpleNamespace,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return current_hidden + 1.0, current_hidden.new_ones(current_hidden.shape[:-1])


class HaltingBaseLifecycleTests(unittest.TestCase):
    def test_base_state_contains_only_shared_lifecycle_fields(self) -> None:
        field_names = {field.name for field in fields(HaltingStateBase)}

        self.assertNotIn("raw_hidden", field_names)
        self.assertIn("output_hidden", field_names)
        self.assertIn("continuation_probability", field_names)

    def test_lifecycle_preserves_rows_outside_the_valid_domain(self) -> None:
        model = _LifecycleHalting()
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        valid_mask = torch.tensor([True, False])

        state = model.run_step(
            None,
            hidden,
            lambda computation: computation.raw_hidden + 2.0,
            valid_mask=valid_mask,
        )
        output, loss = model.finalize(state, state.raw_hidden)

        torch.testing.assert_close(
            state.raw_hidden,
            torch.tensor([[3.0, 4.0], [3.0, 4.0]]),
        )
        torch.testing.assert_close(
            output,
            torch.tensor([[4.0, 5.0], [3.0, 4.0]]),
        )
        torch.testing.assert_close(loss, torch.tensor([1.0, 0.0]))
        self.assertTrue(state.finalized)


if __name__ == "__main__":
    unittest.main()
