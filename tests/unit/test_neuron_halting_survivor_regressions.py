import unittest
from types import SimpleNamespace

import torch

from emperor.neuron._cluster.halting_lifecycle import _NeuronHaltingLifecycle
from unit.test_neuron import NeuronTestCase


class _NeverHaltingModel:
    _usage_tracker = None

    @staticmethod
    def update_halting_state(
        previous_state: SimpleNamespace | None,
        model_hidden_state: torch.Tensor,
    ) -> tuple[SimpleNamespace, torch.Tensor]:
        del previous_state
        state = SimpleNamespace(
            halt_mask=torch.zeros(
                model_hidden_state.shape[:-1],
                dtype=torch.bool,
                device=model_hidden_state.device,
            )
        )
        return state, model_hidden_state


class TestNeuronHaltingSurvivorRegressions(NeuronTestCase):
    def test_stick_breaking_requests_stop_only_after_all_advanced_rows_halt(
        self,
    ) -> None:
        halting_model = (
            self.halting_config(input_dim=2, threshold=0.9).build().double().eval()
        )
        update_mask = torch.tensor([True, True])
        first_hidden = torch.tensor(
            [[1.0, -1.0], [2.0, -2.0]],
            dtype=torch.float64,
        )
        first_candidate = first_hidden + 0.5

        first_state = _NeuronHaltingLifecycle.update(
            halting_model,
            None,
            first_hidden,
            first_candidate,
            update_mask,
        )

        self.assertIsNotNone(first_state)
        self.assertFalse(first_state.halt_mask.any().item())
        self.assertFalse(first_state.stop_requested)

        second_candidate = first_candidate + 0.75
        second_state = _NeuronHaltingLifecycle.update(
            halting_model,
            first_state,
            first_candidate,
            second_candidate,
            update_mask,
        )

        self.assertIsNotNone(second_state)
        self.assertFalse(second_state.halt_mask.any().item())
        self.assertFalse(second_state.stop_requested)
        torch.testing.assert_close(
            second_state.step_indices,
            torch.tensor([1, 1]),
        )

    def test_sparse_updates_preserve_raw_hidden_and_advance_long_step_indices(
        self,
    ) -> None:
        halting_model = _NeverHaltingModel()
        all_rows = torch.tensor([True, True])
        first_candidate = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
        )
        first_state = _NeuronHaltingLifecycle.update(
            halting_model,
            None,
            torch.zeros_like(first_candidate),
            first_candidate,
            all_rows,
        )

        self.assertIsNotNone(first_state)
        self.assertEqual(first_state.step_indices.dtype, torch.long)
        self.assertEqual(first_state.step_indices.device, all_rows.device)
        torch.testing.assert_close(first_state.step_indices, torch.tensor([0, 0]))

        second_candidate = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        second_state = _NeuronHaltingLifecycle.update(
            halting_model,
            first_state,
            first_candidate,
            second_candidate,
            all_rows,
        )

        self.assertIsNotNone(second_state)
        torch.testing.assert_close(second_state.step_indices, torch.tensor([1, 1]))

        sparse_rows = torch.tensor([True, False])
        third_current = torch.tensor(
            [[9.0, 10.0], [11.0, 12.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        third_candidate = torch.tensor(
            [[13.0, 14.0], [15.0, 16.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        third_state = _NeuronHaltingLifecycle.update(
            halting_model,
            second_state,
            third_current,
            third_candidate,
            sparse_rows,
        )

        self.assertIsNotNone(third_state)
        torch.testing.assert_close(
            third_state.raw_hidden,
            torch.tensor(
                [[13.0, 14.0], [7.0, 8.0]],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(third_state.step_indices, torch.tensor([2, 1]))
        self.assertEqual(third_state.step_indices.dtype, torch.long)
        self.assertEqual(third_state.step_indices.device, sparse_rows.device)

        third_state.raw_hidden.sum().backward()
        torch.testing.assert_close(
            second_candidate.grad,
            torch.tensor(
                [[0.0, 0.0], [1.0, 1.0]],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            third_candidate.grad,
            torch.tensor(
                [[1.0, 1.0], [0.0, 0.0]],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            third_current.grad,
            torch.zeros_like(third_current),
        )


if __name__ == "__main__":
    unittest.main()
