import unittest
from types import SimpleNamespace

import torch

from emperor.neuron._cluster.halting_lifecycle import _NeuronHaltingLifecycle


class _HaltingModelStub:
    def __init__(
        self,
        *,
        updated_state: SimpleNamespace | None = None,
        ponder_loss: torch.Tensor | None = None,
    ) -> None:
        self.updated_state = updated_state
        self.ponder_loss = ponder_loss
        self._usage_tracker = None

    def update_halting_state(
        self,
        previous_state: SimpleNamespace,
        model_hidden_state: torch.Tensor,
    ) -> tuple[SimpleNamespace, torch.Tensor]:
        assert self.updated_state is not None
        return self.updated_state, model_hidden_state

    def finalize_weighted_accumulation(
        self,
        state: SimpleNamespace,
        current_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.ponder_loss is not None
        return current_hidden, self.ponder_loss


def _previous_state(**attributes: object) -> SimpleNamespace:
    return SimpleNamespace(
        halt_mask=torch.tensor([False, False]),
        advanced_mask=torch.tensor([True, False]),
        raw_hidden=torch.zeros(2, 1, dtype=torch.float64),
        step_indices=torch.tensor([1, 0]),
        **attributes,
    )


class TestNeuronHaltingLifecycle(unittest.TestCase):
    def test_sparse_update_populates_missing_route_metadata_and_gradients(
        self,
    ) -> None:
        current_hidden = torch.tensor(
            [[1.0], [2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        weighted_candidate = torch.tensor(
            [[3.0], [4.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        model = _HaltingModelStub(
            updated_state=SimpleNamespace(
                halt_mask=torch.tensor([False, True]),
                step_count=torch.tensor(1),
                accumulated_ponder_cost=torch.tensor(2.0, dtype=torch.float64),
            )
        )

        state = _NeuronHaltingLifecycle.update(
            model,
            SimpleNamespace(halt_mask=torch.tensor([False, False])),
            current_hidden,
            weighted_candidate,
            torch.tensor([False, True]),
        )

        self.assertIsNotNone(state)
        torch.testing.assert_close(state.advanced_mask, torch.tensor([False, True]))
        torch.testing.assert_close(state.valid_mask, torch.tensor([False, True]))
        torch.testing.assert_close(
            state.raw_hidden,
            torch.tensor([[1.0], [4.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            state.continuation_probability,
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        )
        torch.testing.assert_close(state.step_count, torch.tensor([1, 1]))
        torch.testing.assert_close(state.step_indices, torch.tensor([1, 1]))
        torch.testing.assert_close(
            state.accumulated_ponder_cost,
            torch.tensor([2.0, 2.0], dtype=torch.float64),
        )
        self.assertTrue(state.stop_requested)
        self.assertFalse(state.finalized)

        state.raw_hidden.sum().backward()
        torch.testing.assert_close(
            current_hidden.grad,
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            weighted_candidate.grad,
            torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        )

    def test_sparse_state_update_preserves_scalar_metric_and_gradient(self) -> None:
        previous_metric = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        updated_metric = torch.tensor(
            [7.0, 8.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        model = _HaltingModelStub(
            updated_state=SimpleNamespace(
                halt_mask=torch.tensor([False, False]),
                metric=updated_metric,
            )
        )

        state = _NeuronHaltingLifecycle.update(
            model,
            _previous_state(metric=previous_metric),
            torch.tensor([[1.0], [2.0]], dtype=torch.float64),
            torch.tensor([[3.0], [4.0]], dtype=torch.float64),
            torch.tensor([False, True]),
        )

        self.assertIsNotNone(state)
        torch.testing.assert_close(
            state.metric,
            torch.tensor([5.0, 8.0], dtype=torch.float64),
        )
        state.metric.sum().backward()
        torch.testing.assert_close(
            previous_metric.grad,
            previous_metric.new_tensor(1.0),
        )
        torch.testing.assert_close(
            updated_metric.grad,
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        )

    def test_sparse_update_preserves_strategy_state_after_scalar_fields(self) -> None:
        previous_metric = torch.tensor(
            [5.0, 6.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        updated_metric = torch.tensor(
            [7.0, 8.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        model = _HaltingModelStub(
            updated_state=SimpleNamespace(
                scalar_prefix=torch.tensor(9.0, dtype=torch.float64),
                metric=updated_metric,
                halt_mask=torch.tensor([False, False]),
            )
        )
        previous_state = SimpleNamespace(
            halt_mask=torch.tensor([False, False]),
            scalar_prefix=torch.tensor(3.0, dtype=torch.float64),
            metric=previous_metric,
            strategy_token="preserved",
        )

        state = _NeuronHaltingLifecycle.update(
            model,
            previous_state,
            torch.ones(2, 1, dtype=torch.float64),
            torch.full((2, 1), 2.0, dtype=torch.float64),
            torch.tensor([False, True]),
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.strategy_token, "preserved")
        torch.testing.assert_close(
            state.scalar_prefix,
            torch.tensor(9.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            state.metric,
            torch.tensor([5.0, 8.0], dtype=torch.float64),
        )
        state.metric.sum().backward()
        torch.testing.assert_close(
            previous_metric.grad,
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            updated_metric.grad,
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        )

    def test_sparse_state_update_rejects_unfreezable_row_state(self) -> None:
        cases = {
            "new-row-state": ({}, torch.tensor([7.0, 8.0])),
            "changed-row-shape": (
                {"metric": torch.zeros(2, 1)},
                torch.zeros(2, 2),
            ),
        }
        for case_name, (previous_attributes, updated_metric) in cases.items():
            with self.subTest(case_name=case_name):
                model = _HaltingModelStub(
                    updated_state=SimpleNamespace(
                        halt_mask=torch.tensor([False, False]),
                        metric=updated_metric,
                    )
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "row-aligned state must retain its tensor schema and shape",
                ):
                    _NeuronHaltingLifecycle.update(
                        model,
                        _previous_state(**previous_attributes),
                        torch.ones(2, 1),
                        torch.ones(2, 1),
                        torch.tensor([False, True]),
                    )

    def test_scalar_ponder_loss_is_already_reduced(self) -> None:
        ponder_loss = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
        model = _HaltingModelStub(ponder_loss=ponder_loss)
        state = SimpleNamespace(advanced_mask=torch.tensor([True, False]))

        _, reduced_loss = _NeuronHaltingLifecycle.finalize(
            model,
            state,
            torch.zeros(2, 1, dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        )

        torch.testing.assert_close(reduced_loss, ponder_loss)
        reduced_loss.backward()
        torch.testing.assert_close(ponder_loss.grad, ponder_loss.new_tensor(1.0))

    def test_multidimensional_ponder_loss_masks_rows_and_gradients(self) -> None:
        ponder_loss = torch.tensor(
            [[1.0, 3.0], [100.0, 200.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        model = _HaltingModelStub(ponder_loss=ponder_loss)
        state = SimpleNamespace(advanced_mask=torch.tensor([True, False]))

        _, reduced_loss = _NeuronHaltingLifecycle.finalize(
            model,
            state,
            torch.zeros(2, 1, dtype=torch.float64),
            None,
        )

        torch.testing.assert_close(reduced_loss, torch.tensor(2.0, dtype=torch.float64))
        reduced_loss.backward()
        torch.testing.assert_close(
            ponder_loss.grad,
            torch.tensor([[0.5, 0.5], [0.0, 0.0]], dtype=torch.float64),
        )

    def test_vector_ponder_loss_requires_advanced_and_live_beam_rows(self) -> None:
        ponder_loss = torch.tensor(
            [2.0, 100.0, 1000.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        model = _HaltingModelStub(ponder_loss=ponder_loss)
        state = SimpleNamespace(
            advanced_mask=torch.tensor([True, True, False]),
        )

        _, reduced_loss = _NeuronHaltingLifecycle.finalize(
            model,
            state,
            torch.zeros(3, 1, dtype=torch.float64),
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        )

        torch.testing.assert_close(
            reduced_loss,
            torch.tensor(2.0, dtype=torch.float64),
        )
        reduced_loss.backward()
        torch.testing.assert_close(
            ponder_loss.grad,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

    def test_vector_ponder_loss_requires_aligned_route_metadata(self) -> None:
        cases = {
            "missing-advanced-mask": (SimpleNamespace(), None, "advanced mask"),
            "misaligned-advanced-mask": (
                SimpleNamespace(advanced_mask=torch.tensor([True])),
                None,
                "advanced mask",
            ),
            "misaligned-beam": (
                SimpleNamespace(
                    advanced_mask=torch.tensor([True, True]),
                ),
                torch.tensor([0.0]),
                "beam path probabilities",
            ),
        }
        for case_name, (state, beam_path_probabilities, message) in cases.items():
            with self.subTest(case_name=case_name):
                model = _HaltingModelStub(ponder_loss=torch.tensor([1.0, 3.0]))

                with self.assertRaisesRegex(ValueError, message):
                    _NeuronHaltingLifecycle.finalize(
                        model,
                        state,
                        torch.zeros(2, 1),
                        beam_path_probabilities,
                    )

    def test_absent_halting_state_marks_no_rows_halted(self) -> None:
        halt_mask = _NeuronHaltingLifecycle.halt_mask_tensor(
            None,
            batch_size=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(halt_mask.dtype, torch.bool)
        self.assertEqual(halt_mask.device, torch.device("cpu"))
        torch.testing.assert_close(halt_mask, torch.tensor([False, False, False]))

    def test_absent_halting_state_inherits_requested_meta_device(self) -> None:
        halt_mask = _NeuronHaltingLifecycle.halt_mask_tensor(
            None,
            batch_size=3,
            device=torch.device("meta"),
        )

        self.assertEqual(halt_mask.shape, (3,))
        self.assertEqual(halt_mask.dtype, torch.bool)
        self.assertEqual(halt_mask.device, torch.device("meta"))

    def test_gather_state_rows_reorders_one_dimensional_state_with_gradients(
        self,
    ) -> None:
        per_row_value = torch.tensor(
            [10.0, 20.0, 30.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        unrelated_rows = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scalar = torch.tensor(5.0)
        state = SimpleNamespace(
            per_row_value=per_row_value,
            per_row_matrix=torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                dtype=torch.float64,
            ),
            unrelated_rows=unrelated_rows,
            scalar=scalar,
            label="state",
        )

        gathered = _NeuronHaltingLifecycle.gather_state_rows(
            state,
            torch.tensor([2, 0, 2]),
        )

        self.assertIsNotNone(gathered)
        torch.testing.assert_close(
            gathered.per_row_value,
            torch.tensor([30.0, 10.0, 30.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            gathered.per_row_matrix,
            torch.tensor(
                [[5.0, 6.0], [1.0, 2.0], [5.0, 6.0]],
                dtype=torch.float64,
            ),
        )
        self.assertIs(gathered.unrelated_rows, unrelated_rows)
        self.assertIs(gathered.scalar, scalar)
        self.assertEqual(gathered.label, "state")

        gathered.per_row_value.sum().backward()
        torch.testing.assert_close(
            per_row_value.grad,
            torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64),
        )
