import unittest
from types import SimpleNamespace

import torch

from emperor.neuron._cluster.beam_routes import _NeuronClusterBeamRoutesMixin
from emperor.neuron._cluster.recurrent_routes import (
    _NeuronClusterRecurrentRoutesMixin,
)
from emperor.neuron._cluster.state import (
    NeuronClusterRouteState,
    _NeuronClusterStateMixin,
)


class TestNeuronMissingRouteLifecycle(unittest.TestCase):
    @staticmethod
    def _route_state(
        *, include_beam_path_probabilities: bool
    ) -> NeuronClusterRouteState:
        hidden = torch.tensor(
            [[1.0], [2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        return NeuronClusterRouteState(
            hidden=hidden,
            positions=torch.tensor([[9, 9, 9], [1, 1, 1]]),
            active_mask=torch.tensor([True, True]),
            escaped_mask=torch.tensor([False, False]),
            final_mask=torch.tensor([False, False]),
            halting_state=SimpleNamespace(marker="halting"),
            loss=torch.tensor(0.25, dtype=torch.float64),
            trace=SimpleNamespace(marker="trace"),
            beam_path_probabilities=(
                torch.tensor([0.75, 0.25], dtype=torch.float64)
                if include_beam_path_probabilities
                else None
            ),
        )

    def test_beam_step_finalizes_only_the_missing_route(self) -> None:
        class BeamHarness(
            _NeuronClusterBeamRoutesMixin,
            _NeuronClusterStateMixin,
        ):
            @staticmethod
            def _coordinate_from_row(row):
                return tuple(int(value) for value in row)

            @staticmethod
            def _neuron_name(x, y, z):
                return f"neuron_{x}_{y}_{z}"

        route_state = self._route_state(include_beam_path_probabilities=True)
        harness = BeamHarness()
        harness.beam_width = 2
        harness.cluster = {}

        finalized = harness._NeuronClusterBeamRoutesMixin__run_beam_route_step(
            route_state,
            torch.tensor([True, False]),
        )

        torch.testing.assert_close(finalized.active_mask, torch.tensor([False, True]))
        torch.testing.assert_close(finalized.final_mask, torch.tensor([True, False]))
        self.assertIs(finalized.hidden, route_state.hidden)
        self.assertIs(finalized.halting_state, route_state.halting_state)
        finalized.hidden.sum().backward()
        torch.testing.assert_close(
            route_state.hidden.grad,
            torch.ones_like(route_state.hidden),
        )

    def test_recurrent_step_finalizes_without_breaking_the_graph(self) -> None:
        class RecurrentHarness(
            _NeuronClusterRecurrentRoutesMixin,
            _NeuronClusterStateMixin,
        ):
            @staticmethod
            def _coordinate_from_row(row):
                return tuple(int(value) for value in row)

            @staticmethod
            def _neuron_name(x, y, z):
                return f"neuron_{x}_{y}_{z}"

        route_state = self._route_state(include_beam_path_probabilities=False)
        harness = RecurrentHarness()
        harness.cluster = {}

        finalized = (
            harness._NeuronClusterRecurrentRoutesMixin__run_recurrent_route_step(
                route_state,
                torch.tensor([True, False]),
            )
        )

        torch.testing.assert_close(finalized.active_mask, torch.tensor([False, True]))
        torch.testing.assert_close(finalized.final_mask, torch.tensor([True, False]))
        self.assertIs(finalized.halting_state, route_state.halting_state)
        self.assertIs(finalized.trace, route_state.trace)
        finalized.hidden.sum().backward()
        torch.testing.assert_close(
            route_state.hidden.grad,
            torch.ones_like(route_state.hidden),
        )
