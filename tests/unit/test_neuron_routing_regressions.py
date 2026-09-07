import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from emperor.neuron._cluster.plasticity import ClusterPlasticityDelegate
from emperor.neuron._cluster.routing.delegate import (
    ClusterRoutingDelegate,
)
from emperor.neuron._cluster.routing.state import (
    NeuronClusterRouteState,
    _NeuronClusterForwardContext,
)
from emperor.neuron._cluster.topology import ClusterTopologyDelegate


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
        route_state = self._route_state(include_beam_path_probabilities=True)
        owner = SimpleNamespace(beam_width=2, cluster={})
        routing = self.routing_delegate(owner)
        harness = routing._ClusterRoutingDelegate__beam_routes

        finalized = harness._BeamRoutingDelegate__run_beam_route_step(
            route_state,
            torch.tensor([True, False]),
            _NeuronClusterForwardContext(),
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

    @staticmethod
    def routing_delegate(owner) -> ClusterRoutingDelegate:
        topology = ClusterTopologyDelegate(owner)
        plasticity = ClusterPlasticityDelegate(owner, topology)
        return ClusterRoutingDelegate(owner, topology, plasticity)

    def test_recurrent_step_finalizes_without_breaking_the_graph(self) -> None:
        route_state = self._route_state(include_beam_path_probabilities=False)
        harness = self.routing_delegate(SimpleNamespace(cluster={}))

        finalized = harness._ClusterRoutingDelegate__run_recurrent_route_step(
            route_state,
            torch.tensor([True, False]),
            _NeuronClusterForwardContext(),
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


class TestNeuronRecurrentStepContracts(unittest.TestCase):
    def test_mixed_routes_preserve_state_gradients_and_trace_order(self) -> None:
        events = []
        probabilities = torch.tensor(
            [[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64, requires_grad=True
        )
        auxiliary_loss = torch.tensor(0.125, dtype=torch.float64, requires_grad=True)

        def route_signal(hidden):
            events.append("route")
            return (
                probabilities,
                torch.tensor([[[1, 1, 1], [9, 9, 9]]]).expand(hidden.shape[0], -1, -1),
                auxiliary_loss,
            )

        def process_signal(hidden):
            events.append("process")
            return hidden * 2

        owner = SimpleNamespace(
            cluster={
                "neuron_1_1_1": SimpleNamespace(
                    route_signal=route_signal, process_signal=process_signal
                )
            },
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            training=False,
            escape_counts=None,
            growth_warmup_steps=None,
        )
        topology = ClusterTopologyDelegate(owner)
        routing = ClusterRoutingDelegate(
            owner, topology, ClusterPlasticityDelegate(owner, topology)
        )
        state_delegate = routing._ClusterRoutingDelegate__state
        previous = NeuronClusterRouteState(
            hidden=torch.tensor(
                [[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64, requires_grad=True
            ),
            positions=torch.tensor([[1, 1, 1], [1, 1, 1], [8, 8, 8], [1, 1, 1]]),
            active_mask=torch.ones(4, dtype=torch.bool),
            escaped_mask=torch.zeros(4, dtype=torch.bool),
            final_mask=torch.zeros(4, dtype=torch.bool),
            halting_state=None,
            loss=torch.tensor(0.25, dtype=torch.float64, requires_grad=True),
            trace=SimpleNamespace(steps=[]),
        )
        updated_halting = SimpleNamespace(halt_mask=torch.tensor([False] * 4))

        def update_halting(prior, hidden, candidate, update_mask):
            events.append("halt")
            self.assertIs(prior, previous.halting_state)
            self.assertIs(hidden, previous.hidden)
            torch.testing.assert_close(
                update_mask, torch.tensor([True, True, False, False])
            )
            return updated_halting, candidate

        def detach_trace(tensor):
            events.append("trace")
            return tensor.detach().clone()

        with (
            patch.object(
                state_delegate, "maybe_update_halting_state", side_effect=update_halting
            ),
            patch.object(
                state_delegate, "detach_trace_tensor", side_effect=detach_trace
            ),
        ):
            result = routing._ClusterRoutingDelegate__run_recurrent_route_step(
                previous,
                torch.tensor([True, True, True, False]),
                _NeuronClusterForwardContext(),
            )

        self.assertEqual(events, ["route", "process", "halt"] + ["trace"] * 7)
        torch.testing.assert_close(
            result.hidden, previous.hidden.new_tensor([[1.75], [2.5], [3.0], [4.0]])
        )
        torch.testing.assert_close(
            result.active_mask, torch.tensor([True, False, False, True])
        )
        torch.testing.assert_close(
            result.escaped_mask, torch.tensor([False, True, False, False])
        )
        torch.testing.assert_close(
            result.final_mask, torch.tensor([False, True, True, False])
        )
        torch.testing.assert_close(result.positions, previous.positions)
        self.assertIs(result.halting_state, updated_halting)
        self.assertIs(result.trace, previous.trace)
        torch.testing.assert_close(result.loss, previous.loss.new_tensor(0.375))
        for field in (
            "hidden",
            "positions",
            "active_mask",
            "escaped_mask",
            "final_mask",
        ):
            self.assertNotEqual(
                getattr(result, field).data_ptr(), getattr(previous, field).data_ptr()
            )
        torch.testing.assert_close(
            previous.active_mask, torch.ones(4, dtype=torch.bool)
        )
        torch.testing.assert_close(
            previous.final_mask, torch.zeros(4, dtype=torch.bool)
        )

        (result.hidden.sum() + result.loss).backward()
        torch.testing.assert_close(
            previous.hidden.grad,
            previous.hidden.new_tensor([[1.75], [1.25], [1.0], [1.0]]),
        )
        torch.testing.assert_close(
            probabilities.grad, probabilities.new_tensor([[2.0, 1.0], [4.0, 2.0]])
        )
        torch.testing.assert_close(auxiliary_loss.grad, auxiliary_loss.new_ones(()))
        torch.testing.assert_close(previous.loss.grad, previous.loss.new_ones(()))
        trace_step = result.trace.steps[0]
        self.assertFalse(trace_step.probabilities.requires_grad)
        self.assertIsNone(trace_step.probabilities.grad_fn)
        result.active_mask.zero_()
        torch.testing.assert_close(
            trace_step.active_mask, torch.tensor([True, False, False, True])
        )

    def test_missing_routes_do_not_process_halt_or_append_trace(self) -> None:
        previous = TestNeuronMissingRouteLifecycle._route_state(
            include_beam_path_probabilities=False
        )
        previous.trace = SimpleNamespace(steps=[])
        routing = TestNeuronMissingRouteLifecycle.routing_delegate(
            SimpleNamespace(cluster={})
        )
        state_delegate = routing._ClusterRoutingDelegate__state
        with (
            patch.object(routing, "run_process_branches") as process,
            patch.object(state_delegate, "maybe_update_halting_state") as halt,
            patch.object(state_delegate, "detach_trace_tensor") as trace,
        ):
            result = routing._ClusterRoutingDelegate__run_recurrent_route_step(
                previous, torch.tensor([True, False]), _NeuronClusterForwardContext()
            )

        process.assert_not_called()
        halt.assert_not_called()
        trace.assert_not_called()
        self.assertIs(result.loss, previous.loss)
        self.assertIs(result.halting_state, previous.halting_state)
        self.assertIs(result.trace, previous.trace)
        self.assertEqual(previous.trace.steps, [])
