import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from emperor.neuron._cluster.plasticity import ClusterPlasticityDelegate
from emperor.neuron._cluster.routing.delegate import ClusterRoutingDelegate
from emperor.neuron._cluster.routing.state import (
    NeuronClusterRouteState,
    _NeuronClusterForwardContext,
)
from emperor.neuron._cluster.topology import ClusterTopologyDelegate


class TestNeuronBeamRoutingContracts(unittest.TestCase):
    def test_mixed_beams_keep_parent_state_and_unscaled_probability_gradients(self):
        events = []
        probabilities = torch.tensor(
            [[0.8, 0.2], [0.5, 0.5], [0.6, 0.4]],
            dtype=torch.float64,
            requires_grad=True,
        )
        auxiliary_loss = torch.tensor(0.125, dtype=torch.float64, requires_grad=True)

        def route_signal(hidden):
            events.append("route")
            torch.testing.assert_close(hidden[:, 0], hidden.new_tensor([1, 4, 5]))
            return (
                probabilities,
                torch.tensor([[[1, 1, 1], [9, 9, 9]]]).expand(hidden.shape[0], -1, -1),
                auxiliary_loss,
            )

        def process_signal(hidden):
            events.append("process")
            return hidden + 10

        owner = SimpleNamespace(
            beam_width=4,
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
        state = routing._ClusterRoutingDelegate__state
        beams = routing._ClusterRoutingDelegate__beam_routes
        previous = NeuronClusterRouteState(
            hidden=torch.arange(1.0, 9.0, dtype=torch.float64)
            .unsqueeze(-1)
            .requires_grad_(),
            positions=torch.tensor(
                [
                    [1, 1, 1],
                    [9, 9, 9],
                    [2, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [8, 8, 8],
                    [2, 1, 1],
                    [0, 0, 0],
                ]
            ),
            active_mask=torch.tensor(
                [True, True, False, True, True, True, False, False]
            ),
            escaped_mask=torch.tensor(
                [False, False, True, False, False, False, False, False]
            ),
            final_mask=torch.tensor(
                [False, False, True, False, False, False, True, True]
            ),
            halting_state=SimpleNamespace(marker="previous"),
            loss=torch.tensor(0.25, dtype=torch.float64, requires_grad=True),
            beam_path_probabilities=torch.tensor(
                [0.4, 0.3, 0.2, 0.1, 0.45, 0.25, 0.2, 0.0],
                dtype=torch.float64,
                requires_grad=True,
            ),
        )
        gathered_halting = SimpleNamespace(marker="gathered")
        updated_halting = SimpleNamespace(marker="updated")

        def gather_halting(prior, parent_rows, usable_mask):
            events.append("gather")
            self.assertIs(prior, previous.halting_state)
            self.assertTrue(usable_mask.all())
            torch.testing.assert_close(
                parent_rows, torch.tensor([0, 1, 2, 0, 4, 5, 6, 4])
            )
            return gathered_halting

        def update_halting(prior, hidden, candidate, update_mask):
            events.append("halt")
            self.assertIs(prior, gathered_halting)
            self.assertIs(hidden, candidate)
            torch.testing.assert_close(
                update_mask,
                torch.tensor([True, False, False, False, True, False, False, False]),
            )
            return updated_halting, candidate

        with (
            patch.object(
                state, "gather_halting_state_rows", side_effect=gather_halting
            ),
            patch.object(
                state, "maybe_update_halting_state", side_effect=update_halting
            ),
        ):
            result = beams._BeamRoutingDelegate__run_beam_route_step(
                previous,
                torch.tensor([True, True, False, True, True, True, False, False]),
                _NeuronClusterForwardContext(),
            )

        self.assertEqual(events, ["route", "process", "gather", "halt"])
        self.assertIs(result.halting_state, updated_halting)
        self.assertIsNone(result.trace)
        torch.testing.assert_close(
            result.hidden[:, 0], previous.hidden.new_tensor([11, 2, 3, 1, 15, 6, 7, 5])
        )
        torch.testing.assert_close(
            result.positions,
            torch.tensor(
                [
                    [1, 1, 1],
                    [9, 9, 9],
                    [2, 1, 1],
                    [9, 9, 9],
                    [1, 1, 1],
                    [8, 8, 8],
                    [2, 1, 1],
                    [9, 9, 9],
                ]
            ),
        )
        torch.testing.assert_close(
            result.active_mask,
            torch.tensor([True, False, False, False, True, False, False, False]),
        )
        torch.testing.assert_close(
            result.escaped_mask,
            torch.tensor([False, False, True, True, False, False, False, True]),
        )
        torch.testing.assert_close(
            result.final_mask,
            torch.tensor([False, True, True, True, False, True, True, True]),
        )
        torch.testing.assert_close(
            result.beam_path_probabilities,
            previous.hidden.new_tensor([0.32, 0.3, 0.2, 0.08, 0.27, 0.25, 0.2, 0.18]),
        )
        torch.testing.assert_close(result.loss, previous.loss.new_tensor(0.375))
        torch.testing.assert_close(
            previous.hidden[:, 0], previous.hidden.new_tensor(list(range(1, 9)))
        )
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

        (
            result.hidden[:, 0].mul(result.beam_path_probabilities).sum() + result.loss
        ).backward()
        torch.testing.assert_close(
            previous.hidden.grad[:, 0],
            previous.hidden.new_tensor([0.4, 0.3, 0.2, 0.0, 0.45, 0.25, 0.2, 0.0]),
        )
        torch.testing.assert_close(
            previous.beam_path_probabilities.grad,
            previous.hidden.new_tensor([9.0, 2.0, 3.0, 0.0, 11.0, 6.0, 7.0, 0.0]),
        )
        torch.testing.assert_close(
            probabilities.grad,
            probabilities.new_tensor([[4.4, 0.4], [0.0, 0.0], [6.75, 2.25]]),
        )
        torch.testing.assert_close(previous.loss.grad, previous.loss.new_ones(()))
        torch.testing.assert_close(auxiliary_loss.grad, auxiliary_loss.new_ones(()))
