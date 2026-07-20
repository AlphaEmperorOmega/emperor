import unittest
from types import SimpleNamespace

import torch

from emperor.neuron._cluster.beam_routes import _NeuronClusterBeamRoutesMixin
from emperor.neuron._cluster.recurrent_routes import (
    _NeuronClusterRecurrentRoutesMixin,
)
from emperor.neuron._cluster.routing_numerics import (
    _stable_beam_mixture_with_score_history,
    weighted_branch_candidate,
)
from emperor.neuron._cluster.state import (
    NeuronClusterRouteState,
    _NeuronClusterStateMixin,
)


class TestNeuronRoutingNumerics(unittest.TestCase):
    def test_underflowed_valid_mass_uses_log_scores_for_value_and_gradient(
        self,
    ) -> None:
        logits = torch.tensor(
            [[-30.0, -29.0, 0.0]],
            dtype=torch.float16,
            requires_grad=True,
        )
        branch_outputs = torch.tensor(
            [[[10.0], [20.0], [0.0]]],
            dtype=torch.float16,
        )
        valid_branch_mask = torch.tensor([[True, True, False]])

        candidate = weighted_branch_candidate(
            branch_outputs,
            torch.softmax(logits, dim=1),
            valid_branch_mask,
            log_probabilities=torch.log_softmax(logits, dim=1),
            router_scores=logits,
        )
        candidate.sum().backward()

        reference_logits = torch.tensor([-30.0, -29.0], dtype=torch.float64)
        reference_weights = torch.softmax(reference_logits, dim=0)
        reference_output = (reference_weights * torch.tensor([10.0, 20.0])).sum()
        expected_gradient = torch.tensor(
            [
                reference_weights[0] * (10.0 - reference_output),
                reference_weights[1] * (20.0 - reference_output),
                0.0,
            ],
            dtype=torch.float16,
        ).unsqueeze(0)
        torch.testing.assert_close(
            candidate.float(),
            reference_output.float().reshape(1, 1),
            atol=2e-2,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            logits.grad,
            expected_gradient,
            atol=2e-2,
            rtol=2e-3,
        )

    def test_extreme_route_product_remains_representable(self) -> None:
        logits = torch.tensor(
            [[-110.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        branch_outputs = torch.tensor(
            [[[1e10], [0.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )

        candidate = weighted_branch_candidate(
            branch_outputs,
            torch.softmax(logits, dim=1),
            torch.ones_like(logits, dtype=torch.bool),
            log_probabilities=torch.log_softmax(logits, dim=1),
            router_scores=logits,
        )
        candidate.sum().backward()

        reference_weights = torch.softmax(logits.detach().double(), dim=1)
        reference_output = (
            reference_weights * branch_outputs.detach().double().squeeze(-1)
        ).sum(dim=1)
        expected_logit_gradient = reference_weights * (
            branch_outputs.detach().double().squeeze(-1)
            - reference_output.unsqueeze(1)
        )
        torch.testing.assert_close(
            candidate.squeeze(),
            reference_output.squeeze().float(),
            rtol=5e-5,
            atol=0.0,
        )
        torch.testing.assert_close(
            logits.grad,
            expected_logit_gradient.float(),
            rtol=5e-5,
            atol=0.0,
        )
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_beam_history_cancels_overflow_before_score_reduction(self) -> None:
        values = torch.tensor(
            [[[1e308, -1e308, 1.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        scores = torch.zeros((1, 2), dtype=torch.float64)
        router_scores = torch.tensor(
            [[0.1, -0.3]],
            dtype=torch.float64,
            requires_grad=True,
        )
        output_gradient = torch.tensor(
            [[1e100, 1e100, 1.0]],
            dtype=torch.float64,
        )

        output = _stable_beam_mixture_with_score_history(
            values,
            scores,
            torch.ones_like(scores, dtype=torch.bool),
            (router_scores,),
            (torch.tensor([0, 1]),),
        )
        value_gradient, score_gradient = torch.autograd.grad(
            output,
            (values, router_scores),
            grad_outputs=output_gradient,
        )

        torch.testing.assert_close(
            output,
            torch.tensor([[5e307, -5e307, 0.5]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            value_gradient,
            output_gradient[:, None, :].expand_as(values) * 0.5,
        )
        torch.testing.assert_close(
            score_gradient,
            torch.tensor([[0.25, -0.25]], dtype=torch.float64),
            rtol=5e-15,
            atol=1e-15,
        )
        self.assertTrue(torch.isfinite(score_gradient).all())


class TestNeuronMissingRouteLifecycle(unittest.TestCase):
    @staticmethod
    def _route_state(*, include_beam_scores: bool) -> NeuronClusterRouteState:
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
            beam_scores=(
                torch.log(torch.tensor([0.75, 0.25], dtype=torch.float64))
                if include_beam_scores
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

        route_state = self._route_state(include_beam_scores=True)
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

        route_state = self._route_state(include_beam_scores=False)
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
