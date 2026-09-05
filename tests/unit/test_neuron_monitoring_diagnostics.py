import unittest
from dataclasses import replace

import torch

from emperor.neuron import NeuronClusterConfig
from emperor.neuron._monitoring.diagnostics import _NeuronDiagnostics
from unit.test_neuron import NeuronTestCase


class TestNeuronMonitoringDiagnostics(NeuronTestCase):
    def entry_trace(self, inputs, top_k=1):
        cluster = (
            NeuronClusterConfig(
                x_axis_total_neurons=2,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                neuron_config=self.full_sampler_neuron_config(),
                entry_sampler_config=self.sampler_config(num_experts=2, top_k=top_k),
            )
            .build()
            .eval()
        )
        entry_layer = cluster.entry_sampler.router.model.layers[0].model
        with torch.no_grad():
            entry_layer.weight_params.zero_()
            entry_layer.bias_params.zero_()
            entry_layer.weight_params[0] = torch.tensor([1.0, -1.0])
        _, _, trace = cluster(inputs, return_trace=True)
        return trace

    def test_balanced_top_one_destinations_have_log_two_entropy_and_zero_variation(
        self,
    ):
        trace = self.entry_trace(
            torch.tensor([[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]])
        )
        self.assertFalse(
            torch.equal(
                trace.entry_selected_coordinates[0], trace.entry_selected_coordinates[1]
            )
        )
        metrics = _NeuronDiagnostics.calculate_entry_routing(trace)
        torch.testing.assert_close(metrics.mean_entropy, torch.zeros(()))
        torch.testing.assert_close(metrics.marginal_entropy, torch.tensor(2.0).log())
        torch.testing.assert_close(metrics.coefficient_of_variation, torch.zeros(()))

    def test_single_entry_routing_has_zero_finite_variation(self) -> None:
        cluster = (
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                neuron_config=self.full_sampler_neuron_config(),
            )
            .build()
            .eval()
        )
        _, _, trace = cluster(torch.ones(2, self.input_dim), return_trace=True)

        metrics = _NeuronDiagnostics.calculate_entry_routing(trace)

        self.assertIsNotNone(metrics)
        self.assertEqual(float(metrics.coefficient_of_variation), 0.0)
        self.assertTrue(torch.isfinite(metrics.coefficient_of_variation))

    def test_collapsed_top_one_retains_unused_destination_bin(self):
        trace = self.entry_trace(
            torch.tensor([[2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
        )
        metrics = _NeuronDiagnostics.calculate_entry_routing(trace)
        torch.testing.assert_close(metrics.marginal_entropy, torch.zeros(()))
        torch.testing.assert_close(metrics.coefficient_of_variation, torch.ones(()))

    def test_top_k_metrics_follow_destinations_when_selection_order_changes(self):
        trace = self.entry_trace(
            torch.tensor([[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]), top_k=2
        )
        metrics = _NeuronDiagnostics.calculate_entry_routing(trace)
        torch.testing.assert_close(metrics.marginal_entropy, torch.tensor(2.0).log())
        torch.testing.assert_close(metrics.coefficient_of_variation, torch.zeros(()))
        normalized = trace.entry_probabilities / trace.entry_probabilities.sum(
            -1, keepdim=True
        )
        expected_entropy = -(normalized * normalized.log()).sum(-1).mean()
        torch.testing.assert_close(metrics.mean_entropy, expected_entropy)
        reordered = replace(
            trace,
            entry_probabilities=trace.entry_probabilities.flip(-1),
            entry_selected_coordinates=trace.entry_selected_coordinates.flip(-2),
        )
        reordered_metrics = _NeuronDiagnostics.calculate_entry_routing(reordered)
        for name in metrics.__dataclass_fields__:
            torch.testing.assert_close(
                getattr(metrics, name), getattr(reordered_metrics, name)
            )

    def test_repeated_destinations_accumulate_and_zero_mass_stays_finite(self):
        trace = self.entry_trace(torch.ones(2, self.input_dim), top_k=2)
        repeated = replace(
            trace, entry_selected_coordinates=trace.entry_coordinates[0].expand(2, 2, 3)
        )
        metrics = _NeuronDiagnostics.calculate_entry_routing(repeated)
        torch.testing.assert_close(metrics.marginal_entropy, torch.zeros(()))
        torch.testing.assert_close(metrics.coefficient_of_variation, torch.ones(()))
        zero_mass = replace(
            trace, entry_probabilities=torch.zeros_like(trace.entry_probabilities)
        )
        zero_metrics = _NeuronDiagnostics.calculate_entry_routing(zero_mass)
        for name in zero_metrics.__dataclass_fields__:
            torch.testing.assert_close(getattr(zero_metrics, name), torch.zeros(()))

    def test_empty_observations_have_no_entry_metrics(self):
        trace = self.entry_trace(torch.empty(0, self.input_dim))
        self.assertIsNone(_NeuronDiagnostics.calculate_entry_routing(trace))


if __name__ == "__main__":
    unittest.main()
