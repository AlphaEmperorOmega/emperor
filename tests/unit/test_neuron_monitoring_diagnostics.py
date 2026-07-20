import unittest
from types import SimpleNamespace

import torch

from emperor.neuron._monitoring.diagnostics import _NeuronDiagnostics


class TestNeuronMonitoringDiagnostics(unittest.TestCase):
    def test_single_entry_routing_has_zero_finite_variation(self) -> None:
        trace = SimpleNamespace(entry_probabilities=torch.ones((2, 1)))

        metrics = _NeuronDiagnostics.calculate_entry_routing(trace)

        self.assertIsNotNone(metrics)
        self.assertEqual(float(metrics.coefficient_of_variation), 0.0)
        self.assertTrue(torch.isfinite(metrics.coefficient_of_variation))


if __name__ == "__main__":
    unittest.main()
