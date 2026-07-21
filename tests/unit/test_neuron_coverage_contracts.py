import unittest

from emperor.neuron import NeuronClusterConfig
from emperor.neuron._terminal_topology import initialize_terminal_connections
from emperor.neuron._validation import NeuronValidator
from unit.test_memory import make_memory_config
from unit.test_neuron import NeuronTestCase


class TestNeuronCoverageContracts(NeuronTestCase):
    def test_topology_rejects_unsupported_shape_after_preflight(self):
        terminal_config = self.terminal_config()
        terminal_config.connection_shape = object()

        with self.assertRaisesRegex(ValueError, "Unsupported terminal connection"):
            initialize_terminal_connections(terminal_config)

    def test_matching_axons_memory_dimension_is_accepted(self):
        neuron_config = self.neuron_config()
        neuron_config.axons_config.memory_config = make_memory_config(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )

        NeuronValidator.validate_axons_memory_dimensions(neuron_config)

    def test_cluster_validation_resolves_every_missing_terminal_coordinate(self):
        neuron_config = self.neuron_config()
        terminal_config = neuron_config.terminal_config
        terminal_config.x_axis_position = None
        terminal_config.y_axis_position = None
        terminal_config.z_axis_position = None
        cluster_config = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=neuron_config,
        )

        cluster = cluster_config.build()

        self.assertEqual(len(cluster.cluster), 1)
        self.assertIsNone(terminal_config.x_axis_position)
        self.assertIsNone(terminal_config.y_axis_position)
        self.assertIsNone(terminal_config.z_axis_position)


if __name__ == "__main__":
    unittest.main()
