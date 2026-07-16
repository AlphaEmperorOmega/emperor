from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch import nn

from tests.support.inspection import (
    inspect_model,
    serialize_graph,
)


class InspectionGraphSpecializedNodeTests(unittest.TestCase):
    def test_graph_serializer_reports_neuron_cluster_grid(self) -> None:
        result = inspect_model(
            "neuron/linear",
            "baseline",
            {
                "cluster_x_axis_total_neurons": "3",
                "cluster_y_axis_total_neurons": "3",
                "cluster_z_axis_total_neurons": "1",
                "cluster_initial_x_axis_total_neurons": "2",
                "cluster_initial_y_axis_total_neurons": "2",
            },
        )
        cluster_node = next(
            node for node in result["nodes"] if node["typeName"] == "NeuronCluster"
        )
        cluster = cluster_node["details"]["cluster"]

        self.assertEqual(cluster_node["id"], "neuron_cluster")
        self.assertEqual(cluster["capacity"], [3, 3, 1])
        self.assertEqual(cluster["initial"], [2, 2, 1])
        self.assertEqual(cluster["initialStart"], [1, 1, 1])
        self.assertEqual(cluster["instantiated"], 4)
        self.assertEqual(
            cluster["coordinates"],
            [[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1]],
        )
        self.assertNotIn("recurrent", cluster_node["details"])

    def test_graph_serializer_reports_terminal_reachable_area(self) -> None:
        result = inspect_model("neuron/linear", "baseline")
        terminal_node = next(
            node for node in result["nodes"] if node["typeName"] == "Terminal"
        )
        reach = terminal_node["details"]["terminalReach"]

        self.assertEqual(
            terminal_node["id"],
            "neuron_cluster.cluster.neuron_4_4_1.terminal",
        )
        self.assertEqual(reach["position"], [4, 4, 1])
        self.assertEqual(reach["total"], len(reach["connections"]))
        self.assertIn([4, 4, 1], reach["connections"])
        self.assertIn([3, 3, 1], reach["connections"])
        self.assertTrue(
            all(len(coordinate) == 3 for coordinate in reach["connections"])
        )

    def test_graph_serializer_serializes_all_neuron_variants(self) -> None:
        for model_name in (
            "neuron/linear",
            "neuron/linear_adaptive",
            "neuron/expert_linear",
            "neuron/expert_linear_adaptive",
        ):
            with self.subTest(model_name=model_name):
                result = inspect_model(model_name, "baseline")
                type_names = {node["typeName"] for node in result["nodes"]}

                self.assertEqual(result["modelType"], "neuron")
                self.assertIn("NeuronCluster", type_names)

    def test_graph_serializer_skips_uninitialized_lazy_parameters(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.LazyLinear(3)))
        count_by_id = {node["id"]: node["parameterCount"] for node in nodes}
        size_by_id = {node["id"]: node["parameterSizeBytes"] for node in nodes}
        details_by_id = {node["id"]: node["details"] for node in nodes}

        self.assertEqual(count_by_id["__root__"], 0)
        self.assertEqual(count_by_id["0"], 0)
        self.assertEqual(size_by_id["__root__"], 0)
        self.assertEqual(size_by_id["0"], 0)
        self.assertNotIn("weightShape", details_by_id["0"])
        self.assertNotIn("biasShape", details_by_id["0"])


if __name__ == "__main__":
    unittest.main()
