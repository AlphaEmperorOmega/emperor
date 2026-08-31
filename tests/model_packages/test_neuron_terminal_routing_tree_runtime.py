import importlib
import unittest

from emperor.neuron import TerminalRoutingTreeDepthOptions

PACKAGE_BUILDERS = (
    ("linear", "NeuronLinearConfigBuilder"),
    ("expert_linear", "NeuronExpertLinearConfigBuilder"),
    ("expert_linear_adaptive", "NeuronExpertLinearAdaptiveConfigBuilder"),
)


class TestNeuronTerminalRoutingTreeRuntime(unittest.TestCase):
    @staticmethod
    def runtime_from_flat(package_name: str, values=None):
        runtime_defaults = importlib.import_module(
            f"models.neuron.{package_name}.runtime_defaults"
        )
        return runtime_defaults.runtime_from_flat(values)

    @staticmethod
    def build_model_config(package_name: str, builder_name: str, runtime):
        builder_module = importlib.import_module(
            f"models.neuron.{package_name}.config_builder"
        )
        return getattr(builder_module, builder_name)(runtime=runtime).build()

    @staticmethod
    def terminal_config(model_config):
        return model_config.experiment_config.neuron_cluster_config.neuron_config.terminal_config

    def test_all_packages_default_to_the_flat_terminal(self) -> None:
        for package_name, builder_name in PACKAGE_BUILDERS:
            with self.subTest(package=package_name):
                runtime = self.runtime_from_flat(package_name)
                terminal_options = runtime._as_construction_kwargs()["terminal_options"]
                self.assertIsNone(terminal_options.routing_tree)
                model_config = self.build_model_config(
                    package_name,
                    builder_name,
                    runtime,
                )
                self.assertIsNone(
                    self.terminal_config(model_config).routing_tree_config
                )

    def test_all_packages_build_depth_two_and_three_core_configs(self) -> None:
        configurations = (
            (
                {
                    "cluster_terminal_routing_tree_depth": (
                        TerminalRoutingTreeDepthOptions.TWO
                    ),
                    "cluster_terminal_routing_tree_level_1_branch_count": 4,
                    "cluster_terminal_routing_tree_level_1_top_k": 2,
                },
                (4,),
                (2,),
            ),
            (
                {
                    "cluster_terminal_routing_tree_depth": (
                        TerminalRoutingTreeDepthOptions.THREE
                    ),
                    "cluster_terminal_routing_tree_level_1_branch_count": 2,
                    "cluster_terminal_routing_tree_level_1_top_k": 2,
                    "cluster_terminal_routing_tree_level_2_branch_count": 2,
                    "cluster_terminal_routing_tree_level_2_top_k": 1,
                },
                (2, 2),
                (2, 1),
            ),
        )
        for package_name, builder_name in PACKAGE_BUILDERS:
            for values, expected_branch_counts, expected_top_k in configurations:
                with self.subTest(package=package_name, values=values):
                    runtime = self.runtime_from_flat(package_name, values)
                    model_config = self.build_model_config(
                        package_name,
                        builder_name,
                        runtime,
                    )
                    tree_config = self.terminal_config(model_config).routing_tree_config
                    self.assertEqual(
                        tree_config.direction_branch_counts,
                        expected_branch_counts,
                    )
                    self.assertEqual(tree_config.direction_top_k, expected_top_k)
                    self.assertIsNone(tree_config.direction_sampler_config)

    def test_all_packages_reject_partial_or_extraneous_level_settings(self) -> None:
        invalid_values = (
            {"cluster_terminal_routing_tree_level_1_branch_count": 4},
            {
                "cluster_terminal_routing_tree_depth": (
                    TerminalRoutingTreeDepthOptions.TWO
                ),
                "cluster_terminal_routing_tree_level_1_branch_count": 4,
            },
            {
                "cluster_terminal_routing_tree_depth": (
                    TerminalRoutingTreeDepthOptions.TWO
                ),
                "cluster_terminal_routing_tree_level_1_branch_count": 4,
                "cluster_terminal_routing_tree_level_1_top_k": 1,
                "cluster_terminal_routing_tree_level_2_branch_count": 2,
            },
            {
                "cluster_terminal_routing_tree_depth": (
                    TerminalRoutingTreeDepthOptions.THREE
                ),
                "cluster_terminal_routing_tree_level_1_branch_count": 2,
                "cluster_terminal_routing_tree_level_1_top_k": 1,
            },
        )
        for package_name, _ in PACKAGE_BUILDERS:
            for values in invalid_values:
                with self.subTest(package=package_name, values=values):
                    with self.assertRaises(ValueError):
                        self.runtime_from_flat(package_name, values)


if __name__ == "__main__":
    unittest.main()
