import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from emperor.neuron import (
    NeuronClusterConfig,
    TerminalRoutingTreeConfig,
    TerminalRoutingTreeDepthOptions,
)
from emperor.neuron._terminal.routing import TerminalRoutingTreeDelegate
from emperor.sampler import SamplerConfig, SamplerModel
from emperor.sampler._usage import SamplerUsageTrackerManager
from unit.test_neuron import NeuronTestCase


class _ScriptedTreeSampler(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        probabilities: tuple[float, ...],
        indices: tuple[int, ...],
        auxiliary_loss: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.probabilities = probabilities
        self.indices = indices
        self.auxiliary_loss = auxiliary_loss
        self.calls = 0
        self.observed_inputs = []

    def sample_probabilities_and_indices(self, input_matrix, skip_mask=None):
        self.calls += 1
        self.observed_inputs.append(input_matrix.detach().clone())
        batch_size = input_matrix.shape[0]
        probabilities = input_matrix.new_tensor(self.probabilities).expand(
            batch_size,
            -1,
        )
        indices = torch.tensor(
            self.indices,
            dtype=torch.long,
            device=input_matrix.device,
        ).expand(batch_size, -1)
        return (
            probabilities,
            indices,
            None,
            input_matrix.new_tensor(self.auxiliary_loss),
        )


class TestTerminalRoutingTree(NeuronTestCase):
    def routing_tree_config(
        self,
        *,
        depth=TerminalRoutingTreeDepthOptions.TWO,
        branch_counts=(4,),
        direction_top_k=(2,),
        direction_sampler_config=None,
    ) -> TerminalRoutingTreeConfig:
        return TerminalRoutingTreeConfig(
            depth=depth,
            direction_branch_counts=branch_counts,
            direction_top_k=direction_top_k,
            direction_sampler_config=direction_sampler_config,
        )

    def tree_terminal(
        self,
        *,
        leaf_top_k: int = 2,
        routing_tree_config: TerminalRoutingTreeConfig | None = None,
    ):
        config = self.terminal_config(
            sampler_config=self.sampler_config(top_k=leaf_top_k)
        )
        config.routing_tree_config = routing_tree_config or self.routing_tree_config()
        return config.build()

    def test_disabled_tree_preserves_flat_sampler_and_rng_contract(self) -> None:
        first_config = self.terminal_config()
        second_config = copy.deepcopy(first_config)
        second_config.routing_tree_config = None

        torch.manual_seed(173)
        first_terminal = first_config.build()
        first_next_random_values = torch.randn(8)
        torch.manual_seed(173)
        second_terminal = second_config.build()
        second_next_random_values = torch.randn(8)

        self.assertIsInstance(first_terminal.sampler, SamplerModel)
        self.assertIsInstance(second_terminal.sampler, SamplerModel)
        self.assertEqual(
            tuple(first_terminal.state_dict()),
            tuple(second_terminal.state_dict()),
        )
        torch.testing.assert_close(
            first_next_random_values,
            second_next_random_values,
        )

    def test_compiler_balances_three_dimensions_and_covers_connections_once(self):
        coordinates = torch.cartesian_prod(
            torch.arange(3),
            torch.arange(3),
            torch.arange(2),
        )
        config = self.terminal_config(sampler_config=self.sampler_config(top_k=1))
        config.routing_tree_config = self.routing_tree_config(
            branch_counts=(12,),
            direction_top_k=(1,),
        )
        plan = TerminalRoutingTreeDelegate.preflight(
            config,
            coordinates,
        )

        self.assertEqual(plan.root.subdivision, (3, 2, 2))
        assigned_indices = [
            index for child in plan.root.children for index in child.connection_indices
        ]
        self.assertEqual(sorted(assigned_indices), list(range(len(coordinates))))
        self.assertEqual(len(assigned_indices), len(set(assigned_indices)))
        self.assertEqual(len(plan.root.children), 12)

    def test_compiler_is_translation_invariant_and_prunes_empty_regions(self):
        sparse_coordinates = torch.tensor(
            [
                [-2, 0, 0],
                [-1, 0, 0],
                [0, -2, 0],
                [0, -1, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, 2, 0],
                [1, 0, 0],
                [2, 0, 0],
            ]
        )
        tree_config = self.routing_tree_config(
            branch_counts=(8,),
            direction_top_k=(1,),
        )
        config = self.terminal_config(sampler_config=self.sampler_config(top_k=1))
        config.routing_tree_config = tree_config
        original_plan = TerminalRoutingTreeDelegate.preflight(
            config,
            sparse_coordinates,
        )
        translated_plan = TerminalRoutingTreeDelegate.preflight(
            config,
            sparse_coordinates + torch.tensor([11, -7, 5]),
        )

        original_assignments = tuple(
            child.connection_indices for child in original_plan.root.children
        )
        translated_assignments = tuple(
            child.connection_indices for child in translated_plan.root.children
        )
        self.assertEqual(original_assignments, translated_assignments)
        self.assertLess(len(original_plan.root.children), 8)
        self.assertEqual(
            sum(4 in child.connection_indices for child in original_plan.root.children),
            1,
        )

    def test_depth_two_multiplies_local_probabilities_sorts_paths_and_sums_losses(
        self,
    ) -> None:
        terminal = self.tree_terminal()
        self.assertIsInstance(terminal.sampler, TerminalRoutingTreeDelegate)
        root = terminal.sampler.root
        root.sampler = _ScriptedTreeSampler(
            num_experts=4,
            probabilities=(0.4, 0.3),
            indices=(1, 0),
            auxiliary_loss=1.0,
        )
        root.branches[1].sampler = _ScriptedTreeSampler(
            num_experts=root.branches[1].sampler.num_experts,
            probabilities=(0.8, 0.2),
            indices=(0, 1),
            auxiliary_loss=2.0,
        )
        root.branches[0].sampler = _ScriptedTreeSampler(
            num_experts=root.branches[0].sampler.num_experts,
            probabilities=(0.5, 0.25),
            indices=(0, 1),
            auxiliary_loss=3.0,
        )
        input_batch = torch.randn(self.batch_size, self.input_dim)

        _, probabilities, selected_coordinates, auxiliary_loss = terminal(input_batch)

        torch.testing.assert_close(
            probabilities,
            torch.tensor([[0.32, 0.15, 0.08, 0.075]]).expand(
                self.batch_size,
                -1,
            ),
        )
        child_one_global = root.branches[1].global_connection_indices[:2]
        child_zero_global = root.branches[0].global_connection_indices[:2]
        unsorted_probabilities = torch.tensor([0.32, 0.08, 0.15, 0.075])
        unsorted_indices = torch.cat((child_one_global, child_zero_global))
        expected_order = unsorted_probabilities.sort(
            descending=True, stable=True
        ).indices
        expected_coordinates = terminal.neuron_connections[
            unsorted_indices[expected_order]
        ]
        torch.testing.assert_close(
            selected_coordinates,
            expected_coordinates.expand(self.batch_size, -1, -1),
        )
        torch.testing.assert_close(auxiliary_loss, torch.tensor(6.0))
        self.assertEqual(root.branches[2].sampler.get_auxiliary_loss().item(), 0.0)
        self.assertEqual(root.branches[3].sampler.get_auxiliary_loss().item(), 0.0)
        torch.testing.assert_close(
            root.branches[0].sampler.observed_inputs[0],
            input_batch,
        )
        torch.testing.assert_close(
            root.branches[1].sampler.observed_inputs[0],
            input_batch,
        )

    def test_depth_three_has_fixed_branch_preserving_width(self) -> None:
        terminal = self.tree_terminal(
            leaf_top_k=1,
            routing_tree_config=self.routing_tree_config(
                depth=TerminalRoutingTreeDepthOptions.THREE,
                branch_counts=(2, 2),
                direction_top_k=(2, 1),
            ),
        )

        _, probabilities, selected_coordinates, _ = terminal(
            torch.randn(self.batch_size, self.input_dim)
        )

        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_coordinates.shape, (self.batch_size, 2, 3))
        self.assertEqual(terminal.sampler.plan.output_width, 2)

    def test_delegate_owns_plan_without_retaining_terminal_connections(self) -> None:
        terminal = self.tree_terminal()

        self.assertIsInstance(terminal.sampler, TerminalRoutingTreeDelegate)
        self.assertFalse(hasattr(terminal, "routing_tree_plan"))
        self.assertNotIn("neuron_connections", terminal.sampler.__dict__)
        self.assertNotIn(
            "neuron_connections",
            dict(terminal.sampler.named_buffers()),
        )

    def test_distinct_direction_template_is_derived_without_mutation(self) -> None:
        direction_template = self.sampler_config(top_k=1)
        direction_template.normalize_probabilities_flag = True
        original_num_experts = direction_template.num_experts
        terminal = self.tree_terminal(
            routing_tree_config=self.routing_tree_config(
                direction_sampler_config=direction_template,
            )
        )
        root = terminal.sampler.root

        self.assertTrue(root.sampler.sampler_config.normalize_probabilities_flag)
        self.assertEqual(root.sampler.sampler_config.num_experts, 4)
        self.assertEqual(root.sampler.sampler_config.top_k, 2)
        self.assertFalse(
            root.branches[0].sampler.sampler_config.normalize_probabilities_flag
        )
        self.assertEqual(direction_template.num_experts, original_num_experts)
        self.assertEqual(direction_template.top_k, 1)
        self.assertTrue(
            set(map(id, root.sampler.parameters())).isdisjoint(
                map(id, root.branches[0].sampler.parameters())
            )
        )

    def test_only_selected_subtree_receives_task_gradient(self) -> None:
        terminal = self.tree_terminal(
            leaf_top_k=1,
            routing_tree_config=self.routing_tree_config(direction_top_k=(1,)),
        )
        terminal.eval()
        root = terminal.sampler.root
        root_bias = root.sampler.router.model.layers[0].model.bias_params
        selected_leaf_bias = (
            root.branches[0].sampler.router.model.layers[0].model.bias_params
        )
        with torch.no_grad():
            root_bias.fill_(0.0)
            root_bias[0] = 2.0
            selected_leaf_bias.fill_(0.0)
            selected_leaf_bias[0] = 2.0

        input_batch = torch.zeros(self.batch_size, self.input_dim)
        probabilities = terminal(input_batch)[1]
        probabilities.sum().backward()

        self.assertGreater(root_bias.grad.abs().sum().item(), 0.0)
        self.assertGreater(selected_leaf_bias.grad.abs().sum().item(), 0.0)
        for unselected_leaf in root.branches[1:]:
            self.assertTrue(
                all(
                    parameter.grad is None for parameter in unselected_leaf.parameters()
                )
            )

        analytical_gradient = root_bias.grad[0].detach().clone()
        finite_difference_epsilon = 1e-3
        with torch.no_grad():
            root_bias[0] += finite_difference_epsilon
            positive_output = terminal(input_batch)[1].sum()
            root_bias[0] -= 2 * finite_difference_epsilon
            negative_output = terminal(input_batch)[1].sum()
            root_bias[0] += finite_difference_epsilon
        finite_difference_gradient = (positive_output - negative_output) / (
            2 * finite_difference_epsilon
        )
        torch.testing.assert_close(
            analytical_gradient,
            finite_difference_gradient,
            rtol=2e-3,
            atol=2e-4,
        )

    def test_tree_integrates_with_cluster_dtype_and_growth(self) -> None:
        neuron_config = self.neuron_config()
        neuron_config.terminal_config.sampler_config.top_k = 1
        neuron_config.terminal_config.routing_tree_config = self.routing_tree_config(
            direction_top_k=(1,)
        )
        cluster = (
            NeuronClusterConfig(
                x_axis_total_neurons=3,
                y_axis_total_neurons=3,
                z_axis_total_neurons=1,
                initial_x_axis_total_neurons=1,
                initial_y_axis_total_neurons=1,
                initial_z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=1,
                neuron_config=neuron_config,
            )
            .build()
            .double()
        )

        output, auxiliary_loss = cluster(
            torch.randn(self.batch_size, self.input_dim, dtype=torch.double)
        )

        self.assertEqual(output.dtype, torch.double)
        self.assertEqual(auxiliary_loss.dtype, torch.double)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(cluster.cluster), 2)
        self.assertTrue(
            all(
                isinstance(neuron.terminal.sampler, TerminalRoutingTreeDelegate)
                for neuron in cluster.cluster.values()
            )
        )

    def test_tree_checkpoint_round_trip_and_nested_usage_reset(self) -> None:
        terminal = self.tree_terminal(
            leaf_top_k=1,
            routing_tree_config=self.routing_tree_config(direction_top_k=(1,)),
        )
        terminal.eval()
        root = terminal.sampler.root
        with torch.no_grad():
            root.sampler.router.model.layers[0].model.bias_params.fill_(-8.0)
            root.sampler.router.model.layers[0].model.bias_params[0] = 8.0

        input_batch = torch.randn(self.batch_size, self.input_dim)
        expected = terminal(input_batch)
        restored = self.tree_terminal(
            leaf_top_k=1,
            routing_tree_config=self.routing_tree_config(direction_top_k=(1,)),
        )
        restored.load_state_dict(terminal.state_dict())
        restored.eval()
        actual = restored(input_batch)
        torch.testing.assert_close(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

        tracker_manager = SamplerUsageTrackerManager()
        samplers = [
            module for module in terminal.modules() if isinstance(module, SamplerModel)
        ]
        for sampler in samplers:
            tracker = tracker_manager.attach(sampler)
            tracker.last_expert_usage_counts.fill_(7.0)
            tracker.last_expert_usage_mass.fill_(7.0)
        terminal(input_batch)
        unvisited_tracker = root.branches[1].sampler.usage_tracker
        self.assertEqual(unvisited_tracker.last_expert_usage_counts.sum().item(), 0.0)
        self.assertEqual(unvisited_tracker.last_expert_usage_mass.sum().item(), 0.0)

    def test_invalid_tree_fails_before_trainable_construction(self) -> None:
        invalid_configs = (
            self.routing_tree_config(branch_counts=(1,), direction_top_k=(1,)),
            self.routing_tree_config(branch_counts=(4,), direction_top_k=(5,)),
            self.routing_tree_config(
                depth=TerminalRoutingTreeDepthOptions.THREE,
                branch_counts=(2,),
                direction_top_k=(1,),
            ),
        )
        for routing_tree_config in invalid_configs:
            with self.subTest(config=routing_tree_config):
                config = self.terminal_config()
                config.routing_tree_config = routing_tree_config
                torch.manual_seed(991)
                rng_before = torch.random.get_rng_state().clone()
                with patch.object(SamplerConfig, "build") as sampler_build:
                    with self.assertRaises((TypeError, ValueError)):
                        config.build()
                    sampler_build.assert_not_called()
                torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_routerless_sampling_is_rejected_only_for_tree_mode(self) -> None:
        total_connections = self.terminal_total_connections()
        config = self.terminal_config(
            input_dim=total_connections,
            sampler_config=self.sampler_config(
                input_dim=total_connections,
                num_experts=total_connections,
                top_k=1,
                router_config=None,
            ),
        )
        config.routing_tree_config = self.routing_tree_config(direction_top_k=(1,))

        with self.assertRaisesRegex(ValueError, "routerless direct-logit"):
            config.build()


if __name__ == "__main__":
    unittest.main()
