import copy
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.nn as nn

from emperor.neuron import (
    NeuronClusterConfig,
    TerminalConnectionShapeOptions,
    TerminalRangeOptions,
    TerminalRoutingTreeConfig,
    TerminalRoutingTreeDepthOptions,
)
from emperor.neuron._terminal.connection_topology import TargetCoordinateBuilder
from emperor.neuron._terminal.routing import RoutingTreeDelegate
from emperor.neuron._terminal.routing.node import RoutingTreeNode
from emperor.neuron._terminal.routing_tree_topology import (
    RoutingTreeCompiler,
)
from emperor.neuron._terminal.validation import RoutingTreeDelegateValidator, Validator
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
    def test_empty_tree_probabilities_keep_visited_sampler_gradients(self):
        terminal = self.tree_terminal(leaf_top_k=1)
        source = torch.empty(0, self.input_dim, requires_grad=True)
        _, probabilities, coordinates, _ = terminal(source)
        self.assertEqual(probabilities.shape, (0, terminal.sampler.root.output_width))
        self.assertEqual(coordinates.shape, (0, terminal.sampler.root.output_width, 3))
        probabilities.sum().backward()
        torch.testing.assert_close(source.grad, torch.zeros_like(source))

    def test_empty_tree_only_visits_root_at_both_depths_and_under_autocast(self):
        for depth, counts in (
            (TerminalRoutingTreeDepthOptions.TWO, (2,)),
            (TerminalRoutingTreeDepthOptions.THREE, (2, 2)),
        ):
            for autocast in (False, True):
                with self.subTest(depth=depth, autocast=autocast):
                    terminal = self.tree_terminal(
                        leaf_top_k=2,
                        routing_tree_config=self.routing_tree_config(
                            depth=depth,
                            branch_counts=counts,
                            direction_top_k=(1,) * len(counts),
                        ),
                    ).eval()
                    source = torch.empty(0, self.input_dim, requires_grad=True)
                    root = terminal.sampler.root
                    random_state = torch.get_rng_state().clone()
                    with patch.object(
                        root.branches[0],
                        "route",
                        side_effect=AssertionError("empty child visited"),
                    ):
                        with (
                            torch.autocast("cpu", dtype=torch.bfloat16)
                            if autocast
                            else nullcontext()
                        ):
                            _, probabilities, coordinates, loss = terminal(source)
                    torch.testing.assert_close(torch.get_rng_state(), random_state)
                    self.assertEqual(probabilities.shape, (0, root.output_width))
                    self.assertEqual(coordinates.shape, (0, root.output_width, 3))
                    self.assertEqual(coordinates.dtype, torch.long)
                    self.assertEqual(coordinates.device, source.device)
                    self.assertTrue(torch.isfinite(loss))
                    probabilities.sum().backward()
                    torch.testing.assert_close(source.grad, torch.zeros_like(source))
                    for parameter in root.sampler.parameters():
                        self.assertIsNotNone(parameter.grad)
                        torch.testing.assert_close(
                            parameter.grad, torch.zeros_like(parameter)
                        )
                    self.assertTrue(
                        all(
                            parameter.grad is None
                            for parameter in root.branches.parameters()
                        )
                    )

    def test_tree_promotes_direction_and_leaf_probabilities_without_detaching(self):
        def double_hidden(module, inputs, output):
            output.hidden = output.hidden.double()
            return output

        for depth, counts in (
            (TerminalRoutingTreeDepthOptions.TWO, (2,)),
            (TerminalRoutingTreeDepthOptions.THREE, (2, 2)),
        ):
            with self.subTest(depth=depth):
                torch.manual_seed(17)
                terminal = self.tree_terminal(
                    leaf_top_k=1,
                    routing_tree_config=self.routing_tree_config(
                        depth=depth,
                        branch_counts=counts,
                        direction_top_k=(1,) * len(counts),
                    ),
                ).eval()
                for node in terminal.sampler.root.modules():
                    if isinstance(node, RoutingTreeNode) and not node.branches:
                        node.sampler.router.model.register_forward_hook(double_hidden)
                reference = copy.deepcopy(terminal).double()
                source = torch.ones(2, self.input_dim, requires_grad=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    _, probabilities, coordinates, loss = terminal(source)
                self.assertEqual(probabilities.dtype, torch.float64)
                self.assertEqual(coordinates.dtype, torch.long)
                (probabilities.sum() + loss).backward()
                self.assertTrue(torch.isfinite(source.grad).all())
                visited_gradients = [
                    p.grad for p in terminal.parameters() if p.grad is not None
                ]
                self.assertTrue(visited_gradients)
                self.assertTrue(
                    all(
                        torch.isfinite(gradient).all() for gradient in visited_gradients
                    )
                )
                reference_source = source.detach().double().requires_grad_()
                _, reference_probabilities, reference_coordinates, reference_loss = (
                    reference(reference_source)
                )
                (reference_probabilities.sum() + reference_loss).backward()
                torch.testing.assert_close(
                    probabilities, reference_probabilities, rtol=0.03, atol=1e-5
                )
                torch.testing.assert_close(coordinates, reference_coordinates)
                torch.testing.assert_close(
                    source.grad.double(), reference_source.grad, rtol=0.05, atol=1e-5
                )

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

    def test_node_derives_independent_sampler_configs_without_construction(
        self,
    ) -> None:
        for num_topk_samples, expected_samples in (
            (None, None),
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 2),
        ):
            for noisy_topk_flag in (False, True):
                with self.subTest(
                    num_topk_samples=num_topk_samples,
                    noisy_topk_flag=noisy_topk_flag,
                ):
                    template = self.sampler_config(input_dim=8, num_experts=9, top_k=3)
                    template.num_topk_samples = num_topk_samples
                    template.noisy_topk_flag = noisy_topk_flag
                    template.router_config.noisy_topk_flag = not noisy_topk_flag
                    original_template = copy.deepcopy(template)
                    expected_config = copy.deepcopy(template)
                    expected_config.num_experts = 4
                    expected_config.top_k = 2
                    expected_config.num_topk_samples = expected_samples
                    expected_config.router_config.input_dim = 6
                    expected_config.router_config.num_experts = 4
                    expected_config.router_config.noisy_topk_flag = noisy_topk_flag
                    rng_before = torch.random.get_rng_state().clone()

                    with patch.object(SamplerConfig, "build") as sampler_build:
                        derived_configs = [
                            RoutingTreeNode.derive_sampler_config(
                                template,
                                input_dim=6,
                                num_experts=4,
                                top_k=2,
                            )
                            for _ in range(2)
                        ]
                        sampler_build.assert_not_called()

                    for derived_config in derived_configs:
                        self.assertEqual(derived_config, expected_config)
                        self.assertIsNot(derived_config, template)
                        self.assertIsNot(
                            derived_config.router_config, template.router_config
                        )
                        self.assertIsNot(
                            derived_config.router_config.model_config,
                            template.router_config.model_config,
                        )
                    self.assertIsNot(derived_configs[0], derived_configs[1])
                    self.assertIsNot(
                        derived_configs[0].router_config.model_config,
                        derived_configs[1].router_config.model_config,
                    )
                    self.assertEqual(template, original_template)
                    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_node_sampler_config_derivation_requires_a_router(self) -> None:
        template = self.sampler_config(router_config=None)
        original_template = copy.deepcopy(template)
        rng_before = torch.random.get_rng_state().clone()

        with patch.object(SamplerConfig, "build") as sampler_build:
            with self.assertRaisesRegex(
                ValueError,
                "Terminal routing trees require learned router_config values for "
                "both direction and connection sampler templates\\.",
            ):
                RoutingTreeNode.derive_sampler_config(
                    template,
                    input_dim=6,
                    num_experts=4,
                    top_k=2,
                )
            sampler_build.assert_not_called()

        self.assertEqual(template, original_template)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_composition_and_runtime_validate_same_plan_and_sampler_templates(
        self,
    ) -> None:
        validation_requests = []
        validate_plan = RoutingTreeDelegateValidator.validate_plan_sampler_configs

        def record_plan_validation(**request):
            validate_plan(**request)
            validation_requests.append(request)

        for direction_sampler_config in (None, self.sampler_config(top_k=1)):
            with self.subTest(direction_sampler_config=direction_sampler_config):
                validation_requests.clear()
                config = self.terminal_config()
                config.routing_tree_config = self.routing_tree_config(
                    direction_sampler_config=direction_sampler_config,
                )
                rng_before = torch.random.get_rng_state().clone()
                with patch.object(
                    RoutingTreeDelegateValidator,
                    "validate_plan_sampler_configs",
                    side_effect=record_plan_validation,
                ):
                    Validator.validate_config_composition(config)
                    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
                    terminal = config.build()

                composition_request, runtime_request = validation_requests
                self.assertEqual(
                    composition_request["routing_tree_plan"],
                    runtime_request["routing_tree_plan"],
                )
                self.assertIs(
                    runtime_request["routing_tree_plan"],
                    terminal.sampler.plan,
                )
                self.assertEqual(composition_request["input_dim"], terminal.input_dim)
                self.assertIs(
                    composition_request["leaf_sampler_config"],
                    config.sampler_config,
                )
                self.assertIs(
                    composition_request["leaf_sampler_config"],
                    runtime_request["leaf_sampler_config"],
                )
                self.assertIs(
                    composition_request["direction_sampler_config"],
                    direction_sampler_config or config.sampler_config,
                )
                self.assertIs(
                    composition_request["direction_sampler_config"],
                    runtime_request["direction_sampler_config"],
                )
                self.assertIs(
                    runtime_request["direction_sampler_config"],
                    terminal.sampler.direction_sampler_config,
                )

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

    def test_tree_config_validation_preserves_first_error_order(self) -> None:
        invalid_fields = (
            (
                "depth",
                2,
                TypeError,
                "routing_tree_config.depth must be a TerminalRoutingTreeDepthOptions, got int.",
            ),
            (
                "direction_branch_counts",
                [4],
                TypeError,
                "routing_tree_config.direction_branch_counts must be a tuple, got list.",
            ),
            (
                "direction_top_k",
                (0,),
                ValueError,
                "routing_tree_config.direction_top_k[0] must be a positive integer, received 0.",
            ),
            (
                "direction_sampler_config",
                "invalid",
                TypeError,
                "routing_tree_config.direction_sampler_config must be a SamplerConfig or None, got str.",
            ),
            (
                None,
                None,
                ValueError,
                "sampler_config.top_k must be a positive integer for a Terminal routing tree, received 0.",
            ),
        )
        for first_invalid_index, (field_name, _, error_type, message) in enumerate(
            invalid_fields
        ):
            with self.subTest(first_invalid_field=field_name):
                config = self.terminal_config(
                    sampler_config=self.sampler_config(top_k=0)
                )
                config.routing_tree_config = self.routing_tree_config()
                for invalid_field, value, _, _ in invalid_fields[first_invalid_index:]:
                    if invalid_field is not None:
                        setattr(config.routing_tree_config, invalid_field, value)
                config_before = copy.deepcopy(config)
                rng_before = torch.random.get_rng_state().clone()

                with self.assertRaises(error_type) as raised:
                    Validator.validate_routing_tree_config_fields(config)

                self.assertEqual(str(raised.exception), message)
                self.assertEqual(config, config_before)
                torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_plan_validation_checks_both_templates_before_node_constraints(
        self,
    ) -> None:
        config = self.terminal_config()
        routing_tree_config = self.routing_tree_config(direction_top_k=(5,))
        plan = RoutingTreeCompiler(
            TargetCoordinateBuilder(config).build(), routing_tree_config, 1
        ).compile()
        invalid_cases = (
            (
                True,
                True,
                "sampler_config.router_config must be a RouterConfig for Terminal routing trees; routerless direct-logit sampling is available only in flat mode.",
            ),
            (
                False,
                True,
                "routing_tree_config.direction_sampler_config.router_config must be a RouterConfig for Terminal routing trees; routerless direct-logit sampling is available only in flat mode.",
            ),
            (
                False,
                False,
                "Terminal routing tree internal node <root> contains 4 nonempty regions, fewer than direction_top_k[0]=5.",
            ),
        )
        for routerless_leaf, routerless_direction, message in invalid_cases:
            with self.subTest(
                routerless_leaf=routerless_leaf,
                routerless_direction=routerless_direction,
            ):
                leaf_config = self.sampler_config()
                direction_config = self.sampler_config()
                if routerless_leaf:
                    leaf_config.router_config = None
                if routerless_direction:
                    direction_config.router_config = None
                templates_before = copy.deepcopy((leaf_config, direction_config))
                rng_before = torch.random.get_rng_state().clone()

                with patch.object(SamplerConfig, "build") as sampler_build:
                    with self.assertRaises(ValueError) as raised:
                        RoutingTreeDelegateValidator.validate_plan_sampler_configs(
                            input_dim=config.input_dim,
                            leaf_sampler_config=leaf_config,
                            direction_sampler_config=direction_config,
                            routing_tree_plan=plan,
                        )
                    sampler_build.assert_not_called()

                self.assertEqual(str(raised.exception), message)
                self.assertEqual((leaf_config, direction_config), templates_before)
                torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_compiler_snapshots_inputs_as_public_topology_values(self) -> None:
        coordinates = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
            ]
        )
        routing_tree_config = self.routing_tree_config(
            branch_counts=(2,),
            direction_top_k=(1,),
        )
        compiler = RoutingTreeCompiler(
            neuron_connections=coordinates,
            routing_tree_config=routing_tree_config,
            leaf_top_k=1,
        )

        self.assertEqual(compiler.coordinate_rows, ((0, 0, 0), (1, 1, 1)))
        self.assertEqual(compiler.depth, 2)
        self.assertEqual(compiler.direction_branch_counts, (2,))
        self.assertEqual(compiler.direction_top_k, (1,))
        self.assertEqual(compiler.leaf_top_k, 1)

        coordinates.add_(10)
        routing_tree_config.depth = TerminalRoutingTreeDepthOptions.THREE
        routing_tree_config.direction_branch_counts = (2, 2)
        routing_tree_config.direction_top_k = (1, 1)

        plan = compiler.compile()

        self.assertEqual(plan.depth, 2)
        self.assertEqual(plan.direction_top_k, (1,))
        self.assertEqual(plan.root.bounds, ((0, 1), (0, 1), (0, 1)))

    def test_compiler_preserves_subdivision_scoring_and_axis_tie_breaks(self) -> None:
        subdivision_cases = (
            ((2, 2, 2), 2, (2, 1, 1)),
            ((2, 2, 2), 4, (2, 2, 1)),
            ((5, 3, 2), 8, (4, 2, 1)),
            ((1, 5, 1), 3, (1, 3, 1)),
            ((3, 3, 2), 12, (3, 2, 2)),
        )
        for axis_spans, branch_count, expected_subdivision in subdivision_cases:
            with self.subTest(axis_spans=axis_spans, branch_count=branch_count):
                coordinates = torch.cartesian_prod(
                    *(torch.arange(span) for span in axis_spans)
                )
                routing_tree_config = self.routing_tree_config(
                    branch_counts=(branch_count,),
                    direction_top_k=(1,),
                )
                plan = RoutingTreeCompiler(
                    coordinates, routing_tree_config, 1
                ).compile()

                self.assertEqual(plan.root.subdivision, expected_subdivision)

    def test_compiler_preserves_uneven_intervals_and_input_connection_order(
        self,
    ) -> None:
        coordinates = torch.tensor(
            [[0, 4, 0], [0, 0, 0], [0, 2, 0], [0, 1, 0], [0, 3, 0]]
        )
        routing_tree_config = self.routing_tree_config(
            branch_counts=(3,), direction_top_k=(1,)
        )

        plan = RoutingTreeCompiler(coordinates, routing_tree_config, 1).compile()

        self.assertEqual(plan.root.subdivision, (1, 3, 1))
        self.assertEqual(
            tuple(child.bounds for child in plan.root.children),
            (
                ((0, 0), (0, 1), (0, 0)),
                ((0, 0), (2, 3), (0, 0)),
                ((0, 0), (4, 4), (0, 0)),
            ),
        )
        self.assertEqual(
            tuple(child.connection_indices for child in plan.root.children),
            ((1, 3), (2, 4), (0,)),
        )
        self.assertEqual(
            tuple(child.path for child in plan.root.children), ((0,), (1,), (2,))
        )

    def test_compiler_numbers_nonempty_children_without_path_gaps(self) -> None:
        coordinates = torch.tensor([[0, 0, 0], [2, 2, 0]])
        routing_tree_config = self.routing_tree_config(
            branch_counts=(4,), direction_top_k=(1,)
        )

        plan = RoutingTreeCompiler(coordinates, routing_tree_config, 1).compile()

        self.assertEqual(plan.root.subdivision, (2, 2, 1))
        self.assertEqual(tuple(node.path for node in plan.walk()), ((), (0,), (1,)))
        self.assertEqual(
            tuple(child.connection_indices for child in plan.root.children),
            ((0,), (1,)),
        )
        self.assertEqual(
            tuple(child.bounds for child in plan.root.children),
            (((0, 1), (0, 1), (0, 0)), ((2, 2), (2, 2), (0, 0))),
        )

    def test_compiler_rejects_empty_connections_without_consuming_rng(self) -> None:
        coordinates = torch.empty((0, 3), dtype=torch.long)
        rng_before = torch.random.get_rng_state().clone()
        compiler = RoutingTreeCompiler(coordinates, self.routing_tree_config(), 1)

        with self.assertRaisesRegex(
            ValueError, "Terminal routing tree requires at least one connection\\."
        ):
            compiler.compile()

        self.assertEqual(compiler.coordinate_rows, ())
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

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
        plan = RoutingTreeCompiler(
            neuron_connections=coordinates,
            routing_tree_config=config.routing_tree_config,
            leaf_top_k=config.sampler_config.top_k,
        ).compile()

        self.assertEqual(plan.root.subdivision, (3, 2, 2))
        assigned_indices = [
            index for child in plan.root.children for index in child.connection_indices
        ]
        self.assertEqual(sorted(assigned_indices), list(range(len(coordinates))))
        self.assertEqual(len(assigned_indices), len(set(assigned_indices)))
        self.assertEqual(len(plan.root.children), 12)

    def test_compiler_covers_new_sparse_terminal_shapes_once(self):
        shape_connection_counts = (
            (TerminalConnectionShapeOptions.DIAGONAL, 25),
            (TerminalConnectionShapeOptions.CROSS_DIAGONAL, 45),
        )
        for connection_shape, expected_connection_count in shape_connection_counts:
            with self.subTest(connection_shape=connection_shape):
                terminal_config = self.terminal_config(
                    xy_axis_range=TerminalRangeOptions.FOUR,
                    z_axis_range=TerminalRangeOptions.TWO,
                    sampler_config=self.sampler_config(
                        num_experts=expected_connection_count,
                        top_k=1,
                    ),
                    connection_shape=connection_shape,
                )
                connections = TargetCoordinateBuilder(terminal_config).build()
                routing_tree_config = self.routing_tree_config(
                    branch_counts=(8,),
                    direction_top_k=(1,),
                )
                plan = RoutingTreeCompiler(
                    neuron_connections=connections,
                    routing_tree_config=routing_tree_config,
                    leaf_top_k=terminal_config.sampler_config.top_k,
                ).compile()

                assigned_connection_indices = [
                    connection_index
                    for child in plan.root.children
                    for connection_index in child.connection_indices
                ]
                assigned_coordinates = {
                    tuple(connections[connection_index].tolist())
                    for connection_index in assigned_connection_indices
                }
                expected_coordinates = {
                    tuple(connection.tolist()) for connection in connections
                }
                self.assertEqual(len(connections), expected_connection_count)
                self.assertEqual(
                    sorted(assigned_connection_indices),
                    list(range(expected_connection_count)),
                )
                self.assertEqual(
                    len(assigned_connection_indices),
                    len(set(assigned_connection_indices)),
                )
                self.assertEqual(assigned_coordinates, expected_coordinates)
                self.assertIn((1, -3, -1), assigned_coordinates)
                self.assertIn((1, 5, 3), assigned_coordinates)

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
        original_plan = RoutingTreeCompiler(
            neuron_connections=sparse_coordinates,
            routing_tree_config=tree_config,
            leaf_top_k=config.sampler_config.top_k,
        ).compile()
        translated_plan = RoutingTreeCompiler(
            neuron_connections=sparse_coordinates + torch.tensor([11, -7, 5]),
            routing_tree_config=tree_config,
            leaf_top_k=config.sampler_config.top_k,
        ).compile()

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
        self.assertIsInstance(terminal.sampler, RoutingTreeDelegate)
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

    def test_tree_maps_symmetric_z_candidates_back_to_absolute_coordinates(
        self,
    ) -> None:
        terminal = self.tree_terminal(
            leaf_top_k=2,
            routing_tree_config=self.routing_tree_config(direction_top_k=(1,)),
        )
        root = terminal.sampler.root
        selected_branch = root.branches[0]
        lower_global_index = self.terminal_connection_index(terminal, (1, 1, 0))
        upper_global_index = self.terminal_connection_index(terminal, (1, 1, 2))
        branch_global_indices = selected_branch.global_connection_indices.tolist()
        lower_local_index = branch_global_indices.index(lower_global_index)
        upper_local_index = branch_global_indices.index(upper_global_index)
        root.sampler = _ScriptedTreeSampler(
            num_experts=len(root.branches),
            probabilities=(1.0,),
            indices=(0,),
            auxiliary_loss=0.0,
        )
        selected_branch.sampler = _ScriptedTreeSampler(
            num_experts=len(branch_global_indices),
            probabilities=(0.6, 0.4),
            indices=(lower_local_index, upper_local_index),
            auxiliary_loss=0.0,
        )

        _, probabilities, selected_coordinates, _ = terminal(
            torch.zeros(self.batch_size, self.input_dim)
        )

        torch.testing.assert_close(
            probabilities,
            torch.tensor([[0.6, 0.4]]).expand(self.batch_size, -1),
        )
        torch.testing.assert_close(
            selected_coordinates,
            torch.tensor([[[1, 1, 0], [1, 1, 2]]]).expand(
                self.batch_size,
                -1,
                -1,
            ),
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

    def test_delegate_retains_connections_without_registering_a_second_buffer(
        self,
    ) -> None:
        terminal = self.tree_terminal()

        self.assertIsInstance(terminal.sampler, RoutingTreeDelegate)
        self.assertFalse(hasattr(terminal, "routing_tree_plan"))
        self.assertIs(
            terminal.sampler.neuron_connections,
            terminal.neuron_connections,
        )
        self.assertNotIn(
            "neuron_connections",
            dict(terminal.sampler.named_buffers()),
        )

    def test_delegate_requires_routing_tree_config_before_initialization(self) -> None:
        config = self.terminal_config()
        candidate_coordinates = TargetCoordinateBuilder(config).build()

        with self.assertRaisesRegex(
            ValueError,
            "RoutingTreeDelegate requires routing_tree_config",
        ):
            RoutingTreeDelegate(config, candidate_coordinates)

    def test_plan_compilation_uses_initialized_fields_instead_of_live_config(
        self,
    ) -> None:
        terminal = self.tree_terminal()
        delegate = terminal.sampler
        expected_plan = delegate.plan
        expected_direction_sampler_config = delegate.direction_sampler_config
        delegate.cfg.input_dim = None
        delegate.cfg.sampler_config = None
        delegate.cfg.routing_tree_config = None
        delegate.routing_tree_config.direction_sampler_config = self.sampler_config(
            router_config=None,
        )

        actual_plan = delegate.compile_routing_tree_plan()

        self.assertEqual(actual_plan, expected_plan)
        self.assertIs(
            delegate.direction_sampler_config,
            expected_direction_sampler_config,
        )

    def test_plan_compilation_passes_delegate_and_plan_to_validator(self) -> None:
        terminal = self.tree_terminal()
        delegate = terminal.sampler
        validator_arguments = {}

        class TrackingValidator(RoutingTreeDelegateValidator):
            @classmethod
            def validate_routing_tree_plan(cls, model, routing_tree_plan):
                validator_arguments["model"] = model
                validator_arguments["routing_tree_plan"] = routing_tree_plan

        with patch.object(RoutingTreeDelegate, "VALIDATOR", TrackingValidator):
            routing_tree_plan = delegate.compile_routing_tree_plan()

        self.assertIs(validator_arguments["model"], delegate)
        self.assertIs(
            validator_arguments["routing_tree_plan"],
            routing_tree_plan,
        )

    def test_delegate_validates_sampling_inputs(self) -> None:
        delegate = self.tree_terminal().sampler

        with self.assertRaisesRegex(TypeError, "input_matrix must be a Tensor"):
            delegate.sample_probabilities_and_indices([[1.0]])

        invalid_input = torch.zeros(self.batch_size, self.input_dim + 1)
        with self.assertRaisesRegex(
            ValueError,
            "Terminal routing tree input must have shape",
        ):
            delegate.sample_probabilities_and_indices(invalid_input)

        valid_input = torch.zeros(self.batch_size, self.input_dim)
        with self.assertRaisesRegex(ValueError, "shared skip_mask"):
            delegate.sample_probabilities_and_indices(
                valid_input,
                skip_mask=torch.ones(self.batch_size, dtype=torch.bool),
            )

    def test_missing_direction_sampler_config_uses_leaf_sampler_config(self) -> None:
        delegate = self.tree_terminal().sampler

        self.assertIs(
            delegate.direction_sampler_config,
            delegate.leaf_sampler_config,
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
        self.assertIs(terminal.sampler.direction_sampler_config, direction_template)
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
                isinstance(neuron.terminal.sampler, RoutingTreeDelegate)
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
                for operation in (
                    lambda config=config: Validator.validate_config_composition(config),
                    config.build,
                ):
                    torch.manual_seed(991)
                    rng_before = torch.random.get_rng_state().clone()
                    with patch.object(SamplerConfig, "build") as sampler_build:
                        with self.assertRaises((TypeError, ValueError)):
                            operation()
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
