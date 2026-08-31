import unittest
from types import SimpleNamespace

import torch

from emperor.neuron._terminal.topology import _compile_terminal_routing_tree


def routing_tree_config(*, branch_counts: tuple[int, ...]):
    return SimpleNamespace(
        depth=SimpleNamespace(value=2),
        direction_branch_counts=branch_counts,
        direction_top_k=(1,),
    )


class TestTerminalRoutingTreeCompiler(unittest.TestCase):
    def test_balances_three_dimensions_and_covers_connections_once(self):
        coordinates = torch.cartesian_prod(
            torch.arange(3),
            torch.arange(3),
            torch.arange(2),
        )

        plan = _compile_terminal_routing_tree(
            coordinates,
            routing_tree_config(branch_counts=(12,)),
            leaf_top_k=1,
        )

        self.assertEqual(plan.root.subdivision, (3, 2, 2))
        assigned_indices = [
            index for child in plan.root.children for index in child.connection_indices
        ]
        self.assertEqual(sorted(assigned_indices), list(range(len(coordinates))))
        self.assertEqual(len(assigned_indices), len(set(assigned_indices)))
        self.assertEqual(len(plan.root.children), 12)

    def test_is_translation_invariant_and_prunes_empty_regions(self):
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
        tree_config = routing_tree_config(branch_counts=(8,))

        original_plan = _compile_terminal_routing_tree(
            sparse_coordinates,
            tree_config,
            leaf_top_k=1,
        )
        translated_plan = _compile_terminal_routing_tree(
            sparse_coordinates + torch.tensor([11, -7, 5]),
            tree_config,
            leaf_top_k=1,
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


if __name__ == "__main__":
    unittest.main()
