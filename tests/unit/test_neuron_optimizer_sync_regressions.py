import unittest
from types import SimpleNamespace

import torch
from torch import nn

from emperor.neuron import NeuronClusterOptimizerSyncCallback


class _RoleNeuron(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nucleus = nn.Linear(1, 1, bias=False)
        self.terminal = nn.Linear(1, 1, bias=False)


class _DynamicCluster(nn.Module):
    def __init__(self, neuron_count: int = 1) -> None:
        super().__init__()
        self.cluster = nn.ModuleDict(
            {
                f"neuron_{index}_0_0": _RoleNeuron()
                for index in range(neuron_count)
            }
        )
        self._checkpoint_removed_parameter_ids: set[int] = set()

    def grow(self) -> _RoleNeuron:
        neuron = _RoleNeuron()
        self.cluster[f"neuron_{len(self.cluster)}_0_0"] = neuron
        return neuron


class _HostModule(nn.Module):
    def __init__(self, cluster: _DynamicCluster) -> None:
        super().__init__()
        self.neuron_cluster = cluster


def _callback_for(cluster: _DynamicCluster) -> NeuronClusterOptimizerSyncCallback:
    callback = NeuronClusterOptimizerSyncCallback()
    callback._NeuronClusterOptimizerSyncCallback__find_neuron_clusters = (
        lambda module: [cluster]
    )
    return callback


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


class TestNeuronOptimizerSyncRegressions(unittest.TestCase):
    def test_growth_inherits_the_existing_group_for_each_parameter_role(self) -> None:
        cluster = _DynamicCluster()
        module = _HostModule(cluster)
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(initial_neuron.nucleus.parameters()),
                    "lr": 0.001,
                    "weight_decay": 0.01,
                },
                {
                    "params": list(initial_neuron.terminal.parameters()),
                    "lr": 0.002,
                    "weight_decay": 0.02,
                },
            ]
        )
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)

        grown_neuron = cluster.grow()
        callback.sync_optimizers(trainer, module)

        nucleus_parameter = grown_neuron.nucleus.weight
        terminal_parameter = grown_neuron.terminal.weight
        self.assertIn(nucleus_parameter, optimizer.param_groups[0]["params"])
        self.assertNotIn(nucleus_parameter, optimizer.param_groups[1]["params"])
        self.assertIn(terminal_parameter, optimizer.param_groups[1]["params"])
        self.assertNotIn(terminal_parameter, optimizer.param_groups[0]["params"])
        self.assertEqual(
            [
                (group["lr"], group["weight_decay"])
                for group in optimizer.param_groups
            ],
            [(0.001, 0.01), (0.002, 0.02)],
        )

    def test_checkpoint_removed_parameters_and_their_state_are_pruned(self) -> None:
        cluster = _DynamicCluster(neuron_count=2)
        module = _HostModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        removed_neuron = cluster.cluster["neuron_1_0_0"]
        removed_parameters = list(removed_neuron.parameters())
        removed_parameter_ids = {id(parameter) for parameter in removed_parameters}
        retained_parameter = cluster.cluster["neuron_0_0_0"].nucleus.weight
        retained_state = {"sentinel": torch.ones_like(retained_parameter)}
        optimizer.state[retained_parameter] = retained_state
        for parameter in removed_parameters:
            optimizer.state[parameter] = {"sentinel": torch.zeros_like(parameter)}

        cluster._checkpoint_removed_parameter_ids.update(removed_parameter_ids)
        del cluster.cluster["neuron_1_0_0"]
        callback = _callback_for(cluster)
        callback.sync_optimizers(trainer, module)

        self.assertTrue(
            removed_parameter_ids.isdisjoint(_optimizer_parameter_ids(optimizer))
        )
        self.assertTrue(
            all(
                id(parameter) not in removed_parameter_ids
                for parameter in optimizer.state
            )
        )
        self.assertIs(optimizer.state[retained_parameter], retained_state)
        self.assertFalse(cluster._checkpoint_removed_parameter_ids)

    def test_late_group_validation_failure_rolls_back_and_can_retry(self) -> None:
        cluster = _DynamicCluster()
        module = _HostModule(cluster)
        names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        initial_neuron = cluster.cluster["neuron_0_0_0"]
        nucleus_parameters = list(initial_neuron.nucleus.parameters())
        terminal_parameters = list(initial_neuron.terminal.parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": nucleus_parameters,
                    "param_names": [
                        names_by_parameter_id[id(parameter)]
                        for parameter in nucleus_parameters
                    ],
                },
                {
                    "params": terminal_parameters,
                    "param_names": [
                        names_by_parameter_id[id(parameter)]
                        for parameter in terminal_parameters
                    ],
                },
            ],
            lr=0.01,
        )
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        malformed_names = optimizer.param_groups[1]["param_names"]
        missing_terminal_name = malformed_names.pop()
        original_param_lists = [group["params"] for group in optimizer.param_groups]
        original_param_contents = [tuple(params) for params in original_param_lists]
        original_name_lists = [
            group["param_names"] for group in optimizer.param_groups
        ]
        original_name_contents = [tuple(names) for names in original_name_lists]
        grown_neuron = cluster.grow()
        grown_parameter_ids = {
            id(parameter) for parameter in grown_neuron.parameters()
        }

        with self.assertRaisesRegex(RuntimeError, "param_names"):
            callback.sync_optimizers(trainer, module)

        for index, group in enumerate(optimizer.param_groups):
            self.assertIs(group["params"], original_param_lists[index])
            self.assertEqual(tuple(group["params"]), original_param_contents[index])
            self.assertIs(group["param_names"], original_name_lists[index])
            self.assertEqual(
                tuple(group["param_names"]), original_name_contents[index]
            )
        self.assertTrue(
            grown_parameter_ids.isdisjoint(_optimizer_parameter_ids(optimizer))
        )

        malformed_names.append(missing_terminal_name)
        callback.sync_optimizers(trainer, module)

        self.assertTrue(
            grown_parameter_ids.issubset(_optimizer_parameter_ids(optimizer))
        )
        current_names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        for group in optimizer.param_groups:
            self.assertEqual(
                group["param_names"],
                [
                    current_names_by_parameter_id[id(parameter)]
                    for parameter in group["params"]
                ],
            )


if __name__ == "__main__":
    unittest.main()
