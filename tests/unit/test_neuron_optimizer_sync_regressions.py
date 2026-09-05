import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from emperor.neuron import _optimizer_sync as optimizer_sync
from emperor.neuron import NeuronClusterConfig, NeuronClusterOptimizerSyncCallback
from unit.test_neuron import NeuronTestCase


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
    def test_ddp_registration_is_independent_of_live_optimizer_membership(self) -> None:
        cluster = _DynamicCluster()
        module = _HostModule(cluster)
        neuron = cluster.cluster["neuron_0_0_0"]
        neuron.terminal.weight.requires_grad_(False)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        self.assertIn(id(neuron.nucleus.weight), callback._ddp_registered_param_ids)
        self.assertNotIn(id(neuron.terminal.weight), callback._ddp_registered_param_ids)
        neuron.terminal.weight.requires_grad_(True)
        callback.on_before_backward(trainer, module, None)
        self.assertIn(id(neuron.terminal.weight), callback._post_wrap_param_ids)
        del cluster.cluster["neuron_0_0_0"]
        callback.sync_optimizers(trainer, module)
        cluster.cluster["neuron_0_0_0"] = neuron
        callback.sync_optimizers(trainer, module)
        self.assertNotIn(id(neuron.nucleus.weight), callback._post_wrap_param_ids)

    @pytest.mark.training
    def test_second_forward_grown_parameter_participates_in_current_update(
        self,
    ) -> None:
        torch.manual_seed(17)
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            max_total_growths=1,
            neuron_config=NeuronTestCase().full_sampler_neuron_config(),
        ).build()
        module = _HostModule(cluster)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        first_output, first_loss = cluster(source)
        second_output, second_loss = cluster(source / 10)
        loss = (
            first_output.square().mean()
            + second_output.square().mean()
            + first_loss
            + second_loss
        )
        child_parameter = cluster.cluster["neuron_2_1_1"].nucleus.model.weight
        callback.on_before_zero_grad(trainer, module, optimizer)
        optimizer.zero_grad(set_to_none=True)
        callback.on_before_backward(trainer, module, loss)
        loss.backward()
        self.assertGreater(float(child_parameter.grad.norm()), 0.0)
        self.assertIn(id(child_parameter), _optimizer_parameter_ids(optimizer))
        self.assertIn(id(child_parameter), callback._post_wrap_param_ids)
        expected = child_parameter.detach() - 0.01 * child_parameter.grad
        optimizer.step()
        torch.testing.assert_close(child_parameter, expected)

    def __assert_scheduler_preflight_order(self, fail_second_preflight: bool) -> None:
        cluster = _DynamicCluster(neuron_count=2)
        module = _HostModule(cluster)
        retained_neuron, removed_neuron = cluster.cluster.values()
        optimizer = torch.optim.Adam(
            [
                {"params": list(retained_neuron.parameters())},
                {"params": list(removed_neuron.parameters())},
            ],
            lr=0.01,
        )
        schedulers = [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
            for step_size in (2, 3)
        ]
        trainer = SimpleNamespace(
            optimizers=[optimizer],
            lr_scheduler_configs=[
                SimpleNamespace(scheduler=scheduler) for scheduler in schedulers
            ],
        )
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        removed_parameter = removed_neuron.nucleus.weight
        optimizer.state[removed_parameter] = {"sentinel": 1}
        original_groups = tuple(optimizer.param_groups)
        original_parameter_lists = [group["params"] for group in original_groups]
        original_state = optimizer.state
        del cluster.cluster["neuron_1_0_0"]
        events = []
        preflight = optimizer_sync.preflight_scheduler_group_removal
        remove_groups = optimizer_sync.remove_scheduler_groups

        def record_preflight(scheduler, indices, *, previous_group_count):
            events.append(("preflight", schedulers.index(scheduler)))
            self.assertEqual(indices, (1,))
            self.assertEqual(previous_group_count, 2)
            self.assertEqual(tuple(optimizer.param_groups), original_groups)
            for index, group in enumerate(optimizer.param_groups):
                self.assertIs(group["params"], original_parameter_lists[index])
            self.assertIn(removed_parameter, optimizer.state)
            if fail_second_preflight and scheduler is schedulers[1]:
                raise RuntimeError("second scheduler rejected pruning")
            preflight(scheduler, indices, previous_group_count=previous_group_count)

        def record_removal(scheduler, indices, *, previous_group_count):
            events.append(("remove", schedulers.index(scheduler)))
            self.assertEqual(len(optimizer.param_groups), 1)
            self.assertIs(optimizer.param_groups[0], original_groups[0])
            self.assertNotIn(removed_parameter, optimizer.state)
            remove_groups(scheduler, indices, previous_group_count=previous_group_count)

        with (
            patch.object(
                optimizer_sync,
                "preflight_scheduler_group_removal",
                record_preflight,
            ),
            patch.object(optimizer_sync, "remove_scheduler_groups", record_removal),
        ):
            if fail_second_preflight:
                with self.assertRaisesRegex(RuntimeError, "second scheduler"):
                    callback.sync_optimizers(trainer, module)
            else:
                callback.sync_optimizers(trainer, module)

        expected_events = [("preflight", 0), ("preflight", 1)]
        if fail_second_preflight:
            for index, group in enumerate(optimizer.param_groups):
                self.assertIs(group, original_groups[index])
                self.assertIs(group["params"], original_parameter_lists[index])
            self.assertIs(optimizer.state, original_state)
            self.assertIn(removed_parameter, optimizer.state)
        else:
            expected_events.extend([("remove", 0), ("remove", 1)])
        self.assertEqual(events, expected_events)

    def test_scheduler_preflights_precede_all_pruning_mutations(self) -> None:
        for fail_second_preflight in (False, True):
            with self.subTest(fail_second_preflight=fail_second_preflight):
                self.__assert_scheduler_preflight_order(fail_second_preflight)

    def test_synchronization_parameter_traversal_is_linear(self) -> None:
        for neuron_count in (10, 100):
            with self.subTest(neuron_count=neuron_count):
                cluster = _DynamicCluster(neuron_count)
                module = _HostModule(cluster)
                optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
                trainer = SimpleNamespace(
                    optimizers=[optimizer], lr_scheduler_configs=[]
                )
                callback = _callback_for(cluster)
                callback.on_fit_start(trainer, module)
                parameters = tuple(cluster.parameters())
                parameter_visits = 0

                def counted_parameters(*args, snapshot=parameters, **kwargs):
                    nonlocal parameter_visits
                    for parameter in snapshot:
                        parameter_visits += 1
                        yield parameter

                with patch.object(cluster, "parameters", counted_parameters):
                    callback.sync_optimizers(trainer, module)

                self.assertLessEqual(parameter_visits, 4 * len(parameters))
                self.assertEqual(
                    _optimizer_parameter_ids(optimizer), {id(p) for p in parameters}
                )
                self.assertFalse(callback._post_wrap_param_ids)

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
