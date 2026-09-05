import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from emperor.neuron import _optimizer_sync as optimizer_sync
from emperor.neuron import NeuronClusterConfig, NeuronClusterOptimizerSyncCallback
from emperor.neuron._optimizer_layout import (
    OPTIMIZER_LAYOUT_CHECKPOINT_KEY,
    NeuronOptimizerNamedLayout,
)
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
    def test_empty_checkpoint_failure_restores_pre_sync_optimizer_membership(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        module = _HostModule(cluster)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        trainer = SimpleNamespace(
            optimizers=[optimizer],
            lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
        )
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        before_ids = _optimizer_parameter_ids(optimizer)
        cluster.grow()
        with self.assertRaisesRegex(RuntimeError, "scheduler counts differ"):
            callback.on_load_checkpoint(
                trainer, module, {"optimizer_states": [], "lr_schedulers": []}
            )
        self.assertEqual(_optimizer_parameter_ids(optimizer), before_ids)

    def test_checkpoint_preparation_failure_restores_pre_sync_membership_and_retries(
        self,
    ) -> None:
        cluster = _DynamicCluster()
        module = _HostModule(cluster)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        trainer = SimpleNamespace(
            optimizers=[optimizer],
            lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
        )
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        before_ids = _optimizer_parameter_ids(optimizer)
        before_synced = {
            key: set(value) for key, value in callback._synced_param_ids.items()
        }
        cluster.grow()
        source_optimizer = torch.optim.SGD(module.parameters(), lr=0.03)
        saved_states = [source_optimizer.state_dict()]
        checkpoint = {
            "optimizer_states": saved_states,
            "lr_schedulers": [],
            OPTIMIZER_LAYOUT_CHECKPOINT_KEY: NeuronOptimizerNamedLayout.capture(
                module, [source_optimizer], saved_states
            ),
        }
        with self.assertRaisesRegex(RuntimeError, "scheduler counts differ"):
            callback.on_load_checkpoint(trainer, module, checkpoint)
        self.assertEqual(_optimizer_parameter_ids(optimizer), before_ids)
        self.assertEqual(callback._synced_param_ids, before_synced)
        self.assertFalse(callback._optimizer_load_hook_handles)
        checkpoint["lr_schedulers"] = [scheduler.state_dict()]
        callback.on_load_checkpoint(trainer, module, checkpoint)
        optimizer.load_state_dict(saved_states[0])
        scheduler.load_state_dict(checkpoint["lr_schedulers"][0])
        callback.on_train_start(trainer, module)
        self.assertEqual(
            _optimizer_parameter_ids(optimizer), {id(p) for p in module.parameters()}
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.03)

    def test_replacement_preserves_subset_ownership_through_aliases(self) -> None:
        cluster = _DynamicCluster()
        module = _HostModule(cluster)
        old_parameter = cluster.cluster["neuron_0_0_0"].terminal.weight
        optimizer = torch.optim.Adam([old_parameter], lr=0.02)
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = _callback_for(cluster)
        with self.assertWarns(UserWarning):
            callback.on_fit_start(trainer, module)
        replacement = _RoleNeuron()
        replacement.terminal.weight = replacement.nucleus.weight
        cluster.cluster["neuron_0_0_0"] = replacement
        callback.on_before_backward(trainer, module, None)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertIs(
            optimizer.param_groups[0]["params"][0], replacement.nucleus.weight
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.02)

    def test_replacement_split_inherits_shared_parameter_group(self) -> None:
        cluster = _DynamicCluster()
        old_neuron = cluster.cluster["neuron_0_0_0"]
        old_neuron.terminal.weight = old_neuron.nucleus.weight
        module = _HostModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.02)
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        cluster.cluster["neuron_0_0_0"] = _RoleNeuron()
        callback.on_before_backward(trainer, module, None)
        self.assertEqual(
            _optimizer_parameter_ids(optimizer), {id(p) for p in cluster.parameters()}
        )
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 2)

    def test_replacement_merge_rejects_conflicting_groups_before_mutation(self) -> None:
        cluster = _DynamicCluster()
        module = _HostModule(cluster)
        optimizer = torch.optim.Adam(
            [
                {"params": [parameter], "lr": learning_rate}
                for parameter, learning_rate in zip(
                    cluster.parameters(), (0.01, 0.02), strict=True
                )
            ]
        )
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        old_groups = tuple(optimizer.param_groups)
        old_lists = tuple(group["params"] for group in old_groups)
        old_ids = _optimizer_parameter_ids(optimizer)
        replacement = _RoleNeuron()
        replacement.terminal.weight = replacement.nucleus.weight
        cluster.cluster["neuron_0_0_0"] = replacement
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            callback.on_before_backward(trainer, module, None)
        self.assertEqual(_optimizer_parameter_ids(optimizer), old_ids)
        for index, group in enumerate(optimizer.param_groups):
            self.assertIs(group, old_groups[index])
            self.assertIs(group["params"], old_lists[index])

    def test_live_alias_split_keeps_both_roles_in_registered_order(self) -> None:
        cluster = _DynamicCluster()
        neuron = cluster.cluster["neuron_0_0_0"]
        neuron.terminal.weight = neuron.nucleus.weight
        module = _HostModule(cluster)
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(module.parameters()),
                    "param_names": [name for name, _ in module.named_parameters()],
                }
            ],
            lr=0.01,
        )
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        neuron.nucleus.weight = nn.Parameter(torch.ones_like(neuron.nucleus.weight))
        callback.on_before_backward(trainer, module, None)
        self.assertEqual(
            list(map(id, optimizer.param_groups[0]["params"])),
            list(map(id, module.parameters())),
        )
        self.assertEqual(
            optimizer.param_groups[0]["param_names"],
            [name for name, _ in module.named_parameters()],
        )

    def test_same_name_replacement_preserves_sole_exemplar_groups(self) -> None:
        cluster = _DynamicCluster()
        module = _HostModule(cluster)
        named_parameters = list(module.named_parameters())
        optimizer = torch.optim.Adam(
            [
                {"params": [parameter], "param_names": [name], "lr": learning_rate}
                for (name, parameter), learning_rate in zip(
                    named_parameters, (0.01, 0.02), strict=True
                )
            ]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        trainer = SimpleNamespace(
            optimizers=[optimizer],
            lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
        )
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        original_groups = tuple(optimizer.param_groups)
        old_parameters = tuple(cluster.parameters())
        for parameter in old_parameters:
            optimizer.state[parameter] = {"sentinel": torch.ones_like(parameter)}
        cluster.cluster["neuron_0_0_0"] = _RoleNeuron()

        callback.on_before_backward(trainer, module, None)

        for index, (name, parameter) in enumerate(module.named_parameters()):
            self.assertIs(optimizer.param_groups[index], original_groups[index])
            self.assertIs(optimizer.param_groups[index]["params"][0], parameter)
            self.assertEqual(optimizer.param_groups[index]["param_names"], [name])
            self.assertNotIn(parameter, optimizer.state)
            self.assertIn(id(parameter), callback._post_wrap_param_ids)
        self.assertTrue(
            all(parameter not in optimizer.state for parameter in old_parameters)
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups], [0.01, 0.02]
        )
        self.assertEqual(scheduler.base_lrs, [0.01, 0.02])

    def test_replacement_and_pruning_roll_back_together_before_retry(self) -> None:
        cluster = _DynamicCluster(neuron_count=2)
        module = _HostModule(cluster)
        optimizer = torch.optim.Adam(
            [
                {"params": list(neuron.parameters()), "lr": learning_rate}
                for neuron, learning_rate in zip(
                    cluster.cluster.values(), (0.01, 0.02), strict=True
                )
            ]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        trainer = SimpleNamespace(
            optimizers=[optimizer],
            lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
        )
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        original_groups = tuple(optimizer.param_groups)
        original_lists = tuple(group["params"] for group in original_groups)
        original_parameters = tuple(tuple(parameters) for parameters in original_lists)
        cluster.cluster["neuron_0_0_0"] = _RoleNeuron()
        del cluster.cluster["neuron_1_0_0"]
        with patch.object(
            optimizer_sync,
            "preflight_scheduler_group_removal",
            side_effect=RuntimeError("reject pruning"),
        ):
            with self.assertRaisesRegex(RuntimeError, "reject pruning"):
                callback.on_before_backward(trainer, module, None)
        for index, group in enumerate(optimizer.param_groups):
            self.assertIs(group, original_groups[index])
            self.assertIs(group["params"], original_lists[index])
            self.assertEqual(
                tuple(map(id, group["params"])),
                tuple(map(id, original_parameters[index])),
            )
        self.assertEqual(scheduler.base_lrs, [0.01, 0.02])
        callback.on_before_backward(trainer, module, None)
        self.assertEqual(
            _optimizer_parameter_ids(optimizer), {id(p) for p in cluster.parameters()}
        )
        self.assertEqual(scheduler.base_lrs, [0.01])

    def test_replacement_ties_keep_one_slot_without_retired_state(self) -> None:
        cluster = _DynamicCluster()
        module = _HostModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
        trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
        callback = _callback_for(cluster)
        callback.on_fit_start(trainer, module)
        replacement = _RoleNeuron()
        replacement.terminal.weight = replacement.nucleus.weight
        cluster.cluster["neuron_0_0_0"] = replacement
        callback.on_before_zero_grad(trainer, module, optimizer)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 1)
        self.assertIs(
            optimizer.param_groups[0]["params"][0], replacement.nucleus.weight
        )

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
