import copy
import unittest
import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn
from lightning.pytorch.trainer.states import TrainerFn

from emperor.neuron import (
    NeuronClusterConfig,
    NeuronClusterOptimizerSyncCallback,
)
from emperor.neuron._optimizer_checkpoint import (
    LegacyOptimizerAppendPolicy,
    NeuronOptimizerCheckpointReconciler,
)
from emperor.neuron._optimizer_layout import (
    OPTIMIZER_LAYOUT_CHECKPOINT_KEY,
    NeuronOptimizerNamedLayout,
)
from emperor.neuron._optimizer_scheduler import (
    NeuronSchedulerCheckpointReconciler,
    NeuronSchedulerMutationTransaction,
    SchedulerGroupLoadBinding,
    reconcile_scheduler_group_count,
)
from unit.test_neuron import NeuronTestCase


class FakeLightningModule(nn.Module):
    def __init__(self, cluster):
        super().__init__()
        self.neuron_cluster = cluster
        self.other = nn.Linear(cluster.input_dim, cluster.input_dim)


class PrefixLightningModule(nn.Module):
    def __init__(self, cluster):
        super().__init__()
        self.other = nn.Linear(cluster.input_dim, cluster.input_dim)
        self.neuron_cluster = cluster


class AliasedClusterLightningModule(nn.Module):
    def __init__(self, cluster):
        super().__init__()
        self.primary_cluster = cluster
        self.alias_cluster = cluster
        self.other = nn.Linear(cluster.input_dim, cluster.input_dim)


class TwoClusterLightningModule(nn.Module):
    def __init__(self, first_cluster, second_cluster):
        super().__init__()
        self.first_cluster = first_cluster
        self.second_cluster = second_cluster
        self.other = nn.Linear(first_cluster.input_dim, first_cluster.input_dim)


class HiddenParameterLightningModule(FakeLightningModule):
    def __init__(self, cluster):
        super().__init__(cluster)
        self.hidden_parameter_ids: set[int] = set()

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        return (
            (name, parameter)
            for name, parameter in super().named_parameters(
                prefix=prefix,
                recurse=recurse,
                remove_duplicate=remove_duplicate,
            )
            if id(parameter) not in self.hidden_parameter_ids
        )


class FakeTrainer:
    def __init__(self, optimizers):
        self.optimizers = optimizers


class UnsupportedLambdaLR(torch.optim.lr_scheduler.LambdaLR):
    pass


class TestNeuronClusterOptimizerSyncCallback(NeuronTestCase):
    def build_growing_cluster(self):
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=4,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

    def grow_once(self, cluster):
        initial_keys = set(cluster.cluster.keys())
        self.assertEqual(len(initial_keys), 4)

        cluster.train()
        cluster(torch.randn(self.batch_size, self.input_dim))

        added_keys = set(cluster.cluster.keys()) - initial_keys
        self.assertEqual(len(cluster.cluster), 5)
        self.assertEqual(len(added_keys), 1)
        return cluster.cluster[added_keys.pop()]

    def optimizer_param_ids(self, optimizer) -> set[int]:
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    def optimizer_identity_snapshot(self, optimizer):
        return SimpleNamespace(
            param_groups=optimizer.param_groups,
            groups=tuple(optimizer.param_groups),
            group_values=tuple(dict(group) for group in optimizer.param_groups),
            group_lists=tuple(
                {
                    name: (value, tuple(value))
                    for name, value in group.items()
                    if isinstance(value, list)
                }
                for group in optimizer.param_groups
            ),
            state=optimizer.state,
            state_items=tuple(optimizer.state.items()),
        )

    def assert_optimizer_matches_identity_snapshot(
        self,
        optimizer,
        snapshot,
    ) -> None:
        self.assertIs(optimizer.param_groups, snapshot.param_groups)
        self.assertEqual(len(optimizer.param_groups), len(snapshot.groups))
        for group, original_group, original_values, original_lists in zip(
            optimizer.param_groups,
            snapshot.groups,
            snapshot.group_values,
            snapshot.group_lists,
            strict=True,
        ):
            self.assertIs(group, original_group)
            self.assertEqual(set(group), set(original_values))
            for name, original_value in original_values.items():
                if name not in original_lists:
                    self.assertIs(group[name], original_value)
                    continue
                original_list, original_items = original_lists[name]
                self.assertIs(group[name], original_list)
                self.assertEqual(len(group[name]), len(original_items))
                for value, original_item in zip(
                    group[name],
                    original_items,
                    strict=True,
                ):
                    if isinstance(original_item, (torch.Tensor, nn.Parameter)):
                        self.assertIs(value, original_item)
                    else:
                        self.assertEqual(value, original_item)
        self.assertIs(optimizer.state, snapshot.state)
        self.assertEqual(len(optimizer.state), len(snapshot.state_items))
        for parameter, value in snapshot.state_items:
            self.assertIs(optimizer.state[parameter], value)

    def scheduler_identity_snapshot(self, scheduler):
        return SimpleNamespace(
            values=dict(scheduler.__dict__),
            lists={
                name: (value, tuple(value))
                for name, value in scheduler.__dict__.items()
                if isinstance(value, list)
            },
        )

    def assert_scheduler_matches_identity_snapshot(
        self,
        scheduler,
        snapshot,
    ) -> None:
        self.assertEqual(set(scheduler.__dict__), set(snapshot.values))
        for name, original_value in snapshot.values.items():
            if name not in snapshot.lists:
                self.assertIs(scheduler.__dict__[name], original_value)
                continue
            original_list, original_items = snapshot.lists[name]
            self.assertIs(scheduler.__dict__[name], original_list)
            self.assertEqual(len(original_list), len(original_items))
            for value, original_item in zip(
                original_list,
                original_items,
                strict=True,
            ):
                self.assertIs(value, original_item)

    def legacy_state_for_one_grown_neuron(self, module, optimizer, grown_neuron):
        grown_parameter_ids = {id(parameter) for parameter in grown_neuron.parameters()}
        base_parameters = [
            parameter
            for parameter in optimizer.param_groups[0]["params"]
            if id(parameter) not in grown_parameter_ids
        ]
        dynamic_parameters = list(grown_neuron.parameters())
        serialized_parameters = base_parameters + dynamic_parameters
        base_group = {
            **{
                name: value
                for name, value in optimizer.param_groups[0].items()
                if name != "params"
            },
            "params": list(range(len(base_parameters))),
        }
        dynamic_group = {
            **{
                name: value
                for name, value in optimizer.param_groups[0].items()
                if name != "params"
            },
            "params": list(range(len(base_parameters), len(serialized_parameters))),
        }
        names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        expected_state = {}
        serialized_state = {}
        for index, parameter in enumerate(serialized_parameters, start=1):
            sentinel = torch.full_like(parameter, index / 100.0)
            expected_state[names_by_parameter_id[id(parameter)]] = sentinel
            serialized_state[index - 1] = {
                "step": torch.tensor(3.0),
                "exp_avg": sentinel.clone(),
                "exp_avg_sq": torch.full_like(parameter, index / 100.0 + 1.0),
            }
        return {
            "state": serialized_state,
            "param_groups": [base_group, dynamic_group],
        }, expected_state

    def test_sync_adds_grown_neuron_params_to_cluster_optimizer(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=0.012,
            weight_decay=0.034,
        )
        trainer = FakeTrainer([optimizer])

        new_neuron = self.grow_once(cluster)
        new_params = [
            parameter
            for parameter in new_neuron.parameters()
            if parameter.requires_grad
        ]
        self.assertGreater(len(new_params), 0)
        before_sync_param_ids = self.optimizer_param_ids(optimizer)
        self.assertTrue(
            all(id(parameter) not in before_sync_param_ids for parameter in new_params)
        )

        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        after_sync_param_ids = self.optimizer_param_ids(optimizer)
        self.assertTrue(
            all(id(parameter) in after_sync_param_ids for parameter in new_params)
        )

    def test_frozen_growth_stays_optimizer_owned_across_later_unfreeze(self):
        cluster = self.build_growing_cluster()
        cluster.requires_grad_(False)
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.125)
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)

        grown_neuron = self.grow_once(cluster)
        grown_parameters = list(grown_neuron.parameters())
        self.assertTrue(grown_parameters)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in grown_parameters)
        )
        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        optimized_parameter_ids = self.optimizer_param_ids(optimizer)
        self.assertTrue(
            all(
                id(parameter) in optimized_parameter_ids
                for parameter in grown_parameters
            )
        )
        frozen_snapshots = [
            parameter.detach().clone() for parameter in grown_parameters
        ]
        optimizer.step()
        for parameter, snapshot in zip(
            grown_parameters,
            frozen_snapshots,
            strict=True,
        ):
            torch.testing.assert_close(parameter, snapshot)

        cluster.requires_grad_(True)
        callback.on_train_batch_start(
            trainer,
            module,
            batch=None,
            batch_idx=1,
        )
        updated_parameter = grown_neuron.nucleus.model.weight
        before_update = updated_parameter.detach().clone()
        updated_parameter.grad = torch.ones_like(updated_parameter)
        optimizer.step()

        torch.testing.assert_close(
            updated_parameter,
            before_update - 0.125,
            rtol=0.0,
            atol=0.0,
        )

    def test_sync_uses_all_alias_roles_to_locate_grown_parameter_group(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        existing_router_weights = [
            neuron.terminal.sampler.router.model.layers[0].model.weight_params
            for neuron in cluster.cluster.values()
        ]
        existing_router_weight_ids = {
            id(parameter) for parameter in existing_router_weights
        }
        optimizer = torch.optim.Adam(
            [
                {"params": existing_router_weights},
                {
                    "params": [
                        parameter
                        for parameter in module.parameters()
                        if id(parameter) not in existing_router_weight_ids
                    ]
                },
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        grown_neuron = self.grow_once(cluster)
        grown_router_weight = grown_neuron.terminal.sampler.router.model.layers[
            0
        ].model.weight_params
        grown_neuron.nucleus.model.router_weight_alias = grown_router_weight
        canonical_grown_names = {
            name
            for name, parameter in cluster.named_parameters()
            if parameter is grown_router_weight
        }
        all_grown_names = {
            name
            for name, parameter in cluster.named_parameters(remove_duplicate=False)
            if parameter is grown_router_weight
        }
        self.assertEqual(
            canonical_grown_names,
            {"cluster.neuron_5_1_1.nucleus.model.router_weight_alias"},
        )
        self.assertIn(
            "cluster.neuron_5_1_1.terminal.sampler.router.model.layers.0."
            "model.weight_params",
            all_grown_names,
        )

        callback.sync_optimizers(trainer, module)

        synchronized_parameters = optimizer.param_groups[0]["params"]
        self.assertEqual(
            sum(
                parameter is grown_router_weight
                for parameter in synchronized_parameters
            ),
            1,
        )

    def test_sync_keeps_official_optimizer_param_names_aligned_after_growth(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        named_parameters = list(module.named_parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameter for _, parameter in named_parameters],
                    "param_names": [name for name, _ in named_parameters],
                }
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        self.grow_once(cluster)

        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        expected_names = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        group = optimizer.param_groups[0]
        self.assertEqual(len(group["params"]), len(group["param_names"]))
        self.assertEqual(
            group["param_names"],
            [expected_names[id(parameter)] for parameter in group["params"]],
        )

    def test_sync_is_idempotent(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        self.grow_once(cluster)
        callback = NeuronClusterOptimizerSyncCallback()

        callback.sync_optimizers(trainer, module)
        param_group_count = len(optimizer.param_groups)
        param_count = sum(len(group["params"]) for group in optimizer.param_groups)
        callback.sync_optimizers(trainer, module)

        self.assertEqual(len(optimizer.param_groups), param_group_count)
        self.assertEqual(
            sum(len(group["params"]) for group in optimizer.param_groups),
            param_count,
        )

    def test_legacy_groups_and_named_resave_preserve_exact_custom_order(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        grown_neuron = self.grow_once(cluster)
        optimizer = torch.optim.Adam(
            reversed(list(module.parameters())),
            lr=0.012,
        )
        original_order = list(optimizer.param_groups[0]["params"])
        grown_parameter_ids = {id(parameter) for parameter in grown_neuron.parameters()}
        expected_legacy_groups = [
            [
                parameter
                for parameter in original_order
                if id(parameter) not in grown_parameter_ids
            ],
            list(grown_neuron.parameters()),
        ]
        legacy_state, expected_state = self.legacy_state_for_one_grown_neuron(
            module,
            optimizer,
            grown_neuron,
        )
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load([optimizer], [cluster], [legacy_state])
        self.assertEqual(len(optimizer.param_groups), 2)
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        self.assertEqual(
            [group["params"] for group in optimizer.param_groups],
            expected_legacy_groups,
        )
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(
                optimizer.state[parameter]["exp_avg"],
                expected_state[name],
            )

        named_state = optimizer.state_dict()
        named_layout = NeuronOptimizerNamedLayout.capture(
            module,
            [optimizer],
            [named_state],
            {id(optimizer): LegacyOptimizerAppendPolicy(1, 0)},
        )
        next_optimizer = torch.optim.Adam(
            reversed(list(module.parameters())),
            lr=0.012,
        )
        layout_manager = NeuronOptimizerNamedLayout()
        layout_manager.prepare_for_load(
            module,
            [next_optimizer],
            [named_state],
            named_layout,
        )
        next_optimizer.load_state_dict(named_state)
        self.assertTrue(layout_manager.complete_optimizer_load(next_optimizer))
        self.assertEqual(len(next_optimizer.param_groups), 2)
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(
                next_optimizer.state[parameter]["exp_avg"],
                expected_state[name],
            )

    def test_legacy_load_preserves_frozen_dynamic_optimizer_membership(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        grown_neuron = self.grow_once(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        legacy_state, expected_state = self.legacy_state_for_one_grown_neuron(
            module,
            optimizer,
            grown_neuron,
        )
        grown_parameters = tuple(grown_neuron.parameters())
        grown_parameter_ids = {id(parameter) for parameter in grown_parameters}
        cluster.requires_grad_(False)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in grown_parameters)
        )
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load([optimizer], [cluster], [legacy_state])
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(
            {id(parameter) for parameter in optimizer.param_groups[1]["params"]},
            grown_parameter_ids,
        )
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(
                optimizer.state[parameter]["exp_avg"],
                expected_state[name],
            )

        cluster.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        updated_parameter = grown_neuron.nucleus.model.weight
        before_update = updated_parameter.detach().clone()
        updated_parameter.grad = torch.ones_like(updated_parameter)
        optimizer.step()

        self.assertFalse(torch.equal(updated_parameter, before_update))

    def test_legacy_load_preserves_dynamic_parameter_tied_to_external_base(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = []
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0),
        )
        grown_neuron = self.grow_once(cluster)
        grown_neuron_name = next(
            name for name, neuron in cluster.cluster.items() if neuron is grown_neuron
        )
        shared_parameter = module.other.weight
        grown_neuron.nucleus.model.weight = shared_parameter
        callback.sync_optimizers(trainer, module)
        shared_parameter_ids_by_group = [
            sum(parameter is shared_parameter for parameter in group["params"])
            for group in optimizer.param_groups
        ]
        self.assertEqual(shared_parameter_ids_by_group, [1, 0])

        for index, parameter in enumerate(module.parameters(), start=1):
            parameter.grad = torch.full_like(parameter, index / 1000.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved_optimizer_state = copy.deepcopy(optimizer.state_dict())
        source_names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        expected_group_names = [
            [
                source_names_by_parameter_id[id(parameter)]
                for parameter in group["params"]
            ]
            for group in optimizer.param_groups
        ]
        expected_state_by_name = {
            source_names_by_parameter_id[id(parameter)]: {
                state_name: state_value.detach().clone()
                for state_name, state_value in state.items()
                if isinstance(state_value, torch.Tensor)
            }
            for parameter, state in optimizer.state.items()
        }

        target_module = copy.deepcopy(module)
        target_optimizer = torch.optim.Adam(target_module.parameters(), lr=0.012)
        target_trainer = FakeTrainer([target_optimizer])
        target_trainer.state = SimpleNamespace(fn=TrainerFn.FITTING)
        target_trainer.lr_scheduler_configs = []
        target_callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = {"optimizer_states": [saved_optimizer_state]}

        target_callback.on_load_checkpoint(
            target_trainer,
            target_module,
            checkpoint,
        )
        target_optimizer.load_state_dict(saved_optimizer_state)
        target_callback.on_fit_start(target_trainer, target_module)

        target_names_by_parameter_id = {
            id(parameter): name for name, parameter in target_module.named_parameters()
        }
        self.assertEqual(
            [
                [
                    target_names_by_parameter_id[id(parameter)]
                    for parameter in group["params"]
                ]
                for group in target_optimizer.param_groups
            ],
            expected_group_names,
        )
        target_shared_parameter = target_module.other.weight
        self.assertIs(
            target_module.neuron_cluster.cluster[
                grown_neuron_name
            ].nucleus.model.weight,
            target_shared_parameter,
        )
        for name, parameter in target_module.named_parameters():
            for state_name, expected_value in expected_state_by_name[name].items():
                torch.testing.assert_close(
                    target_optimizer.state[parameter][state_name],
                    expected_value,
                )

        for index, (
            (source_name, source_parameter),
            (target_name, target_parameter),
        ) in enumerate(
            zip(
                module.named_parameters(),
                target_module.named_parameters(),
                strict=True,
            ),
            start=1,
        ):
            self.assertEqual(source_name, target_name)
            gradient = torch.full_like(source_parameter, index / 700.0)
            source_parameter.grad = gradient
            target_parameter.grad = gradient.clone()
        optimizer.step()
        target_optimizer.step()

        for (source_name, source_parameter), (target_name, target_parameter) in zip(
            module.named_parameters(),
            target_module.named_parameters(),
            strict=True,
        ):
            self.assertEqual(source_name, target_name)
            torch.testing.assert_close(
                target_parameter, source_parameter, rtol=0, atol=0
            )

    def test_legacy_load_reconstructs_nested_cluster_inside_grown_neuron(
        self,
    ) -> None:
        outer_cluster = self.build_growing_cluster()
        initial_inner_cluster = self.build_growing_cluster()
        next(iter(outer_cluster.cluster.values())).add_module(
            "nested_cluster",
            initial_inner_cluster,
        )
        module = FakeLightningModule(outer_cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = []
        callback = NeuronClusterOptimizerSyncCallback()
        callback.sync_optimizers(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0, (0,)),
        )

        grown_neuron = self.grow_once(outer_cluster)
        grown_inner_cluster = self.build_growing_cluster()
        grown_neuron.add_module("nested_cluster", grown_inner_cluster)
        callback.sync_optimizers(trainer, module)
        grown_inner_parameter_ids = {
            id(parameter) for parameter in grown_inner_cluster.parameters()
        }
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertTrue(grown_inner_parameter_ids)
        self.assertTrue(
            grown_inner_parameter_ids.issubset(
                {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
            )
        )

        for index, parameter in enumerate(module.parameters(), start=1):
            parameter.grad = torch.full_like(parameter, index / 1000.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved_optimizer_state = copy.deepcopy(optimizer.state_dict())
        source_names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        expected_group_names = [
            [
                source_names_by_parameter_id[id(parameter)]
                for parameter in group["params"]
            ]
            for group in optimizer.param_groups
        ]
        expected_state_by_name = {
            source_names_by_parameter_id[id(parameter)]: {
                state_name: state_value.detach().clone()
                for state_name, state_value in state.items()
                if isinstance(state_value, torch.Tensor)
            }
            for parameter, state in optimizer.state.items()
        }

        target_module = copy.deepcopy(module)
        target_optimizer = torch.optim.Adam(target_module.parameters(), lr=0.012)
        target_trainer = FakeTrainer([target_optimizer])
        target_trainer.state = SimpleNamespace(fn=TrainerFn.FITTING)
        target_trainer.lr_scheduler_configs = []
        target_callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = {"optimizer_states": [saved_optimizer_state]}

        target_callback.on_load_checkpoint(
            target_trainer,
            target_module,
            checkpoint,
        )
        target_optimizer.load_state_dict(saved_optimizer_state)
        target_callback.on_fit_start(target_trainer, target_module)

        target_names_by_parameter_id = {
            id(parameter): name for name, parameter in target_module.named_parameters()
        }
        self.assertEqual(
            [
                [
                    target_names_by_parameter_id[id(parameter)]
                    for parameter in group["params"]
                ]
                for group in target_optimizer.param_groups
            ],
            expected_group_names,
        )
        for name, parameter in target_module.named_parameters():
            for state_name, expected_value in expected_state_by_name[name].items():
                torch.testing.assert_close(
                    target_optimizer.state[parameter][state_name],
                    expected_value,
                )

        for index, (
            (source_name, source_parameter),
            (target_name, target_parameter),
        ) in enumerate(
            zip(
                module.named_parameters(),
                target_module.named_parameters(),
                strict=True,
            ),
            start=1,
        ):
            self.assertEqual(source_name, target_name)
            gradient = torch.full_like(source_parameter, index / 700.0)
            source_parameter.grad = gradient
            target_parameter.grad = gradient.clone()
        optimizer.step()
        target_optimizer.step()

        for (source_name, source_parameter), (target_name, target_parameter) in zip(
            module.named_parameters(),
            target_module.named_parameters(),
            strict=True,
        ):
            self.assertEqual(source_name, target_name)
            torch.testing.assert_close(
                target_parameter,
                source_parameter,
                rtol=0,
                atol=0,
            )

    def test_legacy_load_deduplicates_shared_grown_neuron_across_clusters(
        self,
    ) -> None:
        first_cluster = self.build_growing_cluster()
        second_cluster = self.build_growing_cluster()
        module = TwoClusterLightningModule(first_cluster, second_cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = []
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0, (0,)),
        )

        shared_grown_neuron = self.grow_once(first_cluster)
        replaced_grown_neuron = self.grow_once(second_cluster)
        replaced_name = next(
            name
            for name, neuron in second_cluster.cluster.items()
            if neuron is replaced_grown_neuron
        )
        second_cluster.cluster[replaced_name] = shared_grown_neuron
        callback.sync_optimizers(trainer, module)
        shared_parameter_ids = {
            id(parameter) for parameter in shared_grown_neuron.parameters()
        }
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(
            [
                sum(
                    id(parameter) in shared_parameter_ids
                    for parameter in group["params"]
                )
                for group in optimizer.param_groups
            ],
            [0, len(shared_parameter_ids)],
        )

        for index, parameter in enumerate(module.parameters(), start=1):
            parameter.grad = torch.full_like(parameter, index / 1000.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved_optimizer_state = copy.deepcopy(optimizer.state_dict())
        source_names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        expected_group_names = [
            [
                source_names_by_parameter_id[id(parameter)]
                for parameter in group["params"]
            ]
            for group in optimizer.param_groups
        ]
        expected_state_by_name = {
            source_names_by_parameter_id[id(parameter)]: {
                state_name: state_value.detach().clone()
                for state_name, state_value in state.items()
                if isinstance(state_value, torch.Tensor)
            }
            for parameter, state in optimizer.state.items()
        }

        target_module = copy.deepcopy(module)
        target_optimizer = torch.optim.Adam(target_module.parameters(), lr=0.012)
        target_trainer = FakeTrainer([target_optimizer])
        target_trainer.state = SimpleNamespace(fn=TrainerFn.FITTING)
        target_trainer.lr_scheduler_configs = []
        target_callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = {"optimizer_states": [saved_optimizer_state]}

        target_callback.on_load_checkpoint(
            target_trainer,
            target_module,
            checkpoint,
        )
        target_optimizer.load_state_dict(saved_optimizer_state)
        target_callback.on_fit_start(target_trainer, target_module)

        target_names_by_parameter_id = {
            id(parameter): name for name, parameter in target_module.named_parameters()
        }
        self.assertEqual(
            [
                [
                    target_names_by_parameter_id[id(parameter)]
                    for parameter in group["params"]
                ]
                for group in target_optimizer.param_groups
            ],
            expected_group_names,
        )
        for name, parameter in target_module.named_parameters():
            for state_name, expected_value in expected_state_by_name[name].items():
                torch.testing.assert_close(
                    target_optimizer.state[parameter][state_name],
                    expected_value,
                )

        for index, (
            (source_name, source_parameter),
            (target_name, target_parameter),
        ) in enumerate(
            zip(
                module.named_parameters(),
                target_module.named_parameters(),
                strict=True,
            ),
            start=1,
        ):
            self.assertEqual(source_name, target_name)
            gradient = torch.full_like(source_parameter, index / 700.0)
            source_parameter.grad = gradient
            target_parameter.grad = gradient.clone()
        optimizer.step()
        target_optimizer.step()

        for (source_name, source_parameter), (target_name, target_parameter) in zip(
            module.named_parameters(),
            target_module.named_parameters(),
            strict=True,
        ):
            self.assertEqual(source_name, target_name)
            torch.testing.assert_close(
                target_parameter,
                source_parameter,
                rtol=0,
                atol=0,
            )

    def test_named_layout_cancel_restores_payload_and_retry_maps_state_and_names(
        self,
    ) -> None:
        module = nn.ParameterDict(
            {
                "a": nn.Parameter(torch.zeros(2, dtype=torch.float64)),
                "b": nn.Parameter(torch.zeros(2, dtype=torch.float64)),
            }
        )
        a_parameter = module["a"]
        b_parameter = module["b"]
        source_optimizer = torch.optim.Adam(
            [
                {
                    "params": [a_parameter, b_parameter],
                    "param_names": ["custom-a", "custom-b"],
                }
            ]
        )
        for parameter, sentinel in ((a_parameter, 11.0), (b_parameter, 22.0)):
            source_optimizer.state[parameter] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.full_like(parameter, sentinel),
                "exp_avg_sq": torch.full_like(parameter, sentinel + 1.0),
            }
        saved_state = source_optimizer.state_dict()
        saved_layout = NeuronOptimizerNamedLayout.capture(
            module,
            [source_optimizer],
            [saved_state],
            {},
        )
        original_saved_ids = list(saved_state["param_groups"][0]["params"])
        original_saved_names = list(saved_state["param_groups"][0]["param_names"])
        target_optimizer = torch.optim.Adam(
            [
                {
                    "params": [b_parameter, a_parameter],
                    "param_names": ["custom-b", "custom-a"],
                }
            ]
        )
        layout_manager = NeuronOptimizerNamedLayout()

        layout_manager.prepare_for_load(
            module,
            [target_optimizer],
            [saved_state],
            saved_layout,
        )
        self.assertEqual(saved_state["param_groups"][0]["params"], [1, 0])
        self.assertEqual(
            saved_state["param_groups"][0]["param_names"],
            ["custom-b", "custom-a"],
        )
        layout_manager.clear()
        self.assertEqual(saved_state["param_groups"][0]["params"], original_saved_ids)
        self.assertEqual(
            saved_state["param_groups"][0]["param_names"],
            original_saved_names,
        )

        layout_manager.prepare_for_load(
            module,
            [target_optimizer],
            [saved_state],
            saved_layout,
        )
        target_optimizer.load_state_dict(saved_state)
        self.assertIsNone(layout_manager.complete_optimizer_load(target_optimizer))

        torch.testing.assert_close(
            target_optimizer.state[a_parameter]["exp_avg"],
            torch.full_like(a_parameter, 11.0),
        )
        torch.testing.assert_close(
            target_optimizer.state[b_parameter]["exp_avg"],
            torch.full_like(b_parameter, 22.0),
        )
        self.assertEqual(
            target_optimizer.param_groups[0]["param_names"],
            ["custom-b", "custom-a"],
        )
        self.assertEqual(saved_state["param_groups"][0]["params"], original_saved_ids)
        self.assertEqual(
            saved_state["param_groups"][0]["param_names"],
            original_saved_names,
        )

    def test_named_layout_late_param_names_failure_is_atomic_and_retryable(
        self,
    ) -> None:
        module = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(2, dtype=torch.float64))
                for name in ("a", "b", "c", "d")
            }
        )
        parameters = dict(module.named_parameters())
        source_optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameters["a"], parameters["b"]],
                    "param_names": ["custom-a", "custom-b"],
                },
                {
                    "params": [parameters["c"], parameters["d"]],
                    "param_names": ["custom-c", "custom-d"],
                },
            ]
        )
        for index, parameter in enumerate(parameters.values(), start=1):
            source_optimizer.state[parameter] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.full_like(parameter, float(index)),
                "exp_avg_sq": torch.full_like(parameter, float(index + 10)),
            }
        saved_state = source_optimizer.state_dict()
        saved_layout = NeuronOptimizerNamedLayout.capture(
            module,
            [source_optimizer],
            [saved_state],
            {},
        )
        target_optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameters["b"], parameters["a"]],
                    "param_names": ["custom-b", "custom-a"],
                },
                {
                    "params": [parameters["d"], parameters["c"]],
                    "param_names": ["custom-d", "custom-c"],
                },
            ]
        )
        layout_manager = NeuronOptimizerNamedLayout()
        first_saved_group = saved_state["param_groups"][0]
        first_saved_ids = first_saved_group["params"]
        first_saved_names = first_saved_group["param_names"]
        original_first_ids = tuple(first_saved_ids)
        original_first_names = tuple(first_saved_names)
        second_saved_names = saved_state["param_groups"][1]["param_names"]
        original_second_names = tuple(second_saved_names)
        second_saved_names.pop()

        with self.assertRaisesRegex(
            RuntimeError,
            "param_names metadata",
        ):
            layout_manager.prepare_for_load(
                module,
                [target_optimizer],
                [saved_state],
                saved_layout,
            )

        self.assertIs(first_saved_group["params"], first_saved_ids)
        self.assertEqual(tuple(first_saved_ids), original_first_ids)
        self.assertIs(first_saved_group["param_names"], first_saved_names)
        self.assertEqual(tuple(first_saved_names), original_first_names)

        second_saved_names[:] = original_second_names
        layout_manager.prepare_for_load(
            module,
            [target_optimizer],
            [saved_state],
            saved_layout,
        )
        self.assertEqual(first_saved_group["params"], [1, 0])
        self.assertEqual(
            first_saved_group["param_names"],
            ["custom-b", "custom-a"],
        )
        target_optimizer.load_state_dict(saved_state)
        self.assertIsNone(layout_manager.complete_optimizer_load(target_optimizer))

        for index, (name, parameter) in enumerate(parameters.items(), start=1):
            torch.testing.assert_close(
                target_optimizer.state[parameter]["exp_avg"],
                torch.full_like(parameter, float(index)),
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_legacy_late_suffix_param_names_failure_is_atomic_and_retryable(
        self,
    ) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=3,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        first_dynamic_neuron = cluster._initialize_neuron(1, 1, 1)
        second_dynamic_neuron = cluster._initialize_neuron(1, 1, 3)
        cluster.cluster["neuron_1_1_1"] = first_dynamic_neuron
        cluster.cluster["neuron_1_1_3"] = second_dynamic_neuron
        module = FakeLightningModule(cluster)
        names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(module.parameters()),
                    "param_names": [name for name, _ in module.named_parameters()],
                }
            ],
            lr=0.012,
        )
        state_parameter = module.other.weight
        state_value = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.ones_like(state_parameter),
            "exp_avg_sq": torch.full_like(state_parameter, 2.0),
        }
        optimizer.state[state_parameter] = state_value
        serialized_state = optimizer.state_dict()
        serialized_ids_by_parameter_id = {
            id(parameter): serialized_id
            for parameter, serialized_id in zip(
                optimizer.param_groups[0]["params"],
                serialized_state["param_groups"][0]["params"],
                strict=True,
            )
        }
        dynamic_parameter_groups = [
            [
                parameter
                for parameter in first_dynamic_neuron.parameters()
                if parameter.requires_grad
            ],
            [
                parameter
                for parameter in second_dynamic_neuron.parameters()
                if parameter.requires_grad
            ],
        ]
        dynamic_parameter_ids = {
            id(parameter) for group in dynamic_parameter_groups for parameter in group
        }
        base_parameters = [
            parameter
            for parameter in optimizer.param_groups[0]["params"]
            if id(parameter) not in dynamic_parameter_ids
        ]
        saved_group_options = {
            name: value
            for name, value in serialized_state["param_groups"][0].items()
            if name not in {"params", "param_names"}
        }

        def saved_group(parameters):
            return {
                **saved_group_options,
                "params": [
                    serialized_ids_by_parameter_id[id(parameter)]
                    for parameter in parameters
                ],
                "param_names": [
                    names_by_parameter_id[id(parameter)] for parameter in parameters
                ],
            }

        saved_state = {
            "state": serialized_state["state"],
            "param_groups": [
                saved_group(base_parameters),
                *(saved_group(group) for group in dynamic_parameter_groups),
            ],
        }
        malformed_names = saved_state["param_groups"][2]["param_names"]
        original_names = tuple(malformed_names)
        malformed_names.pop()
        reconciler = NeuronOptimizerCheckpointReconciler()
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)

        with self.assertRaisesRegex(RuntimeError, "param_names"):
            reconciler.prepare_for_load([optimizer], [cluster], [saved_state])

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )

        malformed_names[:] = original_names
        reconciler.prepare_for_load([optimizer], [cluster], [saved_state])
        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertEqual(
            optimizer.param_groups[0]["params"],
            base_parameters,
        )
        self.assertEqual(
            optimizer.param_groups[1]["params"],
            dynamic_parameter_groups[0],
        )
        self.assertEqual(
            optimizer.param_groups[2]["params"],
            dynamic_parameter_groups[1],
        )
        optimizer.load_state_dict(saved_state)
        self.assertEqual(
            reconciler.complete_optimizer_load(optimizer),
            LegacyOptimizerAppendPolicy(1, 0, (0, 0, 0)),
        )
        torch.testing.assert_close(
            optimizer.state[state_parameter]["exp_avg"],
            state_value["exp_avg"],
        )

    def test_legacy_custom_role_options_match_uninterrupted_next_step(self) -> None:
        cluster = self.build_growing_cluster()
        grown_neuron = self.grow_once(cluster)
        template = FakeLightningModule(cluster)
        uninterrupted_module = copy.deepcopy(template)
        resumed_module = copy.deepcopy(template)
        grown_parameter_ids = {id(parameter) for parameter in grown_neuron.parameters()}
        grown_names = {
            name
            for name, parameter in template.named_parameters()
            if id(parameter) in grown_parameter_ids
        }

        def parameters_by_role(module):
            named_parameters = dict(module.named_parameters())
            base_nucleus = [
                parameter
                for name, parameter in named_parameters.items()
                if name not in grown_names and ".nucleus." in name
            ]
            base_other = [
                parameter
                for name, parameter in named_parameters.items()
                if name not in grown_names and ".nucleus." not in name
            ]
            dynamic = [named_parameters[name] for name in grown_names]
            return base_nucleus, base_other, dynamic

        source_base_nucleus, source_base_other, source_dynamic = parameters_by_role(
            uninterrupted_module
        )
        uninterrupted_optimizer = torch.optim.Adam(
            [
                {"params": source_base_nucleus, "lr": 0.01},
                {"params": source_base_other, "lr": 0.02},
                {"params": source_dynamic, "lr": 0.01},
            ]
        )
        legacy_state = uninterrupted_optimizer.state_dict()

        target_base_nucleus, target_base_other, target_dynamic = parameters_by_role(
            resumed_module
        )
        resumed_optimizer = torch.optim.Adam(
            [
                {
                    "params": target_base_nucleus
                    + [
                        parameter
                        for name, parameter in zip(
                            grown_names,
                            target_dynamic,
                            strict=True,
                        )
                        if ".nucleus." in name
                    ],
                    "lr": 0.01,
                },
                {
                    "params": target_base_other
                    + [
                        parameter
                        for name, parameter in zip(
                            grown_names,
                            target_dynamic,
                            strict=True,
                        )
                        if ".nucleus." not in name
                    ],
                    "lr": 0.02,
                },
            ]
        )
        reconciler = NeuronOptimizerCheckpointReconciler()
        reconciler.prepare_for_load(
            [resumed_optimizer],
            [resumed_module.neuron_cluster],
            [legacy_state],
        )
        resumed_optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(resumed_optimizer))
        self.assertEqual(
            [group["lr"] for group in resumed_optimizer.param_groups],
            [0.01, 0.02, 0.01],
        )

        for parameter in uninterrupted_module.parameters():
            parameter.grad = torch.ones_like(parameter)
        for parameter in resumed_module.parameters():
            parameter.grad = torch.ones_like(parameter)
        uninterrupted_optimizer.step()
        resumed_optimizer.step()

        for (source_name, source_parameter), (target_name, target_parameter) in zip(
            uninterrupted_module.named_parameters(),
            resumed_module.named_parameters(),
            strict=True,
        ):
            self.assertEqual(source_name, target_name)
            torch.testing.assert_close(target_parameter, source_parameter)

    def test_legacy_lambda_scheduler_copies_reference_group_configuration(
        self,
    ) -> None:
        parameters = [
            nn.Parameter(torch.tensor(float(index), dtype=torch.float64))
            for index in range(3)
        ]
        optimizer = torch.optim.SGD(
            [
                {"params": [parameters[0]], "lr": 0.1},
                {"params": [parameters[1]], "lr": 0.2},
            ]
        )

        def first_schedule(step):
            return 0.5**step

        def second_schedule(step):
            return 0.8**step

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [first_schedule, second_schedule],
        )
        saved_scheduler_state = scheduler.state_dict()
        optimizer.add_param_group(
            {
                **{
                    name: value
                    for name, value in optimizer.param_groups[0].items()
                    if name != "params"
                },
                "params": [parameters[2]],
            }
        )

        reconcile_scheduler_group_count(
            scheduler,
            saved_scheduler_state,
            LegacyOptimizerAppendPolicy(2, 0),
            target_group_count=3,
        )
        scheduler.load_state_dict(saved_scheduler_state)
        optimizer.step()
        scheduler.step()

        self.assertEqual(len(scheduler.lr_lambdas), 3)
        self.assertIs(scheduler.lr_lambdas[2], scheduler.lr_lambdas[0])
        self.assertEqual(len(scheduler.base_lrs), 3)
        self.assertEqual(len(scheduler.get_last_lr()), 3)
        self.assertEqual(
            optimizer.param_groups[2]["lr"],
            optimizer.param_groups[0]["lr"],
        )

    def test_named_scheduler_lineage_rebuilds_live_lambdas_only(self) -> None:
        parameters = [
            nn.Parameter(torch.tensor(float(index), dtype=torch.float64))
            for index in range(3)
        ]
        optimizer = torch.optim.SGD(
            [
                {"params": [parameters[0]], "lr": 0.1},
                {"params": [parameters[1]], "lr": 0.2},
                {"params": [parameters[2]], "lr": 0.3},
            ]
        )

        def first_schedule(step):
            return 0.5**step

        def second_schedule(step):
            return 0.8**step

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [first_schedule, second_schedule, lambda step: 0.9**step],
        )
        saved_scheduler_state = scheduler.state_dict()
        original_saved_state = copy.deepcopy(saved_scheduler_state)
        original_saved_lists = {
            name: value
            for name, value in saved_scheduler_state.items()
            if isinstance(value, list)
        }

        reconcile_scheduler_group_count(
            scheduler,
            saved_scheduler_state,
            LegacyOptimizerAppendPolicy(
                2,
                0,
                group_reference_indices=(0, 1, 1),
            ),
            target_group_count=3,
        )

        self.assertIs(scheduler.lr_lambdas[0], first_schedule)
        self.assertIs(scheduler.lr_lambdas[1], second_schedule)
        self.assertIs(scheduler.lr_lambdas[2], second_schedule)
        self.assertEqual(saved_scheduler_state, original_saved_state)
        for name, original_list in original_saved_lists.items():
            self.assertIs(saved_scheduler_state[name], original_list)

        scheduler.load_state_dict(saved_scheduler_state)

        self.assertIs(scheduler.lr_lambdas[2], second_schedule)

    def test_legacy_append_policy_optimizes_growth_after_checkpoint_load(
        self,
    ) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=6,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=4,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            max_total_growths=2,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        module = FakeLightningModule(cluster)
        nucleus_parameters = []
        other_parameters = []
        for name, parameter in module.named_parameters():
            target = nucleus_parameters if ".nucleus." in name else other_parameters
            target.append(parameter)
        optimizer = torch.optim.Adam(
            [
                {
                    "params": nucleus_parameters,
                    "lr": 0.011,
                    "weight_decay": 0.031,
                    "betas": (0.71, 0.93),
                },
                {
                    "params": other_parameters,
                    "lr": 0.021,
                    "weight_decay": 0.041,
                    "betas": (0.81, 0.94),
                },
            ]
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)

        initial_names = set(cluster.cluster)
        cluster(torch.randn(self.batch_size, self.input_dim))
        first_grown_name = next(iter(set(cluster.cluster) - initial_names))
        first_grown_parameters = list(cluster.cluster[first_grown_name].parameters())
        reference_group = optimizer.param_groups[0]
        optimizer.add_param_group(
            {
                **{
                    name: value
                    for name, value in reference_group.items()
                    if name != "params"
                },
                "params": first_grown_parameters,
            }
        )
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(2, 0),
        )
        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        names_before_second_growth = set(cluster.cluster)
        cluster(torch.randn(self.batch_size, self.input_dim))
        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=1,
        )
        second_grown_name = next(
            iter(set(cluster.cluster) - names_before_second_growth)
        )
        second_grown_parameters = list(cluster.cluster[second_grown_name].parameters())

        self.assertEqual(len(optimizer.param_groups), 4)
        self.assertEqual(
            optimizer.param_groups[-1]["params"],
            second_grown_parameters,
        )
        self.assertEqual(optimizer.param_groups[-1]["lr"], 0.011)
        self.assertEqual(optimizer.param_groups[-1]["weight_decay"], 0.031)
        self.assertEqual(optimizer.param_groups[-1]["betas"], (0.71, 0.93))
        optimized_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        for parameter in second_grown_parameters:
            self.assertEqual(optimized_ids.count(id(parameter)), 1)

    def test_legacy_policy_owns_frozen_growth_before_later_unfreeze(self) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=6,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=4,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            max_total_growths=2,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        module = FakeLightningModule(cluster)
        nucleus_parameters = []
        other_parameters = []
        for name, parameter in module.named_parameters():
            target = nucleus_parameters if ".nucleus." in name else other_parameters
            target.append(parameter)
        optimizer = torch.optim.Adam(
            [
                {"params": nucleus_parameters, "lr": 0.011},
                {"params": other_parameters, "lr": 0.021},
            ]
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)

        initial_names = set(cluster.cluster)
        cluster(torch.randn(self.batch_size, self.input_dim))
        first_grown_name = next(iter(set(cluster.cluster) - initial_names))
        first_grown_parameters = list(cluster.cluster[first_grown_name].parameters())
        optimizer.add_param_group(
            {
                "params": first_grown_parameters,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(2, 0),
        )
        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        cluster.requires_grad_(False)
        names_before_second_growth = set(cluster.cluster)
        cluster(torch.randn(self.batch_size, self.input_dim))
        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=1,
        )
        second_grown_name = next(
            iter(set(cluster.cluster) - names_before_second_growth)
        )
        second_grown_neuron = cluster.cluster[second_grown_name]
        second_grown_parameters = list(second_grown_neuron.parameters())

        self.assertTrue(second_grown_parameters)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in second_grown_parameters)
        )
        optimized_parameter_ids = self.optimizer_param_ids(optimizer)
        self.assertTrue(
            all(
                id(parameter) in optimized_parameter_ids
                for parameter in second_grown_parameters
            )
        )
        self.assertEqual(optimizer.param_groups[-1]["params"], second_grown_parameters)

        cluster.requires_grad_(True)
        callback.on_train_batch_start(
            trainer,
            module,
            batch=None,
            batch_idx=2,
        )
        updated_parameter = second_grown_neuron.nucleus.model.weight
        before_step = updated_parameter.detach().clone()
        updated_parameter.grad = torch.ones_like(updated_parameter)
        optimizer.step()

        self.assertFalse(torch.equal(updated_parameter, before_step))

    def test_named_load_removes_optimizer_orphans_from_reconciled_topology(
        self,
    ) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=3,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=2,
            max_steps=1,
            neuron_config=self.full_sampler_neuron_config(),
        )
        source_cluster = config.build()
        source_module = FakeLightningModule(source_cluster)
        source_optimizer = torch.optim.Adam(source_module.parameters(), lr=0.012)
        source_trainer = FakeTrainer([source_optimizer])
        source_callback = NeuronClusterOptimizerSyncCallback()
        source_callback.on_fit_start(source_trainer, source_module)

        del source_cluster.cluster["neuron_1_1_2"]
        source_callback.sync_optimizers(source_trainer, source_module)
        for parameter_index, parameter in enumerate(
            source_module.parameters(),
            start=1,
        ):
            parameter.grad = torch.full_like(parameter, parameter_index / 1000.0)
        source_optimizer.step()
        source_optimizer.zero_grad(set_to_none=True)
        expected_state_by_name = {
            name: {
                state_name: value.detach().clone()
                for state_name, value in source_optimizer.state[parameter].items()
                if isinstance(value, torch.Tensor)
            }
            for name, parameter in source_module.named_parameters()
        }
        checkpoint = {
            "optimizer_states": [copy.deepcopy(source_optimizer.state_dict())],
        }
        source_callback.on_save_checkpoint(
            source_trainer,
            source_module,
            checkpoint,
        )
        source_state_dict = copy.deepcopy(source_module.state_dict())

        target_cluster = config.build()
        target_module = FakeLightningModule(target_cluster)
        removed_neuron = target_cluster.cluster["neuron_1_1_2"]
        removed_parameter_ids = {
            id(parameter) for parameter in removed_neuron.parameters()
        }
        target_optimizer = torch.optim.Adam(target_module.parameters(), lr=0.5)
        target_trainer = FakeTrainer([target_optimizer])
        target_callback = NeuronClusterOptimizerSyncCallback()

        target_module.load_state_dict(source_state_dict, strict=True)
        self.assertNotIn("neuron_1_1_2", target_cluster.cluster)
        self.assertTrue(
            removed_parameter_ids.issubset(self.optimizer_param_ids(target_optimizer))
        )

        target_callback.on_load_checkpoint(
            target_trainer,
            target_module,
            checkpoint,
        )
        target_optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        target_callback.on_fit_start(target_trainer, target_module)

        live_parameter_ids = {id(parameter) for parameter in target_module.parameters()}
        self.assertTrue(
            self.optimizer_param_ids(target_optimizer).issubset(live_parameter_ids)
        )
        self.assertTrue(
            removed_parameter_ids.isdisjoint(self.optimizer_param_ids(target_optimizer))
        )
        for name, parameter in target_module.named_parameters():
            with self.subTest(parameter_name=name):
                for state_name, expected_value in expected_state_by_name[name].items():
                    torch.testing.assert_close(
                        target_optimizer.state[parameter][state_name],
                        expected_value,
                    )

    def test_legacy_policy_skips_a_second_cluster_the_optimizer_never_owned(
        self,
    ) -> None:
        first_cluster = self.build_growing_cluster()
        second_cluster = self.build_growing_cluster()
        module = TwoClusterLightningModule(first_cluster, second_cluster)
        optimizer = torch.optim.Adam(first_cluster.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        with self.assertWarns(UserWarning):
            callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0),
        )
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)

        grown_neuron = self.grow_once(second_cluster)
        grown_parameter_ids = {id(parameter) for parameter in grown_neuron.parameters()}
        with self.assertWarns(UserWarning):
            callback.sync_optimizers(trainer, module)

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )
        self.assertTrue(
            grown_parameter_ids.isdisjoint(self.optimizer_param_ids(optimizer))
        )

    def test_legacy_growth_extends_lineage_for_two_clusters_in_one_sync(
        self,
    ) -> None:
        first_cluster = self.build_growing_cluster()
        second_cluster = self.build_growing_cluster()
        module = TwoClusterLightningModule(first_cluster, second_cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 0.75**step,
        )
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0, (0,)),
        )
        first_grown_parameters = list(self.grow_once(first_cluster).parameters())
        second_grown_parameters = list(self.grow_once(second_cluster).parameters())

        callback.sync_optimizers(trainer, module)

        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertEqual(
            [id(parameter) for parameter in optimizer.param_groups[1]["params"]],
            [id(parameter) for parameter in first_grown_parameters],
        )
        self.assertEqual(
            [id(parameter) for parameter in optimizer.param_groups[2]["params"]],
            [id(parameter) for parameter in second_grown_parameters],
        )
        self.assertEqual(len(scheduler.base_lrs), 3)
        self.assertEqual(len(scheduler.lr_lambdas), 3)
        self.assertEqual(
            callback._legacy_append_policies[id(optimizer)],
            LegacyOptimizerAppendPolicy(1, 0, (0, 0, 0)),
        )

    def test_legacy_growth_with_unknown_scheduler_is_atomic_and_retryable(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        supported_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 0.5**step,
        )
        unsupported_scheduler = UnsupportedLambdaLR(
            optimizer,
            lambda step: 0.8**step,
        )
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = [
            SimpleNamespace(scheduler=supported_scheduler),
            SimpleNamespace(scheduler=unsupported_scheduler),
        ]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0),
        )
        self.grow_once(cluster)
        original_group = optimizer.param_groups[0]
        original_parameters = list(original_group["params"])

        with self.assertRaisesRegex(RuntimeError, "unrecognized scheduler"):
            callback.sync_optimizers(trainer, module)

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertIs(optimizer.param_groups[0], original_group)
        self.assertEqual(original_group["params"], original_parameters)
        self.assertEqual(len(supported_scheduler.base_lrs), 1)
        self.assertEqual(len(supported_scheduler.lr_lambdas), 1)
        self.assertEqual(len(unsupported_scheduler.base_lrs), 1)

        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=supported_scheduler)]
        callback.sync_optimizers(trainer, module)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(len(supported_scheduler.base_lrs), 2)
        self.assertEqual(len(supported_scheduler.lr_lambdas), 2)

    def test_legacy_growth_rejects_lineage_length_drift_atomically(self) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(1, 0, (0, 0)),
        )
        grown_neuron = self.grow_once(cluster)
        grown_parameter_ids = {id(parameter) for parameter in grown_neuron.parameters()}
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)

        with self.assertRaisesRegex(RuntimeError, "lineage does not match"):
            callback.sync_optimizers(trainer, module)

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )
        self.assertTrue(
            grown_parameter_ids.isdisjoint(self.optimizer_param_ids(optimizer))
        )

    def test_legacy_growth_rejects_a_vanished_base_group_atomically(self) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(2, 0),
        )
        grown_neuron = self.grow_once(cluster)
        grown_parameter_ids = {id(parameter) for parameter in grown_neuron.parameters()}
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)

        with self.assertRaisesRegex(RuntimeError, "base parameter groups"):
            callback.sync_optimizers(trainer, module)

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )
        self.assertTrue(
            grown_parameter_ids.isdisjoint(self.optimizer_param_ids(optimizer))
        )

    def test_legacy_growth_uses_first_group_with_current_cluster_parameters(
        self,
    ) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=3,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=2,
            max_steps=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        dynamic_neuron = cluster._initialize_neuron(1, 1, 3)
        cluster.cluster["neuron_1_1_3"] = dynamic_neuron
        module = FakeLightningModule(cluster)
        reference_neuron = cluster.cluster["neuron_1_1_2"]
        reference_ids = {id(parameter) for parameter in reference_neuron.parameters()}
        dynamic_ids = {id(parameter) for parameter in dynamic_neuron.parameters()}
        names_by_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        reference_parameters = [module.other.weight, *reference_neuron.parameters()]
        other_parameters = [
            parameter
            for parameter in module.parameters()
            if id(parameter) not in reference_ids | dynamic_ids
            and parameter is not module.other.weight
        ]
        dynamic_parameters = list(dynamic_neuron.parameters())

        def named_group(parameters, lr):
            return {
                "params": parameters,
                "param_names": [names_by_id[id(parameter)] for parameter in parameters],
                "lr": lr,
            }

        optimizer = torch.optim.Adam(
            [
                named_group(reference_parameters, 0.01),
                named_group(other_parameters, 0.02),
                named_group(dynamic_parameters, 0.01),
            ]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [
                lambda step: 0.5**step,
                lambda step: 0.8**step,
                lambda step: 0.5**step,
            ],
        )
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(2, 0),
        )

        del cluster.cluster["neuron_1_1_2"]
        callback.sync_optimizers(trainer, module)
        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertEqual(optimizer.param_groups[0]["params"], [module.other.weight])

        replacement_neuron = cluster._initialize_neuron(1, 1, 2)
        cluster.cluster["neuron_1_1_2"] = replacement_neuron
        callback.sync_optimizers(trainer, module)

        replacement_parameters = list(replacement_neuron.parameters())
        appended_group = optimizer.param_groups[-1]
        current_names_by_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        self.assertEqual(
            [id(parameter) for parameter in appended_group["params"]],
            [id(parameter) for parameter in replacement_parameters],
        )
        self.assertEqual(appended_group["lr"], 0.02)
        self.assertEqual(
            appended_group["param_names"],
            [
                current_names_by_id[id(parameter)]
                for parameter in replacement_parameters
            ],
        )
        self.assertIs(scheduler.lr_lambdas[-1], scheduler.lr_lambdas[1])

    def test_pruning_legacy_reference_preserves_group_scheduler_lineage(self) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=3,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=2,
            max_steps=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        dynamic_neuron = cluster._initialize_neuron(1, 1, 3)
        cluster.cluster["neuron_1_1_3"] = dynamic_neuron
        module = FakeLightningModule(cluster)
        reference_neuron = cluster.cluster["neuron_1_1_2"]
        reference_ids = {id(parameter) for parameter in reference_neuron.parameters()}
        dynamic_ids = {id(parameter) for parameter in dynamic_neuron.parameters()}
        other_parameters = [
            parameter
            for parameter in module.parameters()
            if id(parameter) not in reference_ids | dynamic_ids
        ]
        optimizer = torch.optim.Adam(
            [
                {"params": list(reference_neuron.parameters()), "lr": 0.01},
                {"params": other_parameters, "lr": 0.02},
                {"params": list(dynamic_neuron.parameters()), "lr": 0.01},
            ]
        )

        def reference_schedule(step):
            return 0.5**step

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [
                reference_schedule,
                lambda step: 0.8**step,
                reference_schedule,
            ],
        )
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(2, 0),
        )
        self.assertIs(scheduler.lr_lambdas[0], reference_schedule)

        del cluster.cluster["neuron_1_1_2"]
        callback.sync_optimizers(trainer, module)

        self.assertEqual(
            callback._legacy_append_policies[id(optimizer)],
            LegacyOptimizerAppendPolicy(2, 0, (1, 0)),
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups], [0.02, 0.01]
        )
        self.assertIs(scheduler.lr_lambdas[1], reference_schedule)

        replacement_neuron = cluster._initialize_neuron(1, 1, 2)
        cluster.cluster["neuron_1_1_2"] = replacement_neuron
        callback.sync_optimizers(trainer, module)
        self.assertEqual(optimizer.param_groups[-1]["lr"], 0.02)
        self.assertIs(scheduler.lr_lambdas[-1], scheduler.lr_lambdas[0])
        self.assertEqual(
            callback._legacy_append_policies[id(optimizer)],
            LegacyOptimizerAppendPolicy(2, 0, (1, 0, 1)),
        )

    def test_cancelled_legacy_load_rolls_back_temporary_groups(self) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        grown_neuron = self.grow_once(cluster)
        optimizer = torch.optim.Adam(
            reversed(list(module.parameters())),
            lr=0.012,
        )
        original_order = list(optimizer.param_groups[0]["params"])
        legacy_state, _ = self.legacy_state_for_one_grown_neuron(
            module,
            optimizer,
            grown_neuron,
        )
        trainer = FakeTrainer([])
        trainer.state = SimpleNamespace(fn=TrainerFn.FITTING)
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_load_checkpoint(
            trainer,
            module,
            {"optimizer_states": [legacy_state]},
        )
        trainer.optimizers = [optimizer]

        callback.on_fit_start(trainer, module)

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertIsNone(callback._pending_saved_optimizer_states)
        callback.on_fit_end(trainer, module)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["params"], original_order)

    def test_cancelled_scheduler_reconciliation_restores_payload_and_retries(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        grown_neuron = self.grow_once(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        legacy_state, _ = self.legacy_state_for_one_grown_neuron(
            module,
            optimizer,
            grown_neuron,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 0.5**step,
        )
        saved_scheduler_state = scheduler.state_dict()
        checkpoint = {
            "optimizer_states": [legacy_state],
            "lr_schedulers": [saved_scheduler_state],
        }
        trainer = FakeTrainer([])
        trainer.state = SimpleNamespace(fn=TrainerFn.FITTING)
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()

        callback.on_load_checkpoint(trainer, module, checkpoint)
        trainer.optimizers = [optimizer]
        callback.on_fit_start(trainer, module)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(len(scheduler.base_lrs), 2)
        self.assertEqual(len(scheduler.lr_lambdas), 2)
        self.assertEqual(len(saved_scheduler_state["base_lrs"]), 2)

        callback.on_fit_end(trainer, module)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(len(scheduler.base_lrs), 1)
        self.assertEqual(len(scheduler.lr_lambdas), 1)
        self.assertEqual(len(saved_scheduler_state["base_lrs"]), 1)

        callback.on_load_checkpoint(trainer, module, checkpoint)
        optimizer.load_state_dict(legacy_state)
        scheduler.load_state_dict(saved_scheduler_state)
        callback.on_fit_start(trainer, module)
        optimizer.step()
        scheduler.step()
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(len(scheduler.base_lrs), 2)
        self.assertEqual(len(scheduler.lr_lambdas), 2)
        self.assertEqual(len(scheduler.get_last_lr()), 2)

    def test_callback_rolls_back_partial_scheduler_load_and_retries_same_objects(
        self,
    ) -> None:
        class StatefulSchedule:
            def __init__(self) -> None:
                self.factor = 0.5

            def __call__(self, step: int) -> float:
                return self.factor**step

        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        grown_neuron = self.grow_once(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        live_state_value = {"sentinel": torch.tensor(7.0)}
        optimizer.state[module.other.weight] = live_state_value
        legacy_state, _ = self.legacy_state_for_one_grown_neuron(
            module,
            optimizer,
            grown_neuron,
        )
        schedule = StatefulSchedule()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        valid_scheduler_state = copy.deepcopy(scheduler.state_dict())
        saved_scheduler_state = scheduler.state_dict()
        saved_scheduler_state.update(
            {
                "last_epoch": 99,
                "_step_count": 100,
                "unexpected_partial_load_value": "must be removed",
                "lr_lambdas": [{"factor": 0.25}, 7],
            }
        )
        original_saved_scheduler_state = copy.deepcopy(saved_scheduler_state)
        original_saved_scheduler_lists = {
            name: value
            for name, value in saved_scheduler_state.items()
            if isinstance(value, list)
        }
        checkpoint = {
            "optimizer_states": [legacy_state],
            "lr_schedulers": [saved_scheduler_state],
        }
        trainer = FakeTrainer([optimizer])
        trainer.state = SimpleNamespace(fn=TrainerFn.FITTING)
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        original_policy = LegacyOptimizerAppendPolicy(1, 0, (0,))
        callback._record_legacy_append_policy(optimizer, original_policy)
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)
        scheduler_snapshot = self.scheduler_identity_snapshot(scheduler)

        callback.on_load_checkpoint(trainer, module, checkpoint)
        optimizer.load_state_dict(legacy_state)
        self.assertEqual(len(optimizer.param_groups), 2)
        with self.assertRaises(TypeError) as raised:
            scheduler.load_state_dict(saved_scheduler_state)
        self.assertEqual(scheduler.last_epoch, 99)
        self.assertEqual(scheduler._step_count, 100)
        self.assertEqual(schedule.factor, 0.25)

        callback.on_exception(trainer, module, raised.exception)

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )
        self.assert_scheduler_matches_identity_snapshot(
            scheduler,
            scheduler_snapshot,
        )
        self.assertEqual(schedule.factor, 0.5)
        self.assertEqual(
            callback._legacy_append_policies,
            {id(optimizer): original_policy},
        )
        self.assertEqual(
            saved_scheduler_state,
            original_saved_scheduler_state,
        )
        for name, original_list in original_saved_scheduler_lists.items():
            self.assertIs(saved_scheduler_state[name], original_list)

        saved_scheduler_state["last_epoch"] = valid_scheduler_state["last_epoch"]
        saved_scheduler_state["_step_count"] = valid_scheduler_state["_step_count"]
        saved_scheduler_state.pop("unexpected_partial_load_value")
        saved_scheduler_state["lr_lambdas"][:] = valid_scheduler_state["lr_lambdas"]
        callback.on_load_checkpoint(trainer, module, checkpoint)
        optimizer.load_state_dict(legacy_state)
        scheduler.load_state_dict(saved_scheduler_state)
        callback.on_train_start(trainer, module)

        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(len(scheduler.base_lrs), 2)
        self.assertEqual(len(scheduler.lr_lambdas), 2)
        self.assertEqual(len(scheduler.get_last_lr()), 2)
        self.assertEqual(schedule.factor, 0.5)

    def test_partial_scheduler_load_rolls_back_full_state_and_retries(
        self,
    ) -> None:
        class StatefulSchedule:
            def __init__(self) -> None:
                self.factor = 0.5

            def __call__(self, step: int) -> float:
                return self.factor**step

        parameters = [
            nn.Parameter(torch.tensor(float(index), dtype=torch.float64))
            for index in range(2)
        ]
        optimizer = torch.optim.SGD([parameters[0]], lr=0.1)
        schedule = StatefulSchedule()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        saved_scheduler_state = scheduler.state_dict()
        optimizer.add_param_group(
            {
                **{
                    name: value
                    for name, value in optimizer.param_groups[0].items()
                    if name != "params"
                },
                "params": [parameters[1]],
            }
        )
        reconciler = NeuronSchedulerCheckpointReconciler()
        binding = SchedulerGroupLoadBinding(
            scheduler=scheduler,
            saved_state=saved_scheduler_state,
            optimizer=optimizer,
            policy=LegacyOptimizerAppendPolicy(1, 0),
            target_group_count=2,
        )
        original_scheduler_keys = set(scheduler.__dict__)
        original_base_lrs = scheduler.base_lrs
        original_last_lrs = scheduler._last_lr
        original_lambdas = scheduler.lr_lambdas
        original_saved_base_lrs = saved_scheduler_state["base_lrs"]

        reconciler.prepare_for_load([binding])
        malformed_state = copy.deepcopy(saved_scheduler_state)
        malformed_state.update(
            {
                "last_epoch": 99,
                "_step_count": 100,
                "unexpected_partial_load_value": "must be removed",
                "lr_lambdas": [{"factor": 0.25}, 7],
            }
        )
        with self.assertRaises(TypeError):
            scheduler.load_state_dict(malformed_state)
        self.assertEqual(scheduler.last_epoch, 99)
        self.assertEqual(scheduler._step_count, 100)
        self.assertEqual(schedule.factor, 0.25)

        reconciler.clear()

        self.assertEqual(set(scheduler.__dict__), original_scheduler_keys)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(scheduler._step_count, 1)
        self.assertEqual(schedule.factor, 0.5)
        self.assertIs(scheduler.base_lrs, original_base_lrs)
        self.assertEqual(scheduler.base_lrs, [0.1])
        self.assertIs(scheduler._last_lr, original_last_lrs)
        self.assertEqual(scheduler._last_lr, [0.1])
        self.assertIs(scheduler.lr_lambdas, original_lambdas)
        self.assertEqual(scheduler.lr_lambdas, [schedule])
        self.assertIs(
            saved_scheduler_state["base_lrs"],
            original_saved_base_lrs,
        )
        self.assertEqual(saved_scheduler_state["base_lrs"], [0.1])

        reconciler.prepare_for_load([binding])
        scheduler.load_state_dict(saved_scheduler_state)
        reconciler.mark_optimizer_loaded(optimizer)
        reconciler.commit_loaded()
        optimizer.step()
        scheduler.step()

        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(scheduler._step_count, 2)
        self.assertEqual(schedule.factor, 0.5)
        self.assertEqual(len(scheduler.base_lrs), 2)
        self.assertEqual(len(scheduler.lr_lambdas), 2)
        self.assertEqual(len(scheduler.get_last_lr()), 2)

    def test_named_role_load_rolls_back_same_cardinality_scheduler_failure(
        self,
    ) -> None:
        class StatefulSchedule:
            def __init__(self) -> None:
                self.factor = 0.5

            def __call__(self, step: int) -> float:
                return self.factor**step

        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        schedule = StatefulSchedule()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        optimizer_state = optimizer.state_dict()
        optimizer_layout = NeuronOptimizerNamedLayout.capture(
            module,
            [optimizer],
            [optimizer_state],
            {},
        )
        valid_scheduler_state = scheduler.state_dict()
        malformed_scheduler_state = copy.deepcopy(valid_scheduler_state)
        malformed_scheduler_state.update(
            {
                "last_epoch": 99,
                "_step_count": 100,
                "unexpected_partial_load_value": "must be removed",
                "lr_lambdas": [7],
            }
        )
        checkpoint = {
            "optimizer_states": [optimizer_state],
            "lr_schedulers": [malformed_scheduler_state],
            OPTIMIZER_LAYOUT_CHECKPOINT_KEY: optimizer_layout,
        }
        trainer = FakeTrainer([optimizer])
        trainer.state = SimpleNamespace(fn=TrainerFn.FITTING)
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)
        scheduler_snapshot = self.scheduler_identity_snapshot(scheduler)

        callback.on_load_checkpoint(trainer, module, checkpoint)
        optimizer.load_state_dict(optimizer_state)
        with self.assertRaises(TypeError) as raised:
            scheduler.load_state_dict(malformed_scheduler_state)
        callback.on_exception(trainer, module, raised.exception)

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )
        self.assert_scheduler_matches_identity_snapshot(
            scheduler,
            scheduler_snapshot,
        )
        self.assertEqual(schedule.factor, 0.5)
        self.assertNotIn("unexpected_partial_load_value", scheduler.__dict__)

        checkpoint["lr_schedulers"] = [valid_scheduler_state]
        callback.on_load_checkpoint(trainer, module, checkpoint)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(valid_scheduler_state)
        callback.on_train_start(trainer, module)
        optimizer.step()
        scheduler.step()
        self.assertEqual(scheduler.last_epoch, 1)

    def test_cyclic_scheduler_failure_restores_callable_and_full_payload(
        self,
    ) -> None:
        class StatefulScale:
            def __init__(self) -> None:
                self.factor = 0.5

            def __call__(self, cycle: float) -> float:
                return self.factor**cycle

        parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        scale = StatefulScale()
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=2,
            scale_fn=scale,
            cycle_momentum=False,
        )
        valid_scheduler_state = scheduler.state_dict()
        malformed_scheduler_state = copy.deepcopy(valid_scheduler_state)
        malformed_scheduler_state["_scale_fn_custom"] = [("factor", 0.25), 7]
        malformed_scheduler_state["unexpected_partial_load_value"] = "restore"
        original_payload = copy.deepcopy(malformed_scheduler_state)
        original_scheduler_snapshot = self.scheduler_identity_snapshot(scheduler)
        reconciler = NeuronSchedulerCheckpointReconciler()
        binding = SchedulerGroupLoadBinding(
            scheduler=scheduler,
            saved_state=malformed_scheduler_state,
            optimizer=optimizer,
            policy=None,
            target_group_count=1,
        )

        reconciler.prepare_for_load([binding])
        with self.assertRaises(TypeError):
            scheduler.load_state_dict(malformed_scheduler_state)
        self.assertNotIn("_scale_fn_custom", malformed_scheduler_state)
        self.assertEqual(scale.factor, 0.25)

        reconciler.clear()

        self.assert_scheduler_matches_identity_snapshot(
            scheduler,
            original_scheduler_snapshot,
        )
        self.assertEqual(scale.factor, 0.5)
        self.assertEqual(malformed_scheduler_state, original_payload)
        self.assertIn("_scale_fn_custom", malformed_scheduler_state)

        retry_binding = SchedulerGroupLoadBinding(
            scheduler=scheduler,
            saved_state=valid_scheduler_state,
            optimizer=optimizer,
            policy=None,
            target_group_count=1,
        )
        reconciler.prepare_for_load([retry_binding])
        scheduler.load_state_dict(valid_scheduler_state)
        reconciler.mark_optimizer_loaded(optimizer)
        reconciler.commit_loaded()
        optimizer.step()
        scheduler.step()
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(scale.factor, 0.5)

    def test_scheduler_mutation_transaction_rolls_back_or_commits(self) -> None:
        class StatefulSchedule:
            def __init__(self) -> None:
                self.factor = 0.5

            def __call__(self, step: int) -> float:
                return self.factor**step

        parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        schedule = StatefulSchedule()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        original_base_lrs = scheduler.base_lrs
        original_lambdas = scheduler.lr_lambdas
        transaction = NeuronSchedulerMutationTransaction()

        transaction.prepare([scheduler])
        scheduler.base_lrs.append(0.2)
        scheduler.lr_lambdas.append(schedule)
        scheduler.last_epoch = 9
        scheduler.temporary_value = "remove"
        schedule.factor = 0.25
        transaction.clear()

        self.assertIs(scheduler.base_lrs, original_base_lrs)
        self.assertEqual(scheduler.base_lrs, [0.1])
        self.assertIs(scheduler.lr_lambdas, original_lambdas)
        self.assertEqual(scheduler.lr_lambdas, [schedule])
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertFalse(hasattr(scheduler, "temporary_value"))
        self.assertEqual(schedule.factor, 0.5)

        transaction.prepare([scheduler])
        scheduler.last_epoch = 3
        transaction.commit()
        transaction.clear()

        self.assertEqual(scheduler.last_epoch, 3)

    def test_non_fit_checkpoint_does_not_leak_into_later_fit(self) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        grown_neuron = self.grow_once(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        original_order = list(optimizer.param_groups[0]["params"])
        legacy_state, _ = self.legacy_state_for_one_grown_neuron(
            module,
            optimizer,
            grown_neuron,
        )
        trainer = FakeTrainer([])
        trainer.state = SimpleNamespace(fn=TrainerFn.VALIDATING)
        callback = NeuronClusterOptimizerSyncCallback()

        callback.on_load_checkpoint(
            trainer,
            module,
            {"optimizer_states": [legacy_state]},
        )
        trainer.state.fn = TrainerFn.FITTING
        trainer.optimizers = [optimizer]
        callback.on_fit_start(trainer, module)

        self.assertIsNone(callback._pending_saved_optimizer_states)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["params"], original_order)

    def test_legacy_reconciliation_allows_a_pruned_non_entry_initial_neuron(
        self,
    ) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=3,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=2,
            max_steps=1,
            growth_threshold=1,
            pruning_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        self.assertEqual(cluster.entry_coordinates.tolist(), [[1, 1, 1]])
        del cluster.cluster["neuron_1_1_2"]
        grown_neuron = cluster._initialize_neuron(1, 1, 3)
        cluster.cluster["neuron_1_1_3"] = grown_neuron
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        legacy_state, expected_state = self.legacy_state_for_one_grown_neuron(
            module,
            optimizer,
            grown_neuron,
        )
        reconciler = NeuronOptimizerCheckpointReconciler()

        reconciler.prepare_for_load([optimizer], [cluster], [legacy_state])
        optimizer.load_state_dict(legacy_state)
        self.assertTrue(reconciler.complete_optimizer_load(optimizer))

        self.assertEqual(len(optimizer.param_groups), 2)
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(
                optimizer.state[parameter]["exp_avg"],
                expected_state[name],
            )

    def test_sync_appends_to_reference_group_without_changing_group_count(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=0.012,
            weight_decay=0.034,
            betas=(0.8, 0.9),
        )
        trainer = FakeTrainer([optimizer])
        self.grow_once(cluster)

        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        self.assertEqual(len(optimizer.param_groups), 1)
        reference_group = optimizer.param_groups[0]
        self.assertEqual(reference_group["lr"], 0.012)
        self.assertEqual(reference_group["weight_decay"], 0.034)
        self.assertEqual(reference_group["betas"], (0.8, 0.9))
        self.assertEqual(
            [id(parameter) for parameter in reference_group["params"]],
            [id(parameter) for parameter in module.parameters()],
        )

    def test_sync_preserves_parameter_role_group_options(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        nucleus_params = []
        terminal_params = []
        static_params = []
        for name, parameter in module.named_parameters():
            if ".cluster.neuron_" not in f".{name}":
                static_params.append(parameter)
            elif ".nucleus." in name:
                nucleus_params.append(parameter)
            else:
                terminal_params.append(parameter)
        optimizer = torch.optim.Adam(
            [
                {"params": nucleus_params, "lr": 0.001, "weight_decay": 0.01},
                {"params": terminal_params, "lr": 0.002, "weight_decay": 0.02},
                {"params": static_params, "lr": 0.003, "weight_decay": 0.03},
            ]
        )
        trainer = FakeTrainer([optimizer])

        new_neuron = self.grow_once(cluster)
        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        nucleus_ids = {id(parameter) for parameter in new_neuron.nucleus.parameters()}
        terminal_ids = {id(parameter) for parameter in new_neuron.terminal.parameters()}
        group_ids = [
            {id(parameter) for parameter in group["params"]}
            for group in optimizer.param_groups
        ]
        self.assertTrue(nucleus_ids.issubset(group_ids[0]))
        self.assertTrue(terminal_ids.issubset(group_ids[1]))
        self.assertTrue(nucleus_ids.isdisjoint(group_ids[1]))
        self.assertTrue(terminal_ids.isdisjoint(group_ids[0]))
        self.assertEqual(
            [(group["lr"], group["weight_decay"]) for group in optimizer.param_groups],
            [(0.001, 0.01), (0.002, 0.02), (0.003, 0.03)],
        )

    def test_late_role_param_names_failure_rolls_back_growth_and_retries(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        nucleus_parameters = []
        terminal_parameters = []
        static_parameters = []
        for name, parameter in module.named_parameters():
            if ".cluster.neuron_" not in f".{name}":
                static_parameters.append(parameter)
            elif ".nucleus." in name:
                nucleus_parameters.append(parameter)
            else:
                terminal_parameters.append(parameter)

        def named_group(parameters):
            return {
                "params": parameters,
                "param_names": [
                    names_by_parameter_id[id(parameter)] for parameter in parameters
                ],
            }

        optimizer = torch.optim.Adam(
            [
                named_group(nucleus_parameters),
                named_group(terminal_parameters),
                named_group(static_parameters),
            ],
            lr=0.012,
        )
        malformed_names = optimizer.param_groups[1]["param_names"]
        malformed_names.pop()
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        grown_neuron = self.grow_once(cluster)
        grown_parameter_ids = {
            id(parameter)
            for parameter in grown_neuron.parameters()
            if parameter.requires_grad
        }
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)

        with self.assertRaisesRegex(RuntimeError, "param_names"):
            callback.sync_optimizers(trainer, module)

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )
        self.assertTrue(
            grown_parameter_ids.isdisjoint(self.optimizer_param_ids(optimizer))
        )

        malformed_names[:] = [
            names_by_parameter_id[id(parameter)]
            for parameter in optimizer.param_groups[1]["params"]
        ]
        callback.sync_optimizers(trainer, module)

        current_names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        self.assertTrue(
            grown_parameter_ids.issubset(self.optimizer_param_ids(optimizer))
        )
        for group in optimizer.param_groups:
            self.assertEqual(len(group["params"]), len(group["param_names"]))
            self.assertEqual(
                group["param_names"],
                [
                    current_names_by_parameter_id[id(parameter)]
                    for parameter in group["params"]
                ],
            )

    def test_aligned_noncanonical_param_names_reject_growth_atomically(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        parameters = list(module.parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": parameters,
                    "param_names": [
                        f"custom_parameter_{index}" for index in range(len(parameters))
                    ],
                }
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        grown_neuron = self.grow_once(cluster)
        grown_parameter_ids = {id(parameter) for parameter in grown_neuron.parameters()}
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)

        with self.assertRaisesRegex(
            RuntimeError,
            "fully-qualified module names",
        ):
            callback.sync_optimizers(trainer, module)

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )
        self.assertTrue(
            grown_parameter_ids.isdisjoint(self.optimizer_param_ids(optimizer))
        )

    def test_hidden_grown_parameter_is_rejected_before_any_registration(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = HiddenParameterLightningModule(cluster)
        named_parameters = list(module.named_parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameter for _, parameter in named_parameters],
                    "param_names": [name for name, _ in named_parameters],
                }
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        grown_neuron = self.grow_once(cluster)
        grown_parameters = list(grown_neuron.parameters())
        module.hidden_parameter_ids.add(id(grown_parameters[0]))
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)

        with self.assertRaisesRegex(
            RuntimeError,
            "not registered on the Lightning module",
        ):
            callback.sync_optimizers(trainer, module)

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )
        self.assertTrue(
            {id(parameter) for parameter in grown_parameters}.isdisjoint(
                self.optimizer_param_ids(optimizer)
            )
        )

    def test_sync_assigns_each_grown_parameter_to_only_its_existing_optimizer(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        nucleus_params = []
        terminal_params = []
        for name, parameter in cluster.named_parameters():
            if name.startswith("cluster.neuron_") and ".nucleus." in name:
                nucleus_params.append(parameter)
            elif name.startswith("cluster.neuron_"):
                terminal_params.append(parameter)
        nucleus_optimizer = torch.optim.Adam(nucleus_params, lr=0.001)
        terminal_optimizer = torch.optim.Adam(terminal_params, lr=0.002)
        trainer = FakeTrainer([nucleus_optimizer, terminal_optimizer])

        new_neuron = self.grow_once(cluster)
        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        nucleus_ids = {id(parameter) for parameter in new_neuron.nucleus.parameters()}
        terminal_ids = {id(parameter) for parameter in new_neuron.terminal.parameters()}
        nucleus_optimizer_ids = self.optimizer_param_ids(nucleus_optimizer)
        terminal_optimizer_ids = self.optimizer_param_ids(terminal_optimizer)
        self.assertTrue(nucleus_ids.issubset(nucleus_optimizer_ids))
        self.assertTrue(terminal_ids.issubset(terminal_optimizer_ids))
        self.assertTrue(nucleus_ids.isdisjoint(terminal_optimizer_ids))
        self.assertTrue(terminal_ids.isdisjoint(nucleus_optimizer_ids))

    def test_legacy_growth_preserves_role_split_across_two_optimizers(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        nucleus_parameters = []
        other_neuron_parameters = []
        for name, parameter in cluster.named_parameters():
            if not name.startswith("cluster.neuron_"):
                continue
            target = (
                nucleus_parameters if ".nucleus." in name else other_neuron_parameters
            )
            target.append(parameter)
        nucleus_optimizer = torch.optim.Adam(nucleus_parameters, lr=0.001)
        other_neuron_optimizer = torch.optim.Adam(
            other_neuron_parameters,
            lr=0.002,
        )
        trainer = FakeTrainer([nucleus_optimizer, other_neuron_optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            nucleus_optimizer,
            LegacyOptimizerAppendPolicy(1, 0),
        )
        callback._record_legacy_append_policy(
            other_neuron_optimizer,
            LegacyOptimizerAppendPolicy(1, 0),
        )

        callback.sync_optimizers(trainer, module)

        nucleus_optimizer_ids = self.optimizer_param_ids(nucleus_optimizer)
        other_optimizer_ids = self.optimizer_param_ids(other_neuron_optimizer)
        self.assertTrue(nucleus_optimizer_ids.isdisjoint(other_optimizer_ids))
        self.assertEqual(len(nucleus_optimizer.param_groups), 1)
        self.assertEqual(len(other_neuron_optimizer.param_groups), 1)

        new_neuron = self.grow_once(cluster)
        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        nucleus_optimizer_ids = self.optimizer_param_ids(nucleus_optimizer)
        other_optimizer_ids = self.optimizer_param_ids(other_neuron_optimizer)
        new_nucleus_ids = {
            id(parameter) for parameter in new_neuron.nucleus.parameters()
        }
        new_other_ids = {
            id(parameter)
            for part in (new_neuron.axons, new_neuron.terminal)
            for parameter in part.parameters()
        }
        self.assertTrue(new_nucleus_ids.issubset(nucleus_optimizer_ids))
        self.assertTrue(new_other_ids.issubset(other_optimizer_ids))
        self.assertTrue(new_nucleus_ids.isdisjoint(other_optimizer_ids))
        self.assertTrue(new_other_ids.isdisjoint(nucleus_optimizer_ids))
        self.assertTrue(nucleus_optimizer_ids.isdisjoint(other_optimizer_ids))
        self.assertEqual(len(nucleus_optimizer.param_groups), 2)
        self.assertEqual(len(other_neuron_optimizer.param_groups), 2)

    def test_sync_skips_optimizers_without_existing_cluster_params(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        cluster_optimizer = torch.optim.Adam(module.neuron_cluster.parameters())
        other_optimizer = torch.optim.Adam(module.other.parameters())
        trainer = FakeTrainer([cluster_optimizer, other_optimizer])

        new_neuron = self.grow_once(cluster)
        new_param_ids = {
            id(parameter)
            for parameter in new_neuron.parameters()
            if parameter.requires_grad
        }

        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        self.assertTrue(
            new_param_ids.issubset(self.optimizer_param_ids(cluster_optimizer))
        )
        self.assertTrue(
            new_param_ids.isdisjoint(self.optimizer_param_ids(other_optimizer))
        )

    def test_sync_warns_when_no_optimizer_owns_cluster_params(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        other_optimizer = torch.optim.Adam(module.other.parameters())
        trainer = FakeTrainer([other_optimizer])

        with self.assertWarns(UserWarning):
            NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        self.assertEqual(len(other_optimizer.param_groups), 1)

    def test_shared_cluster_alias_is_reported_once_and_not_double_processed(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = AliasedClusterLightningModule(cluster)
        optimizer = torch.optim.Adam(module.other.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            callback.sync_optimizers(trainer, module)

        cluster_warnings = [
            warning
            for warning in captured_warnings
            if "trainable NeuronCluster parameters missing" in str(warning.message)
        ]
        self.assertEqual(len(cluster_warnings), 1)
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_sync_is_noop_without_optimizers_or_clusters(self) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        callback = NeuronClusterOptimizerSyncCallback()
        cluster_parameter_ids = [id(parameter) for parameter in cluster.parameters()]

        callback.sync_optimizers(FakeTrainer([]), module)

        self.assertEqual(
            [id(parameter) for parameter in cluster.parameters()],
            cluster_parameter_ids,
        )
        self.assertEqual(callback._synced_neuron_names, {})
        self.assertEqual(callback._synced_param_ids, {})

        plain_module = nn.Linear(self.input_dim, self.input_dim)
        optimizer = torch.optim.Adam(
            plain_module.parameters(),
            lr=0.007,
            weight_decay=0.013,
        )
        original_group = optimizer.param_groups[0]
        original_parameter_ids = [
            id(parameter) for parameter in original_group["params"]
        ]

        callback.sync_optimizers(FakeTrainer([optimizer]), plain_module)

        self.assertIs(optimizer.param_groups[0], original_group)
        self.assertEqual(
            [id(parameter) for parameter in original_group["params"]],
            original_parameter_ids,
        )
        self.assertEqual(original_group["lr"], 0.007)
        self.assertEqual(original_group["weight_decay"], 0.013)

    def test_sync_preserves_non_cluster_prefix_and_canonical_cluster_order(
        self,
    ) -> None:
        cluster = self.build_growing_cluster()
        module = PrefixLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        non_cluster_prefix = [id(parameter) for parameter in module.other.parameters()]
        self.grow_once(cluster)

        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        synchronized_ids = [
            id(parameter) for parameter in optimizer.param_groups[0]["params"]
        ]
        self.assertEqual(
            synchronized_ids[: len(non_cluster_prefix)], non_cluster_prefix
        )
        self.assertEqual(
            synchronized_ids,
            [id(parameter) for parameter in module.parameters()],
        )

    def build_pruning_cluster(self, growth_threshold: int | None = None):
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=growth_threshold,
            pruning_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

    def plant_idle_neuron(self, cluster):
        """Adds a neuron outside the entry neuron's terminal range so the
        next training forward atrophies and prunes it."""
        cluster.cluster["neuron_5_1_1"] = cluster._initialize_neuron(5, 1, 1)
        return cluster.cluster["neuron_5_1_1"]

    def prune_once(self, cluster) -> None:
        names_before = set(cluster.cluster.keys())
        cluster.train()
        cluster(torch.randn(self.batch_size, self.input_dim))
        pruned_names = names_before - set(cluster.cluster.keys())
        self.assertEqual(len(pruned_names), 1)

    def test_sync_removes_pruned_neuron_params_from_optimizer(self):
        cluster = self.build_pruning_cluster()
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)

        idle_param_ids = {
            id(parameter)
            for parameter in idle_neuron.parameters()
            if parameter.requires_grad
        }
        self.assertTrue(idle_param_ids.issubset(self.optimizer_param_ids(optimizer)))
        for parameter in idle_neuron.parameters():
            optimizer.state[parameter] = {"exp_avg": torch.zeros_like(parameter)}
        live_parameter = module.other.weight
        live_state = {"exp_avg": torch.ones_like(live_parameter)}
        optimizer.state[live_parameter] = live_state

        self.prune_once(cluster)
        callback.sync_optimizers(trainer, module)

        self.assertTrue(idle_param_ids.isdisjoint(self.optimizer_param_ids(optimizer)))
        self.assertTrue(
            all(id(parameter) not in idle_param_ids for parameter in optimizer.state)
        )
        self.assertIs(optimizer.state[live_parameter], live_state)
        torch.testing.assert_close(
            optimizer.state[live_parameter]["exp_avg"],
            torch.ones_like(live_parameter),
        )

    def test_named_stateless_sgd_prunes_neuron_and_preserves_external_parameter(
        self,
    ) -> None:
        cluster = self.build_pruning_cluster()
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        named_parameters = list(module.named_parameters())
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [parameter for _, parameter in named_parameters],
                    "param_names": [name for name, _ in named_parameters],
                }
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        idle_parameter_ids = {id(parameter) for parameter in idle_neuron.parameters()}
        external_parameter = module.other.weight
        self.assertFalse(optimizer.state)

        self.prune_once(cluster)
        callback.sync_optimizers(trainer, module)

        optimizer_parameter_ids = self.optimizer_param_ids(optimizer)
        self.assertTrue(idle_parameter_ids.isdisjoint(optimizer_parameter_ids))
        self.assertIn(id(external_parameter), optimizer_parameter_ids)
        self.assertFalse(optimizer.state)
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

        external_value_before_step = external_parameter.detach().clone()
        loss = sum(
            parameter.sum()
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        loss.backward()
        optimizer.step()

        torch.testing.assert_close(
            external_parameter,
            external_value_before_step - optimizer.param_groups[0]["lr"],
        )
        self.assertFalse(optimizer.state)

    def test_misaligned_param_names_reject_pruning_without_partial_mutation(
        self,
    ) -> None:
        cluster = self.build_pruning_cluster()
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        named_parameters = list(module.named_parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameter for _, parameter in named_parameters],
                    "param_names": [name for name, _ in named_parameters],
                }
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        idle_parameters = list(idle_neuron.parameters())
        for parameter in idle_parameters:
            optimizer.state[parameter] = {"exp_avg": torch.ones_like(parameter)}
        optimizer.param_groups[0]["param_names"].pop()
        optimizer_snapshot = self.optimizer_identity_snapshot(optimizer)

        self.prune_once(cluster)
        with self.assertRaisesRegex(RuntimeError, "param_names are not aligned"):
            callback.sync_optimizers(trainer, module)

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot,
        )
        self.assertTrue(
            all(parameter in optimizer.state for parameter in idle_parameters)
        )

    def test_prune_preserves_parameter_tied_to_live_external_module(self) -> None:
        cluster = self.build_pruning_cluster()
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        shared_parameter = module.other.weight
        idle_neuron.nucleus.model.weight = shared_parameter
        named_parameters = list(module.named_parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameter for _, parameter in named_parameters],
                    "param_names": [name for name, _ in named_parameters],
                }
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        shared_state = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.ones_like(shared_parameter),
            "exp_avg_sq": torch.full_like(shared_parameter, 2.0),
        }
        optimizer.state[shared_parameter] = shared_state

        self.prune_once(cluster)
        callback.sync_optimizers(trainer, module)

        self.assertIs(module.other.weight, shared_parameter)
        self.assertIn(id(shared_parameter), self.optimizer_param_ids(optimizer))
        self.assertIs(optimizer.state[shared_parameter], shared_state)
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

    def test_prune_refreshes_names_when_all_parameters_remain_externally_tied(
        self,
    ) -> None:
        cluster = self.build_pruning_cluster()
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        externally_tied_parameters = list(idle_neuron.parameters())
        externally_tied_parameter_ids = {
            id(parameter) for parameter in externally_tied_parameters
        }
        module.external_parameters = nn.ParameterList(externally_tied_parameters)
        original_named_parameters = list(module.named_parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameter for _, parameter in original_named_parameters],
                    "param_names": [name for name, _ in original_named_parameters],
                }
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        original_parameter_ids = self.optimizer_param_ids(optimizer)

        self.prune_once(cluster)
        callback.sync_optimizers(trainer, module)

        self.assertEqual(self.optimizer_param_ids(optimizer), original_parameter_ids)
        self.assertTrue(
            all(
                parameter_id in self.optimizer_param_ids(optimizer)
                for parameter_id in externally_tied_parameter_ids
            )
        )
        current_names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        self.assertEqual(
            optimizer.param_groups[0]["param_names"],
            [
                current_names_by_parameter_id[id(parameter)]
                for parameter in optimizer.param_groups[0]["params"]
            ],
        )
        self.assertTrue(
            all(
                name.startswith("external_parameters.")
                for name, parameter in zip(
                    optimizer.param_groups[0]["param_names"],
                    optimizer.param_groups[0]["params"],
                    strict=True,
                )
                if id(parameter) in externally_tied_parameter_ids
            )
        )

    def test_topology_prune_refreshes_names_when_all_ids_survive_in_cluster(
        self,
    ) -> None:
        cluster = self.build_pruning_cluster()
        shared_neuron = cluster.cluster["neuron_3_1_1"]
        cluster.cluster.clear()
        cluster.cluster["neuron_5_1_1"] = shared_neuron
        cluster.cluster["neuron_3_1_1"] = shared_neuron
        module = FakeLightningModule(cluster)
        original_named_parameters = list(module.named_parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameter for _, parameter in original_named_parameters],
                    "param_names": [name for name, _ in original_named_parameters],
                }
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        original_parameter_ids = self.optimizer_param_ids(optimizer)
        self.assertTrue(
            any(
                name.startswith("neuron_cluster.cluster.neuron_5_1_1.")
                for name in optimizer.param_groups[0]["param_names"]
            )
        )

        del cluster.cluster["neuron_5_1_1"]
        callback.sync_optimizers(trainer, module)

        self.assertEqual(self.optimizer_param_ids(optimizer), original_parameter_ids)
        current_names_by_parameter_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        self.assertEqual(
            optimizer.param_groups[0]["param_names"],
            [
                current_names_by_parameter_id[id(parameter)]
                for parameter in optimizer.param_groups[0]["params"]
            ],
        )
        self.assertFalse(
            any(
                name.startswith("neuron_cluster.cluster.neuron_5_1_1.")
                for name in optimizer.param_groups[0]["param_names"]
            )
        )

    def test_sync_drops_empty_param_group_after_prune(self):
        cluster = self.build_pruning_cluster()
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        names_by_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        idle_param_ids = {id(parameter) for parameter in idle_neuron.parameters()}
        live_parameters = [
            parameter
            for parameter in module.parameters()
            if id(parameter) not in idle_param_ids
        ]
        idle_parameters = list(idle_neuron.parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": live_parameters,
                    "param_names": [
                        names_by_id[id(parameter)] for parameter in live_parameters
                    ],
                },
                {
                    "params": idle_parameters,
                    "param_names": [
                        names_by_id[id(parameter)] for parameter in idle_parameters
                    ],
                },
            ],
            lr=0.012,
        )
        trainer = FakeTrainer([optimizer])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [lambda step: 0.5**step, lambda step: 0.8**step],
        )
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        self.assertEqual(len(optimizer.param_groups), 2)

        self.prune_once(cluster)
        callback.sync_optimizers(trainer, module)

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertTrue(all(group["params"] for group in optimizer.param_groups))
        self.assertEqual(
            optimizer.param_groups[0]["param_names"],
            [
                names_by_id[id(parameter)]
                for parameter in optimizer.param_groups[0]["params"]
            ],
        )
        self.assertEqual(len(scheduler.base_lrs), 1)
        self.assertEqual(len(scheduler.lr_lambdas), 1)
        optimizer.step()
        scheduler.step()

    def test_prune_preserves_preexisting_unrelated_empty_optimizer_group(
        self,
    ) -> None:
        cluster = self.build_pruning_cluster()
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        idle_parameter_ids = {id(parameter) for parameter in idle_neuron.parameters()}
        live_parameters = [
            parameter
            for parameter in module.parameters()
            if id(parameter) not in idle_parameter_ids
        ]
        idle_parameters = list(idle_neuron.parameters())
        optimizer = torch.optim.SGD(
            [
                {"params": live_parameters, "lr": 0.1},
                {"params": [], "lr": 0.2},
                {"params": idle_parameters, "lr": 0.3},
            ]
        )
        schedules = [
            lambda step: 0.5**step,
            lambda step: 0.6**step,
            lambda step: 0.7**step,
        ]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)
        reserved_group = optimizer.param_groups[1]
        reserved_schedule = scheduler.lr_lambdas[1]
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)

        self.prune_once(cluster)
        callback.sync_optimizers(trainer, module)

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertIs(optimizer.param_groups[1], reserved_group)
        self.assertEqual(optimizer.param_groups[1]["params"], [])
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.1, 0.2],
        )
        self.assertEqual(len(scheduler.lr_lambdas), 2)
        self.assertIs(scheduler.lr_lambdas[1], reserved_schedule)
        optimizer.step()
        scheduler.step()

    def test_prune_with_unknown_scheduler_is_atomic_and_retryable(self) -> None:
        cluster = self.build_pruning_cluster()
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        idle_parameters = list(idle_neuron.parameters())
        idle_parameter_ids = {id(parameter) for parameter in idle_parameters}
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [
                        parameter
                        for parameter in module.parameters()
                        if id(parameter) not in idle_parameter_ids
                    ]
                },
                {"params": idle_parameters},
            ],
            lr=0.012,
        )
        supported_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [lambda step: 0.5**step, lambda step: 0.8**step],
        )
        unsupported_scheduler = UnsupportedLambdaLR(
            optimizer,
            [lambda step: 0.4**step, lambda step: 0.7**step],
        )
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = [
            SimpleNamespace(scheduler=supported_scheduler),
            SimpleNamespace(scheduler=unsupported_scheduler),
        ]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        for parameter in idle_parameters:
            optimizer.state[parameter] = {"exp_avg": torch.ones_like(parameter)}
        original_groups = list(optimizer.param_groups)
        original_group_parameters = [
            list(group["params"]) for group in optimizer.param_groups
        ]

        self.prune_once(cluster)
        with self.assertRaisesRegex(RuntimeError, "unrecognized scheduler"):
            callback.sync_optimizers(trainer, module)

        self.assertEqual(len(optimizer.param_groups), len(original_groups))
        for group, original_group in zip(
            optimizer.param_groups,
            original_groups,
            strict=True,
        ):
            self.assertIs(group, original_group)
        self.assertEqual(
            [group["params"] for group in optimizer.param_groups],
            original_group_parameters,
        )
        self.assertTrue(
            all(parameter in optimizer.state for parameter in idle_parameters)
        )
        self.assertEqual(len(supported_scheduler.base_lrs), 2)
        self.assertEqual(len(supported_scheduler.lr_lambdas), 2)

        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=supported_scheduler)]
        callback.sync_optimizers(trainer, module)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertTrue(
            all(parameter not in optimizer.state for parameter in idle_parameters)
        )
        self.assertEqual(len(supported_scheduler.base_lrs), 1)
        self.assertEqual(len(supported_scheduler.lr_lambdas), 1)

    def test_later_optimizer_prune_failure_rolls_back_all_optimizers(self) -> None:
        cluster = self.build_pruning_cluster()
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        idle_parameter_ids = {id(parameter) for parameter in idle_neuron.parameters()}
        live_nucleus_parameters = []
        idle_nucleus_parameters = []
        live_other_parameters = []
        idle_other_parameters = []
        for name, parameter in cluster.named_parameters():
            is_idle = id(parameter) in idle_parameter_ids
            if ".nucleus." in name:
                target = idle_nucleus_parameters if is_idle else live_nucleus_parameters
            else:
                target = idle_other_parameters if is_idle else live_other_parameters
            target.append(parameter)
        first_optimizer = torch.optim.Adam(
            [
                {"params": live_nucleus_parameters},
                {"params": idle_nucleus_parameters},
            ],
            lr=0.011,
        )
        second_optimizer = torch.optim.Adam(
            [
                {"params": live_other_parameters},
                {"params": idle_other_parameters},
            ],
            lr=0.022,
        )
        first_scheduler = torch.optim.lr_scheduler.LambdaLR(
            first_optimizer,
            [lambda step: 0.5**step, lambda step: 0.6**step],
        )
        unsupported_scheduler = UnsupportedLambdaLR(
            second_optimizer,
            [lambda step: 0.7**step, lambda step: 0.8**step],
        )
        trainer = FakeTrainer([first_optimizer, second_optimizer])
        trainer.lr_scheduler_configs = [
            SimpleNamespace(scheduler=first_scheduler),
            SimpleNamespace(scheduler=unsupported_scheduler),
        ]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        for optimizer, stale_parameters in (
            (first_optimizer, idle_nucleus_parameters),
            (second_optimizer, idle_other_parameters),
        ):
            for index, parameter in enumerate(stale_parameters, start=1):
                optimizer.state[parameter] = {
                    "sentinel": torch.full_like(parameter, float(index))
                }
        optimizer_snapshots = [
            self.optimizer_identity_snapshot(optimizer)
            for optimizer in trainer.optimizers
        ]
        scheduler_snapshots = [
            self.scheduler_identity_snapshot(scheduler)
            for scheduler in (first_scheduler, unsupported_scheduler)
        ]

        self.prune_once(cluster)
        with self.assertRaisesRegex(RuntimeError, "unrecognized scheduler"):
            callback.sync_optimizers(trainer, module)

        for optimizer, snapshot in zip(
            trainer.optimizers,
            optimizer_snapshots,
            strict=True,
        ):
            self.assert_optimizer_matches_identity_snapshot(optimizer, snapshot)
        for scheduler, snapshot in zip(
            (first_scheduler, unsupported_scheduler),
            scheduler_snapshots,
            strict=True,
        ):
            self.assert_scheduler_matches_identity_snapshot(scheduler, snapshot)

        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=first_scheduler)]
        callback.sync_optimizers(trainer, module)

        self.assertEqual(len(first_optimizer.param_groups), 1)
        self.assertEqual(len(second_optimizer.param_groups), 1)
        self.assertEqual(len(first_scheduler.base_lrs), 1)
        self.assertEqual(len(first_scheduler.lr_lambdas), 1)
        for optimizer in trainer.optimizers:
            self.assertTrue(
                all(
                    id(parameter) not in idle_parameter_ids
                    for parameter in optimizer.state
                )
            )

    def test_grow_and_prune_in_same_interval_triggers_resync(self):
        cluster = self.build_pruning_cluster(growth_threshold=1)
        idle_neuron = self.plant_idle_neuron(cluster)
        module = FakeLightningModule(cluster)
        initial_named_parameters = list(module.named_parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameter for _, parameter in initial_named_parameters],
                    "param_names": [name for name, _ in initial_named_parameters],
                }
            ],
            lr=0.012,
            betas=(0.81, 0.92),
            weight_decay=0.034,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=2,
            gamma=0.5,
        )
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        optimizer_group = optimizer.param_groups[0]
        group_options_before = {
            name: value
            for name, value in optimizer_group.items()
            if name not in {"params", "param_names"}
        }
        scheduler_metadata_before = (
            tuple(scheduler.base_lrs),
            tuple(scheduler.get_last_lr()),
            scheduler.last_epoch,
        )
        idle_parameters = list(idle_neuron.parameters())
        idle_parameter_ids = {id(parameter) for parameter in idle_parameters}
        for index, parameter in enumerate(idle_parameters, start=1):
            optimizer.state[parameter] = {
                "pruned_sentinel": torch.full_like(parameter, float(index))
            }
        neuron_names_before = set(cluster.cluster)
        neuron_count_before = len(cluster.cluster)

        cluster.train()
        cluster(torch.randn(self.batch_size, self.input_dim))
        grown_neuron_names = set(cluster.cluster) - neuron_names_before
        self.assertEqual(len(grown_neuron_names), 1)
        self.assertEqual(neuron_names_before - set(cluster.cluster), {"neuron_5_1_1"})
        self.assertEqual(len(cluster.cluster), neuron_count_before)
        grown_neuron = cluster.cluster[grown_neuron_names.pop()]
        grown_parameter_ids = {id(parameter) for parameter in grown_neuron.parameters()}
        pre_hook_parameter_ids = self.optimizer_param_ids(optimizer)
        self.assertTrue(idle_parameter_ids.issubset(pre_hook_parameter_ids))
        self.assertTrue(grown_parameter_ids.isdisjoint(pre_hook_parameter_ids))

        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        current_named_parameters = list(module.named_parameters())
        synchronized_parameter_ids = [
            id(parameter) for parameter in optimizer_group["params"]
        ]
        self.assertIs(optimizer.param_groups[0], optimizer_group)
        self.assertEqual(
            synchronized_parameter_ids,
            [id(parameter) for _, parameter in current_named_parameters],
        )
        self.assertEqual(
            optimizer_group["param_names"],
            [name for name, _ in current_named_parameters],
        )
        self.assertTrue(idle_parameter_ids.isdisjoint(synchronized_parameter_ids))
        self.assertTrue(grown_parameter_ids.issubset(synchronized_parameter_ids))
        self.assertTrue(
            all(parameter not in optimizer.state for parameter in idle_parameters)
        )
        self.assertEqual(
            {
                name: value
                for name, value in optimizer_group.items()
                if name not in {"params", "param_names"}
            },
            group_options_before,
        )
        self.assertEqual(
            (
                tuple(scheduler.base_lrs),
                tuple(scheduler.get_last_lr()),
                scheduler.last_epoch,
            ),
            scheduler_metadata_before,
        )

    def test_batch_hooks_skip_rescan_without_growth(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        initial_named_parameters = list(module.named_parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameter for _, parameter in initial_named_parameters],
                    "param_names": [name for name, _ in initial_named_parameters],
                }
            ],
            lr=0.012,
            weight_decay=0.034,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        trainer = FakeTrainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        optimizer_group = optimizer.param_groups[0]
        group_options_before = {
            name: value
            for name, value in optimizer_group.items()
            if name not in {"params", "param_names"}
        }
        probe_parameter = next(iter(cluster.cluster.values())).nucleus.model.weight
        probe_index = next(
            index
            for index, parameter in enumerate(optimizer_group["params"])
            if parameter is probe_parameter
        )
        probe_name = optimizer_group["param_names"][probe_index]
        self.assertIs(optimizer_group["params"].pop(probe_index), probe_parameter)
        self.assertEqual(optimizer_group["param_names"].pop(probe_index), probe_name)
        probe_state = {"sentinel": torch.ones_like(probe_parameter)}
        optimizer.state[probe_parameter] = probe_state
        optimizer_snapshot_without_probe = self.optimizer_identity_snapshot(optimizer)
        scheduler_snapshot = self.scheduler_identity_snapshot(scheduler)

        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        self.assert_optimizer_matches_identity_snapshot(
            optimizer,
            optimizer_snapshot_without_probe,
        )
        self.assert_scheduler_matches_identity_snapshot(scheduler, scheduler_snapshot)
        self.assertNotIn(id(probe_parameter), self.optimizer_param_ids(optimizer))

        new_neuron = self.grow_once(cluster)
        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=1,
        )

        new_param_ids = {
            id(parameter)
            for parameter in new_neuron.parameters()
            if parameter.requires_grad
        }
        synchronized_parameter_ids = [
            id(parameter) for parameter in optimizer_group["params"]
        ]
        current_named_parameters = list(module.named_parameters())
        self.assertEqual(
            synchronized_parameter_ids,
            [id(parameter) for _, parameter in current_named_parameters],
        )
        self.assertEqual(
            len(synchronized_parameter_ids), len(set(synchronized_parameter_ids))
        )
        self.assertIn(id(probe_parameter), synchronized_parameter_ids)
        self.assertTrue(new_param_ids.issubset(synchronized_parameter_ids))
        self.assertIs(optimizer.state[probe_parameter], probe_state)
        self.assertEqual(
            optimizer_group["param_names"],
            [name for name, _ in current_named_parameters],
        )
        self.assertEqual(
            {
                name: value
                for name, value in optimizer_group.items()
                if name not in {"params", "param_names"}
            },
            group_options_before,
        )
        self.assert_scheduler_matches_identity_snapshot(scheduler, scheduler_snapshot)

    def test_named_reload_preserves_legacy_lambda_lineage_after_prune_and_growth(
        self,
    ) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=3,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=2,
            max_steps=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        dynamic_neuron = cluster._initialize_neuron(1, 1, 3)
        cluster.cluster["neuron_1_1_3"] = dynamic_neuron
        module = FakeLightningModule(cluster)
        pruned_neuron = cluster.cluster["neuron_1_1_2"]
        pruned_parameter_ids = {
            id(parameter) for parameter in pruned_neuron.parameters()
        }
        dynamic_parameter_ids = {
            id(parameter) for parameter in dynamic_neuron.parameters()
        }
        other_parameters = [
            parameter
            for parameter in module.parameters()
            if id(parameter) not in pruned_parameter_ids | dynamic_parameter_ids
        ]
        optimizer = torch.optim.SGD(
            [
                {"params": list(pruned_neuron.parameters()), "lr": 0.01},
                {"params": other_parameters, "lr": 0.02},
                {"params": list(dynamic_neuron.parameters()), "lr": 0.01},
            ],
            momentum=0.9,
        )

        def first_schedule(step: int) -> float:
            return 0.6**step

        def second_schedule(step: int) -> float:
            return 0.85**step

        base_schedules = (first_schedule, second_schedule)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [first_schedule, second_schedule, first_schedule],
        )
        trainer = FakeTrainer([optimizer])
        trainer.state = SimpleNamespace(fn=TrainerFn.FITTING)
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        callback._record_legacy_append_policy(
            optimizer,
            LegacyOptimizerAppendPolicy(2, 0),
        )

        del cluster.cluster["neuron_1_1_2"]
        callback.sync_optimizers(trainer, module)
        replacement_neuron = cluster._initialize_neuron(1, 1, 2)
        cluster.cluster["neuron_1_1_2"] = replacement_neuron
        callback.sync_optimizers(trainer, module)

        for parameter_index, (_, parameter) in enumerate(
            module.named_parameters(),
            start=1,
        ):
            parameter.grad = torch.full_like(parameter, parameter_index / 1000.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        checkpoint = {
            "optimizer_states": [copy.deepcopy(optimizer.state_dict())],
            "lr_schedulers": [copy.deepcopy(scheduler.state_dict())],
        }
        callback.on_save_checkpoint(trainer, module, checkpoint)
        checkpoint = copy.deepcopy(checkpoint)
        optimizer_layout = checkpoint["emperor_neuron_optimizer_layout"]["optimizers"][
            0
        ]
        serialized_lineage = optimizer_layout["legacy_group_reference_indices"]
        self.assertIsInstance(serialized_lineage, list)
        self.assertEqual(len(serialized_lineage), len(optimizer.param_groups))
        self.assertEqual(set(serialized_lineage), {0, 1})
        self.assertEqual(serialized_lineage[-1], 1)
        for group_schedule, reference_index in zip(
            scheduler.lr_lambdas,
            serialized_lineage,
            strict=True,
        ):
            self.assertIs(group_schedule, base_schedules[reference_index])

        resumed_module = copy.deepcopy(module)
        resumed_parameters = list(resumed_module.parameters())
        split_index = len(resumed_parameters) // 2
        resumed_optimizer = torch.optim.SGD(
            [
                {"params": resumed_parameters[:split_index], "lr": 0.01},
                {"params": resumed_parameters[split_index:], "lr": 0.02},
            ],
            momentum=0.9,
        )
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
            resumed_optimizer,
            [first_schedule, second_schedule],
        )
        resumed_trainer = FakeTrainer([resumed_optimizer])
        resumed_trainer.state = SimpleNamespace(fn=TrainerFn.FITTING)
        resumed_trainer.lr_scheduler_configs = [
            SimpleNamespace(scheduler=resumed_scheduler)
        ]
        resumed_callback = NeuronClusterOptimizerSyncCallback()
        resumed_callback.on_load_checkpoint(
            resumed_trainer,
            resumed_module,
            checkpoint,
        )
        resumed_optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        resumed_scheduler.load_state_dict(checkpoint["lr_schedulers"][0])
        resumed_callback.on_fit_start(resumed_trainer, resumed_module)

        self.assertEqual(len(resumed_scheduler.lr_lambdas), len(serialized_lineage))
        for group_schedule, reference_index in zip(
            resumed_scheduler.lr_lambdas,
            serialized_lineage,
            strict=True,
        ):
            self.assertIs(group_schedule, base_schedules[reference_index])
        source_names_by_id = {
            id(parameter): name for name, parameter in module.named_parameters()
        }
        resumed_names_by_id = {
            id(parameter): name for name, parameter in resumed_module.named_parameters()
        }
        self.assertEqual(
            [
                [source_names_by_id[id(parameter)] for parameter in group["params"]]
                for group in optimizer.param_groups
            ],
            [
                [resumed_names_by_id[id(parameter)] for parameter in group["params"]]
                for group in resumed_optimizer.param_groups
            ],
        )

        for continuation_step in range(2):
            for parameter_index, (
                (source_name, source),
                (resumed_name, resumed),
            ) in enumerate(
                zip(
                    module.named_parameters(),
                    resumed_module.named_parameters(),
                    strict=True,
                ),
                start=1,
            ):
                self.assertEqual(source_name, resumed_name)
                gradient_value = (continuation_step + 1) * parameter_index / 1000.0
                source.grad = torch.full_like(source, gradient_value)
                resumed.grad = torch.full_like(resumed, gradient_value)

            optimizer.step()
            resumed_optimizer.step()
            scheduler.step()
            resumed_scheduler.step()

            self.assertEqual(
                [group["lr"] for group in resumed_optimizer.param_groups],
                [group["lr"] for group in optimizer.param_groups],
            )
            for (source_name, source), (resumed_name, resumed) in zip(
                module.named_parameters(),
                resumed_module.named_parameters(),
                strict=True,
            ):
                self.assertEqual(source_name, resumed_name)
                torch.testing.assert_close(resumed, source, rtol=0, atol=0)

    def test_exception_clears_all_fit_lifecycle_state(self) -> None:
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        self.grow_once(cluster)
        callback.on_train_batch_end(
            trainer,
            module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )
        self.assertTrue(callback._clusters)
        self.assertTrue(callback._synced_neuron_names)
        self.assertTrue(callback._synced_param_ids)
        self.assertTrue(callback._post_wrap_param_ids)
        self.assertTrue(callback._fit_started)

        callback.on_exception(
            trainer,
            module,
            RuntimeError("training failed"),
        )

        self.assertEqual(callback._clusters, [])
        self.assertEqual(callback._synced_neuron_names, {})
        self.assertEqual(callback._synced_param_ids, {})
        self.assertEqual(callback._post_wrap_param_ids, set())
        self.assertFalse(callback._fit_started)


if __name__ == "__main__":
    unittest.main()
