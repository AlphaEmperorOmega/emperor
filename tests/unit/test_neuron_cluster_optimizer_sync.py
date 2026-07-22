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
from emperor.neuron._optimizer_layout import (
    OPTIMIZER_LAYOUT_CHECKPOINT_KEY,
    NeuronOptimizerNamedLayout,
)
from emperor.neuron._optimizer_scheduler import (
    NeuronSchedulerCheckpointReconciler,
    NeuronSchedulerMutationTransaction,
    SchedulerGroupLoadBinding,
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
