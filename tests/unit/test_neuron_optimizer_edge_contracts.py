from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn

from emperor.neuron import NeuronClusterOptimizerSyncCallback
from emperor.neuron._optimizer_layout import (
    OPTIMIZER_LAYOUT_CHECKPOINT_KEY,
    NeuronOptimizerNamedLayout,
)
from emperor.neuron._optimizer_scheduler import (
    NeuronSchedulerCheckpointReconciler,
    SchedulerGroupLoadBinding,
    preflight_scheduler_group_removal,
    remove_scheduler_groups,
)


class _ExplodingOnceDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._should_explode = True

    def __setitem__(self, name, value) -> None:
        if name == "params" and self._should_explode:
            self._should_explode = False
            raise RuntimeError("injected named-layout migration failure")
        super().__setitem__(name, value)


class TestNeuronOptimizerNamedLayoutEdges(unittest.TestCase):
    @staticmethod
    def layout_fixture(*, split_groups: bool = False):
        module = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(index)))
                for index, name in enumerate(("a", "b", "c"), start=1)
            }
        )
        if split_groups:
            optimizer = torch.optim.SGD(
                [
                    {"params": [module["a"]], "lr": 0.1},
                    {"params": [module["b"], module["c"]], "lr": 0.2},
                ]
            )
        else:
            optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        saved_state = optimizer.state_dict()
        layout = NeuronOptimizerNamedLayout.capture(
            module,
            [optimizer],
            [saved_state],
        )
        return module, optimizer, saved_state, layout

    def assert_prepare_rejected(
        self,
        module,
        optimizer,
        saved_state,
        layout,
        pattern: str,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, pattern):
            NeuronOptimizerNamedLayout().prepare_for_load(
                module,
                [optimizer],
                [saved_state],
                layout,
            )

    def test_second_optimizer_failure_rolls_back_every_saved_payload(self) -> None:
        module = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(index)))
                for index, name in enumerate(("a", "b", "c", "d"), start=1)
            }
        )
        source_optimizers = [
            torch.optim.SGD([module["a"], module["b"]], lr=0.1),
            torch.optim.SGD([module["c"], module["d"]], lr=0.2),
        ]
        saved_states = [optimizer.state_dict() for optimizer in source_optimizers]
        layout = NeuronOptimizerNamedLayout.capture(
            module,
            source_optimizers,
            saved_states,
        )
        target_optimizers = [
            torch.optim.SGD([module["b"], module["a"]], lr=0.1),
            torch.optim.SGD([module["d"], module["c"]], lr=0.2),
        ]
        original_saved_ids = [
            tuple(state["param_groups"][0]["params"]) for state in saved_states
        ]
        saved_states[1]["param_groups"][0] = _ExplodingOnceDict(
            saved_states[1]["param_groups"][0]
        )
        layout_manager = NeuronOptimizerNamedLayout()

        with self.assertRaisesRegex(
            RuntimeError,
            "injected named-layout migration failure",
        ):
            layout_manager.prepare_for_load(
                module,
                target_optimizers,
                saved_states,
                layout,
            )

        self.assertEqual(
            [tuple(state["param_groups"][0]["params"]) for state in saved_states],
            original_saved_ids,
        )
        self.assertTrue(
            all(
                not layout_manager.optimizer_requires_completion(optimizer)
                for optimizer in target_optimizers
            )
        )

    def test_completing_one_optimizer_leaves_the_other_pending(self) -> None:
        module = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(index)))
                for index, name in enumerate(("a", "b", "c", "d"), start=1)
            }
        )
        source_optimizers = [
            torch.optim.SGD([module["a"], module["b"]], lr=0.1, momentum=0.9),
            torch.optim.SGD([module["c"], module["d"]], lr=0.2, momentum=0.9),
        ]
        first_optimizer_parameter_ids = {id(module["a"]), id(module["b"])}
        for sentinel, parameter in enumerate(module.parameters(), start=11):
            owning_optimizer = (
                source_optimizers[0]
                if id(parameter) in first_optimizer_parameter_ids
                else source_optimizers[1]
            )
            owning_optimizer.state[parameter] = {
                "momentum_buffer": torch.full_like(parameter, float(sentinel))
            }
        saved_states = [optimizer.state_dict() for optimizer in source_optimizers]
        original_saved_ids = [
            tuple(state["param_groups"][0]["params"]) for state in saved_states
        ]
        layout = NeuronOptimizerNamedLayout.capture(
            module,
            source_optimizers,
            saved_states,
        )
        target_optimizers = [
            torch.optim.SGD([module["b"], module["a"]], lr=0.1, momentum=0.9),
            torch.optim.SGD([module["d"], module["c"]], lr=0.2, momentum=0.9),
        ]
        layout_manager = NeuronOptimizerNamedLayout()
        layout_manager.prepare_for_load(
            module,
            target_optimizers,
            saved_states,
            layout,
        )

        target_optimizers[0].load_state_dict(saved_states[0])
        layout_manager.complete_optimizer_load(target_optimizers[0])

        self.assertFalse(
            layout_manager.optimizer_requires_completion(target_optimizers[0])
        )
        self.assertTrue(
            layout_manager.optimizer_requires_completion(target_optimizers[1])
        )
        self.assertEqual(
            tuple(saved_states[0]["param_groups"][0]["params"]),
            original_saved_ids[0],
        )
        self.assertEqual(saved_states[1]["param_groups"][0]["params"], [1, 0])
        torch.testing.assert_close(
            target_optimizers[0].state[module["a"]]["momentum_buffer"],
            torch.tensor(11.0),
        )

        layout_manager.clear()
        self.assertEqual(
            tuple(saved_states[1]["param_groups"][0]["params"]),
            original_saved_ids[1],
        )

    def test_capture_rejects_unrepresentable_optimizer_layouts(self) -> None:
        module, optimizer, saved_state, _ = self.layout_fixture()
        cases = []

        cases.append(
            (
                "optimizer counts differ",
                lambda: NeuronOptimizerNamedLayout.capture(module, [optimizer], []),
            )
        )

        short_state = copy.deepcopy(saved_state)
        short_state["param_groups"][0]["params"].pop()
        cases.append(
            (
                "parameter-group size differs",
                lambda: NeuronOptimizerNamedLayout.capture(
                    module,
                    [optimizer],
                    [short_state],
                ),
            )
        )

        external_optimizer = torch.optim.SGD(
            [nn.Parameter(torch.tensor(4.0))],
            lr=0.1,
        )
        cases.append(
            (
                "registered on the Lightning module",
                lambda: NeuronOptimizerNamedLayout.capture(
                    module,
                    [external_optimizer],
                    [external_optimizer.state_dict()],
                ),
            )
        )

        for pattern, operation in cases:
            with (
                self.subTest(pattern=pattern),
                self.assertRaisesRegex(
                    RuntimeError,
                    pattern,
                ),
            ):
                operation()

    def test_prepare_rejects_malformed_and_retired_layouts(self) -> None:
        module, optimizer, saved_state, layout = self.layout_fixture()

        with self.assertRaisesRegex(RuntimeError, "Invalid named"):
            NeuronOptimizerNamedLayout().prepare_for_load(
                module,
                [optimizer],
                [saved_state],
                None,
            )

        wrong_version = copy.deepcopy(layout)
        wrong_version["version"] = 2
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            wrong_version,
            "Unsupported named",
        )

        retired_layout = copy.deepcopy(layout)
        retired_layout["optimizers"][0].update(
            {
                "sync_policy": "legacy_append",
                "legacy_base_group_count": 1,
                "legacy_reference_group_index": 0,
            }
        )
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            retired_layout,
            "Invalid named Neuron optimizer layout",
        )

        duplicate_name = copy.deepcopy(layout)
        duplicate_name["optimizers"][0]["parameter_names"][0][1] = "a"
        self.assert_prepare_rejected(
            module,
            optimizer,
            saved_state,
            duplicate_name,
            "duplicate parameter name",
        )

    def test_prepare_rejects_live_membership_and_group_drift(self) -> None:
        module, _, saved_state, layout = self.layout_fixture()
        missing_optimizer = torch.optim.SGD([module["a"], module["b"]], lr=0.1)
        self.assert_prepare_rejected(
            module,
            missing_optimizer,
            saved_state,
            layout,
            "membership differs",
        )

        split_optimizer = torch.optim.SGD(
            [
                {"params": [module["a"]], "lr": 0.1},
                {"params": [module["b"], module["c"]], "lr": 0.1},
            ]
        )
        self.assert_prepare_rejected(
            module,
            split_optimizer,
            saved_state,
            layout,
            "parameter-group counts differ",
        )


class TestNeuronOptimizerSchedulerEdges(unittest.TestCase):
    @staticmethod
    def nested_scheduler_fixture():
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        first = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        second = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([first, second])
        return optimizer, scheduler, scheduler.state_dict()

    def test_nested_scheduler_checkpoint_snapshot_rolls_back_recursively(self) -> None:
        optimizer, scheduler, saved_state = self.nested_scheduler_fixture()
        original_live_epoch = scheduler._schedulers[0].last_epoch
        original_saved_epoch = saved_state["_schedulers"][0]["last_epoch"]
        reconciler = NeuronSchedulerCheckpointReconciler()
        reconciler.prepare_for_load(
            [
                SchedulerGroupLoadBinding(
                    scheduler=scheduler,
                    saved_state=saved_state,
                    optimizer=optimizer,
                )
            ]
        )

        scheduler._schedulers[0].last_epoch = 101
        saved_state["_schedulers"][0]["last_epoch"] = 102
        reconciler.clear()

        self.assertEqual(scheduler._schedulers[0].last_epoch, original_live_epoch)
        self.assertEqual(
            saved_state["_schedulers"][0]["last_epoch"],
            original_saved_epoch,
        )

    def test_nested_scheduler_rejects_saved_child_count_drift(self) -> None:
        optimizer, scheduler, saved_state = self.nested_scheduler_fixture()
        saved_state["_schedulers"].pop()

        with self.assertRaisesRegex(RuntimeError, "child counts differ"):
            NeuronSchedulerCheckpointReconciler().prepare_for_load(
                [
                    SchedulerGroupLoadBinding(
                        scheduler=scheduler,
                        saved_state=saved_state,
                        optimizer=optimizer,
                    )
                ]
            )

    def test_scheduler_partial_commit_is_partitioned_by_optimizer_identity(
        self,
    ) -> None:
        first_optimizer, first_scheduler, first_state = self.nested_scheduler_fixture()
        second_optimizer, second_scheduler, second_state = (
            self.nested_scheduler_fixture()
        )
        first_child = first_scheduler._schedulers[0]
        second_child = second_scheduler._schedulers[0]
        first_original_epoch = first_child.last_epoch
        second_original_epoch = second_child.last_epoch
        reconciler = NeuronSchedulerCheckpointReconciler()
        reconciler.prepare_for_load(
            [
                SchedulerGroupLoadBinding(
                    scheduler=first_scheduler,
                    saved_state=first_state,
                    optimizer=first_optimizer,
                ),
                SchedulerGroupLoadBinding(
                    scheduler=second_scheduler,
                    saved_state=second_state,
                    optimizer=second_optimizer,
                ),
            ]
        )

        first_child.last_epoch = 101
        second_child.last_epoch = 201
        reconciler.mark_optimizer_loaded(first_optimizer)
        reconciler.commit_loaded()
        reconciler.clear()

        self.assertEqual(first_child.last_epoch, 101)
        self.assertEqual(second_child.last_epoch, second_original_epoch)
        self.assertNotEqual(first_original_epoch, first_child.last_epoch)

    def test_nonadjacent_group_removal_keeps_cyclic_scheduler_aligned(self) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(4)]
        optimizer = torch.optim.SGD(
            [
                {"params": [parameter], "lr": learning_rate, "momentum": 0.9}
                for parameter, learning_rate in zip(
                    parameters,
                    (0.10, 0.20, 0.30, 0.40),
                    strict=True,
                )
            ]
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=[0.01, 0.02, 0.03, 0.04],
            max_lr=[0.11, 0.12, 0.13, 0.14],
            step_size_up=2,
        )
        original_groups = tuple(optimizer.param_groups)
        removed_group_indices = (1, 3)

        preflight_scheduler_group_removal(
            scheduler,
            removed_group_indices,
            previous_group_count=4,
        )
        optimizer.param_groups[:] = [original_groups[0], original_groups[2]]
        remove_scheduler_groups(
            scheduler,
            removed_group_indices,
            previous_group_count=4,
        )

        self.assertEqual(scheduler.base_lrs, [0.01, 0.03])
        self.assertEqual(scheduler.max_lrs, [0.11, 0.13])
        self.assertEqual(len(optimizer.param_groups), 2)

    def test_group_removal_rejects_duplicate_indices(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        with self.assertRaisesRegex(RuntimeError, "group-removal indices"):
            preflight_scheduler_group_removal(
                scheduler,
                (0, 0),
                previous_group_count=1,
            )


class TestNeuronOptimizerSyncCheckpointEdges(unittest.TestCase):
    @staticmethod
    def trainer(optimizers):
        return SimpleNamespace(
            optimizers=optimizers,
            state=SimpleNamespace(fn=TrainerFn.FITTING),
            lr_scheduler_configs=[],
        )

    @staticmethod
    def checkpoint(module, optimizer):
        optimizer_state = optimizer.state_dict()
        return {
            "optimizer_states": [optimizer_state],
            OPTIMIZER_LAYOUT_CHECKPOINT_KEY: NeuronOptimizerNamedLayout.capture(
                module,
                [optimizer],
                [optimizer_state],
            ),
        }

    def test_callback_ignores_non_optimizer_checkpoint_payloads(self) -> None:
        module = nn.Linear(1, 1)
        trainer = self.trainer([])
        callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = {"optimizer_states": {}}

        callback.on_load_checkpoint(trainer, module, checkpoint)
        callback.on_save_checkpoint(trainer, module, checkpoint)

        self.assertNotIn(OPTIMIZER_LAYOUT_CHECKPOINT_KEY, checkpoint)

    def test_callback_saves_canonical_named_layout(self) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.03)
        checkpoint = {"optimizer_states": [optimizer.state_dict()]}

        NeuronClusterOptimizerSyncCallback().on_save_checkpoint(
            self.trainer([optimizer]),
            module,
            checkpoint,
        )

        self.assertEqual(
            checkpoint[OPTIMIZER_LAYOUT_CHECKPOINT_KEY]["optimizers"][0]["sync_policy"],
            "role",
        )

    def test_callback_rejects_unregistered_optimizer_parameters_on_save(self) -> None:
        module = nn.Linear(1, 1)
        external_parameter = nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam(
            [*module.parameters(), external_parameter],
            lr=0.03,
        )
        checkpoint = {"optimizer_states": [optimizer.state_dict()]}

        with self.assertRaisesRegex(RuntimeError, "must be registered"):
            NeuronClusterOptimizerSyncCallback().on_save_checkpoint(
                self.trainer([optimizer]),
                module,
                checkpoint,
            )

    def test_callback_rejects_checkpoint_without_named_layout(self) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        with self.assertRaisesRegex(RuntimeError, "retired optimizer layout"):
            NeuronClusterOptimizerSyncCallback().on_load_checkpoint(
                self.trainer([optimizer]),
                module,
                {"optimizer_states": [optimizer.state_dict()]},
            )

    def test_callback_rolls_back_invalid_named_layout(self) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        trainer = self.trainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        original_group = optimizer.param_groups[0]
        checkpoint = {
            "optimizer_states": [optimizer.state_dict()],
            OPTIMIZER_LAYOUT_CHECKPOINT_KEY: {
                "version": 999,
                "optimizers": [],
            },
        }

        with self.assertRaisesRegex(RuntimeError, "Unsupported named"):
            callback.on_load_checkpoint(trainer, module, checkpoint)

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertIs(optimizer.param_groups[0], original_group)

    def test_callback_rejects_scheduler_count_drift(self) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        trainer = self.trainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = self.checkpoint(module, optimizer)
        checkpoint["lr_schedulers"] = []

        with self.assertRaisesRegex(RuntimeError, "scheduler counts differ"):
            callback.on_load_checkpoint(trainer, module, checkpoint)

    def test_callback_ignores_foreign_scheduler_during_current_load(self) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        foreign_parameter = nn.Parameter(torch.tensor(2.0))
        foreign_optimizer = torch.optim.SGD([foreign_parameter], lr=0.2)
        foreign_scheduler = torch.optim.lr_scheduler.StepLR(
            foreign_optimizer,
            step_size=1,
        )
        trainer = self.trainer([optimizer])
        trainer.lr_scheduler_configs = [SimpleNamespace(scheduler=foreign_scheduler)]
        callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = self.checkpoint(module, optimizer)
        checkpoint["lr_schedulers"] = [foreign_scheduler.state_dict()]

        callback.on_load_checkpoint(trainer, module, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        callback.on_train_start(trainer, module)

        self.assertEqual(len(optimizer.param_groups), 1)


if __name__ == "__main__":
    unittest.main()
