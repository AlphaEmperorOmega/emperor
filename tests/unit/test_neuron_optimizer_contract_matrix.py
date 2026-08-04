from __future__ import annotations

import copy
import unittest
import warnings
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
    NeuronSchedulerMutationTransaction,
    SchedulerGroupLoadBinding,
    preflight_scheduler_group_removal,
    remove_scheduler_groups,
)


class TestOptimizerCheckpointValidation(unittest.TestCase):
    @staticmethod
    def fixture(*, split_groups: bool = False):
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
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        saved_states: list[dict],
        layout,
        message: str,
    ) -> None:
        saved_snapshot = copy.deepcopy(saved_states)
        original_groups = tuple(optimizer.param_groups)

        with self.assertRaisesRegex(RuntimeError, message):
            NeuronOptimizerNamedLayout().prepare_for_load(
                module,
                [optimizer],
                saved_states,
                layout,
            )

        self.assertEqual(saved_states, saved_snapshot)
        self.assertEqual(tuple(optimizer.param_groups), original_groups)

    def test_capture_rejects_group_name_and_duplicate_mismatches(self) -> None:
        module, optimizer, saved_state, _layout = self.fixture()

        wrong_group_count = copy.deepcopy(saved_state)
        wrong_group_count["param_groups"] = []
        with self.assertRaisesRegex(RuntimeError, "parameter-group counts differ"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [optimizer],
                [wrong_group_count],
            )

        optimizer.param_groups[0]["param_names"] = ["a"]
        with self.assertRaisesRegex(RuntimeError, "live param_names"):
            NeuronOptimizerNamedLayout.capture(module, [optimizer], [saved_state])
        del optimizer.param_groups[0]["param_names"]

        bad_serialized_names = copy.deepcopy(saved_state)
        bad_serialized_names["param_groups"][0]["param_names"] = "a"
        with self.assertRaisesRegex(RuntimeError, "serialized param_names"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [optimizer],
                [bad_serialized_names],
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            duplicate_optimizer = torch.optim.SGD(
                [module["a"], module["a"]],
                lr=0.1,
            )
        with self.assertRaisesRegex(RuntimeError, "appears more than once"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [duplicate_optimizer],
                [duplicate_optimizer.state_dict()],
            )

    def test_prepare_rejects_outer_entry_and_policy_metadata(self) -> None:
        cases = []

        module, optimizer, saved_state, layout = self.fixture()
        extra_outer_key = copy.deepcopy(layout)
        extra_outer_key["extra"] = True
        cases.append(
            (module, optimizer, [saved_state], extra_outer_key, "Invalid named")
        )

        module, optimizer, saved_state, layout = self.fixture()
        non_list_optimizers = copy.deepcopy(layout)
        non_list_optimizers["optimizers"] = {}
        cases.append(
            (module, optimizer, [saved_state], non_list_optimizers, "Invalid named")
        )

        module, optimizer, saved_state, layout = self.fixture()
        invalid_optimizer_entry = copy.deepcopy(layout)
        invalid_optimizer_entry["optimizers"] = [None]
        cases.append(
            (module, optimizer, [saved_state], invalid_optimizer_entry, "Invalid named")
        )

        module, optimizer, saved_state, layout = self.fixture()
        wrong_policy = copy.deepcopy(layout)
        wrong_policy["optimizers"][0]["sync_policy"] = "append"
        cases.append((module, optimizer, [saved_state], wrong_policy, "sync policy"))

        module, optimizer, saved_state, layout = self.fixture()
        wrong_group_count = copy.deepcopy(layout)
        wrong_group_count["optimizers"][0]["parameter_names"] = []
        cases.append(
            (module, optimizer, [saved_state], wrong_group_count, "group metadata")
        )

        for module, optimizer, states, candidate_layout, message in cases:
            with self.subTest(message=message):
                self.assert_prepare_rejected(
                    module,
                    optimizer,
                    states,
                    candidate_layout,
                    message,
                )

        module, optimizer, _saved_state, layout = self.fixture()
        self.assert_prepare_rejected(
            module,
            optimizer,
            [],
            layout,
            "optimizer counts differ",
        )

    def test_prepare_rejects_saved_name_and_live_group_drift(self) -> None:
        cases = []

        module, optimizer, saved_state, layout = self.fixture()
        non_string_name = copy.deepcopy(layout)
        non_string_name["optimizers"][0]["parameter_names"][0][0] = 1
        cases.append(
            (module, optimizer, [saved_state], non_string_name, "group metadata")
        )

        module, optimizer, saved_state, layout = self.fixture()
        missing_name = copy.deepcopy(layout)
        missing_name["optimizers"][0]["parameter_names"][0][0] = "missing"
        cases.append((module, optimizer, [saved_state], missing_name, "absent from"))

        module, optimizer, saved_state, layout = self.fixture()
        invalid_serialized_ids = copy.deepcopy(saved_state)
        invalid_serialized_ids["param_groups"][0]["params"] = "invalid"
        cases.append(
            (
                module,
                optimizer,
                [invalid_serialized_ids],
                layout,
                "group metadata",
            )
        )

        module, optimizer, saved_state, layout = self.fixture()
        invalid_serialized_names = copy.deepcopy(saved_state)
        invalid_serialized_names["param_groups"][0]["param_names"] = "invalid"
        cases.append(
            (
                module,
                optimizer,
                [invalid_serialized_names],
                layout,
                "param_names metadata",
            )
        )

        for module, optimizer, states, candidate_layout, message in cases:
            with self.subTest(message=message):
                self.assert_prepare_rejected(
                    module,
                    optimizer,
                    states,
                    candidate_layout,
                    message,
                )

        module, _optimizer, saved_state, layout = self.fixture()
        external_parameter = nn.Parameter(torch.tensor(4.0))
        unregistered_optimizer = torch.optim.SGD(
            [*module.parameters(), external_parameter],
            lr=0.1,
        )
        self.assert_prepare_rejected(
            module,
            unregistered_optimizer,
            [saved_state],
            layout,
            "every live optimizer parameter must be registered",
        )

        module, _optimizer, saved_state, layout = self.fixture(split_groups=True)
        regrouped_optimizer = torch.optim.SGD(
            [
                {"params": [module["a"], module["b"]], "lr": 0.1},
                {"params": [module["c"]], "lr": 0.2},
            ]
        )
        self.assert_prepare_rejected(
            module,
            regrouped_optimizer,
            [saved_state],
            layout,
            "parameter-group membership differs",
        )

        unrelated_optimizer = torch.optim.SGD(module.parameters(), lr=0.3)
        self.assertFalse(
            NeuronOptimizerNamedLayout().optimizer_requires_completion(
                unrelated_optimizer
            )
        )


class _StatefulLambda:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, step: int) -> float:
        self.calls.append(step)
        return 1.0


class TestSchedulerCheckpointValidation(unittest.TestCase):
    @staticmethod
    def nested_fixture():
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in (1, 2)]
        optimizer = torch.optim.SGD(
            [
                {"params": [parameters[0]], "lr": 0.1},
                {"params": [parameters[1]], "lr": 0.2},
            ]
        )
        first = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        second = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([first, second])
        return optimizer, scheduler

    def test_nested_removal_already_remaining_and_ambiguous_lengths(self) -> None:
        optimizer, scheduler = self.nested_fixture()
        original_groups = tuple(optimizer.param_groups)

        preflight_scheduler_group_removal(
            scheduler,
            (1,),
            previous_group_count=2,
        )
        optimizer.param_groups[:] = [original_groups[0]]
        remove_scheduler_groups(
            scheduler,
            (1,),
            previous_group_count=2,
        )

        for child in scheduler._schedulers:
            self.assertEqual(len(child.base_lrs), 1)
            self.assertEqual(len(child._last_lr), 1)

        first_child = scheduler._schedulers[0]
        first_child.base_lrs[:] = [0.1]
        remove_scheduler_groups(
            first_child,
            (),
            previous_group_count=1,
        )
        self.assertEqual(first_child.base_lrs, [0.1])

        optimizer, scheduler = self.nested_fixture()
        scheduler._schedulers[0].base_lrs.append(0.3)
        with self.assertRaisesRegex(RuntimeError, "base_lrs has 3 entries"):
            preflight_scheduler_group_removal(
                scheduler,
                (1,),
                previous_group_count=2,
            )

        for invalid_indices in ((-1,), (2,)):
            with (
                self.subTest(indices=invalid_indices),
                self.assertRaisesRegex(RuntimeError, "group-removal indices"),
            ):
                preflight_scheduler_group_removal(
                    scheduler._schedulers[1],
                    invalid_indices,
                    previous_group_count=2,
                )

    def test_snapshot_deduplicates_shared_callables_and_saved_namespaces(self) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in (1, 2)]
        optimizer = torch.optim.SGD(
            [
                {"params": [parameters[0]], "lr": 0.1},
                {"params": [parameters[1]], "lr": 0.2},
            ]
        )
        shared_lambda = _StatefulLambda()
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[shared_lambda, shared_lambda],
        )
        scheduler.builtin_probe = len
        original_calls = tuple(shared_lambda.calls)
        original_base_lrs = tuple(scheduler.base_lrs)
        transaction = NeuronSchedulerMutationTransaction()

        transaction.prepare([scheduler, scheduler])
        shared_lambda.calls.append(99)
        scheduler.base_lrs.append(0.3)
        transaction.clear()

        self.assertEqual(tuple(shared_lambda.calls), original_calls)
        self.assertEqual(tuple(scheduler.base_lrs), original_base_lrs)

        saved_state = scheduler.state_dict()
        saved_snapshot = copy.deepcopy(saved_state)
        reconciler = NeuronSchedulerCheckpointReconciler()
        binding = SchedulerGroupLoadBinding(scheduler, saved_state, optimizer)
        reconciler.prepare_for_load([binding, binding])

        unrelated_optimizer = torch.optim.SGD(
            [nn.Parameter(torch.tensor(3.0))],
            lr=0.3,
        )
        self.assertFalse(reconciler.optimizer_requires_completion(unrelated_optimizer))
        saved_state["last_epoch"] = 99
        reconciler.clear()
        self.assertEqual(saved_state, saved_snapshot)

    def test_nested_snapshot_accepts_absent_saved_state_and_rejects_bad_children(
        self,
    ) -> None:
        optimizer, scheduler = self.nested_fixture()
        reconciler = NeuronSchedulerCheckpointReconciler()

        reconciler.prepare_for_load(
            [SchedulerGroupLoadBinding(scheduler, None, optimizer)]
        )
        self.assertTrue(reconciler.optimizer_requires_completion(optimizer))
        reconciler.clear()

        bad_state = scheduler.state_dict()
        bad_state["_schedulers"] = "invalid"
        with self.assertRaisesRegex(RuntimeError, "child counts differ"):
            reconciler.prepare_for_load(
                [SchedulerGroupLoadBinding(scheduler, bad_state, optimizer)]
            )


class TestOptimizerCheckpointHookLifecycle(unittest.TestCase):
    @staticmethod
    def trainer(optimizer, *, function: TrainerFn = TrainerFn.FITTING):
        return SimpleNamespace(
            optimizers=[optimizer],
            state=SimpleNamespace(fn=function),
            lr_scheduler_configs=[],
        )

    @staticmethod
    def checkpoint(module: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
        saved_state = optimizer.state_dict()
        return {
            "optimizer_states": [saved_state],
            OPTIMIZER_LAYOUT_CHECKPOINT_KEY: NeuronOptimizerNamedLayout.capture(
                module,
                [optimizer],
                [saved_state],
            ),
        }

    def test_non_fitting_deferred_empty_and_delayed_missing_layout_paths(self) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        callback = NeuronClusterOptimizerSyncCallback()
        checkpoint = self.checkpoint(module, optimizer)

        callback.on_load_checkpoint(
            self.trainer(optimizer, function=TrainerFn.VALIDATING),
            module,
            checkpoint,
        )
        self.assertIsNone(callback._pending_saved_optimizer_states)

        fitting_trainer = self.trainer(optimizer)
        callback.on_load_checkpoint(
            fitting_trainer,
            module,
            {"optimizer_states": []},
        )
        self.assertEqual(len(callback._optimizer_load_hook_handles), 1)
        callback.on_fit_end(fitting_trainer, module)
        self.assertEqual(callback._optimizer_load_hook_handles, {})

        delayed = NeuronClusterOptimizerSyncCallback()
        delayed._pending_saved_optimizer_states = [optimizer.state_dict()]
        delayed._pending_named_optimizer_layout = None
        with self.assertRaisesRegex(RuntimeError, "retired optimizer layout"):
            delayed.on_fit_start(fitting_trainer, module)

    def test_duplicate_load_hook_is_skipped_and_fit_cleanup_removes_it(self) -> None:
        module = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
        trainer = self.trainer(optimizer)
        checkpoint = self.checkpoint(module, optimizer)
        callback = NeuronClusterOptimizerSyncCallback()

        callback.on_load_checkpoint(trainer, module, checkpoint)
        first_handle = callback._optimizer_load_hook_handles[id(optimizer)]
        callback.on_load_checkpoint(trainer, module, checkpoint)

        self.assertEqual(len(callback._optimizer_load_hook_handles), 1)
        self.assertIs(
            callback._optimizer_load_hook_handles[id(optimizer)],
            first_handle,
        )

        callback.on_fit_end(trainer, module)

        self.assertEqual(callback._optimizer_load_hook_handles, {})
        self.assertIsNone(callback._pending_saved_optimizer_states)


if __name__ == "__main__":
    unittest.main()
