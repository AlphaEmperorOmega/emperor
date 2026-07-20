import copy
import unittest

import torch
from torch import nn

from emperor.neuron._optimizer_checkpoint import LegacyOptimizerAppendPolicy
from emperor.neuron._optimizer_scheduler import (
    NeuronSchedulerCheckpointReconciler,
    NeuronSchedulerMutationTransaction,
    SchedulerGroupLoadBinding,
    extend_scheduler_for_new_group,
    preflight_scheduler_group_extension,
    preflight_scheduler_group_removal,
    remove_scheduler_groups,
)


class TestNeuronSchedulerTopology(unittest.TestCase):
    def test_group_extension_and_removal_keep_step_lr_aligned(self) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(2)]
        optimizer = torch.optim.SGD([parameters[0]], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        preflight_scheduler_group_extension(
            scheduler,
            previous_group_count=1,
            reference_group_index=0,
        )
        reference_options = {
            name: value
            for name, value in optimizer.param_groups[0].items()
            if name != "params"
        }
        optimizer.add_param_group(
            {**reference_options, "params": [parameters[1]]},
        )
        extend_scheduler_for_new_group(
            scheduler,
            previous_group_count=1,
            reference_group_index=0,
        )

        self.assertEqual(scheduler.base_lrs, [0.1, 0.1])
        self.assertEqual(scheduler.get_last_lr(), [0.1, 0.1])

        preflight_scheduler_group_removal(
            scheduler,
            (1,),
            previous_group_count=2,
        )
        optimizer.param_groups.pop()
        remove_scheduler_groups(
            scheduler,
            (1,),
            previous_group_count=2,
        )

        self.assertEqual(scheduler.base_lrs, [0.1])
        self.assertEqual(scheduler.get_last_lr(), [0.1])

    def test_nested_scheduler_checkpoint_edits_roll_back_recursively(self) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(2)]
        optimizer = torch.optim.SGD([parameters[0]], lr=0.1)
        first = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        second = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([first, second])
        saved_state = copy.deepcopy(scheduler.state_dict())
        optimizer.add_param_group({"params": [parameters[1]], "lr": 0.1})
        reconciler = NeuronSchedulerCheckpointReconciler()

        reconciler.prepare_for_load(
            [
                SchedulerGroupLoadBinding(
                    scheduler=scheduler,
                    saved_state=saved_state,
                    optimizer=optimizer,
                    policy=LegacyOptimizerAppendPolicy(1, 0),
                    target_group_count=2,
                )
            ]
        )

        self.assertEqual(first.base_lrs, [0.1, 0.1])
        self.assertEqual(second.base_lrs, [0.1, 0.1])
        self.assertEqual(saved_state["_schedulers"][0]["base_lrs"], [0.1, 0.1])
        reconciler.clear()
        self.assertEqual(first.base_lrs, [0.1])
        self.assertEqual(second.base_lrs, [0.1])
        self.assertEqual(saved_state["_schedulers"][0]["base_lrs"], [0.1])

    def test_mutation_transaction_restores_scheduler_namespace_in_place(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        original_base_lrs = scheduler.base_lrs
        original_last_epoch = scheduler.last_epoch
        transaction = NeuronSchedulerMutationTransaction()

        transaction.prepare([scheduler])
        scheduler.base_lrs.append(0.2)
        scheduler.last_epoch = 99
        transaction.clear()

        self.assertIs(scheduler.base_lrs, original_base_lrs)
        self.assertEqual(scheduler.base_lrs, [0.1])
        self.assertEqual(scheduler.last_epoch, original_last_epoch)
