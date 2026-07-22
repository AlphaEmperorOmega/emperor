import unittest

import torch
from torch import nn

from emperor.neuron._optimizer_scheduler import (
    NeuronSchedulerCheckpointReconciler,
    NeuronSchedulerMutationTransaction,
    SchedulerGroupLoadBinding,
    preflight_scheduler_group_removal,
    remove_scheduler_groups,
)


class TestNeuronSchedulerTopology(unittest.TestCase):
    def test_group_removal_keeps_step_lr_aligned(self) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(2)]
        optimizer = torch.optim.SGD(
            [
                {"params": [parameters[0]], "lr": 0.1},
                {"params": [parameters[1]], "lr": 0.2},
            ]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

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
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        first = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        second = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([first, second])
        saved_state = scheduler.state_dict()
        original_first_epoch = first.last_epoch
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

        first.last_epoch = 41
        saved_state["_schedulers"][0]["last_epoch"] = 42
        reconciler.clear()
        self.assertEqual(first.last_epoch, original_first_epoch)
        self.assertEqual(
            saved_state["_schedulers"][0]["last_epoch"],
            original_saved_epoch,
        )

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
