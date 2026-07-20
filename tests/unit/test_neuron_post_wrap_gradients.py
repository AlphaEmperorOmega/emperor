import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from emperor.neuron import NeuronClusterOptimizerSyncCallback
from emperor.neuron._distributed_gradients import average_post_wrap_gradients


class TestPostWrapGradientAveraging(unittest.TestCase):
    def test_noops_without_an_active_multi_rank_process_group(self) -> None:
        parameter = nn.Parameter(torch.tensor([2.0]))
        parameter.grad = torch.tensor([3.0])
        module = nn.ParameterList([parameter])
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        cases = (
            (False, False, 2, 0),
            (True, False, 2, 0),
            (True, True, 1, 1),
        )
        for available, initialized, world_size, world_size_calls in cases:
            with self.subTest(
                available=available,
                initialized=initialized,
                world_size=world_size,
            ):
                with (
                    patch("torch.distributed.is_available", return_value=available),
                    patch(
                        "torch.distributed.is_initialized",
                        return_value=initialized,
                    ),
                    patch(
                        "torch.distributed.get_world_size",
                        return_value=world_size,
                    ) as get_world_size,
                    patch("torch.distributed.all_reduce") as all_reduce,
                ):
                    average_post_wrap_gradients(
                        module,
                        optimizer,
                        {id(parameter)},
                    )

                self.assertEqual(get_world_size.call_count, world_size_calls)
                all_reduce.assert_not_called()
                torch.testing.assert_close(parameter.grad, torch.tensor([3.0]))

    def test_averages_only_tracked_optimizer_parameters(self) -> None:
        module = nn.Sequential(nn.Linear(2, 1), nn.Linear(1, 1))
        tracked_parameter = module[1].weight
        untracked_parameter = module[0].weight
        foreign_parameter = module[0].bias
        optimizer = torch.optim.SGD(
            [tracked_parameter, untracked_parameter],
            lr=0.1,
        )
        tracked_parameter.grad = torch.ones_like(tracked_parameter)
        untracked_parameter.grad = torch.full_like(untracked_parameter, 2.0)
        foreign_parameter.grad = torch.full_like(foreign_parameter, 3.0)

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce") as all_reduce,
        ):
            average_post_wrap_gradients(
                module,
                optimizer,
                {id(tracked_parameter), id(foreign_parameter)},
            )

        self.assertEqual(all_reduce.call_count, 3)
        torch.testing.assert_close(
            tracked_parameter.grad,
            torch.full_like(tracked_parameter, 0.5),
        )
        torch.testing.assert_close(
            untracked_parameter.grad,
            torch.full_like(untracked_parameter, 2.0),
        )
        torch.testing.assert_close(
            foreign_parameter.grad,
            torch.full_like(foreign_parameter, 3.0),
        )

    def test_callback_forwards_only_parameters_added_after_fit_start(self) -> None:
        initial_parameter = nn.Parameter(torch.tensor([1.0]))
        grown_parameter = nn.Parameter(torch.tensor([2.0]))
        cluster = nn.Module()
        cluster.cluster = nn.ParameterDict({"initial": initial_parameter})
        host = nn.Module()
        host.cluster = cluster
        optimizer = torch.optim.SGD(cluster.parameters(), lr=0.1)
        trainer = SimpleNamespace(optimizers=[optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback._NeuronClusterOptimizerSyncCallback__find_neuron_clusters = (
            lambda module: [cluster]
        )

        callback.on_fit_start(trainer, host)
        cluster.cluster["grown"] = grown_parameter
        callback.sync_optimizers(trainer, host)

        with patch(
            "emperor.neuron._optimizer_sync.average_post_wrap_gradients"
        ) as average:
            callback.on_before_optimizer_step(trainer, host, optimizer)

        average.assert_called_once_with(
            host,
            optimizer,
            {id(grown_parameter)},
        )
