import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from emperor.neuron import NeuronClusterConfig, NeuronClusterOptimizerSyncCallback
from emperor.neuron._distributed_gradients import average_post_wrap_gradients
from unit.test_neuron import NeuronTestCase


def _initialize_process_group(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _real_post_wrap_sync_worker(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _initialize_process_group(rank, world_size, init_file)
    try:
        for dtype_name in ("float32", "float64"):
            dtype = getattr(torch, dtype_name)
            torch.manual_seed(0)
            cluster = config.build().to(dtype=dtype)
            host = nn.Module()
            host.cluster = cluster
            named_parameters = list(host.named_parameters())
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": [parameter for _, parameter in named_parameters],
                        "param_names": [name for name, _ in named_parameters],
                    }
                ],
                lr=0.1,
            )
            trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])
            callback = NeuronClusterOptimizerSyncCallback()
            callback.on_fit_start(trainer, host)

            parent = cluster.cluster["neuron_1_1_1"]
            grown_neuron = cluster._initialize_neuron(
                2,
                1,
                1,
                runtime_template=parent,
            )
            cluster._add_neuron(
                cluster.cluster,
                "neuron_2_1_1",
                grown_neuron,
            )
            callback.sync_optimizers(trainer, host)

            old_parameter = parent.nucleus.model.weight
            grown_parameter = grown_neuron.nucleus.model.weight
            with torch.no_grad():
                old_parameter.fill_(5.0)
                grown_parameter.fill_(7.0)
            optimizer.zero_grad(set_to_none=True)
            local_old_gradient = float(2 * rank + 1)
            local_grown_gradient = float(4 * rank + 2)
            old_parameter.grad = torch.full_like(
                old_parameter,
                local_old_gradient,
            )
            grown_parameter.grad = torch.full_like(
                grown_parameter,
                local_grown_gradient,
            )

            callback.on_before_optimizer_step(trainer, host, optimizer)

            torch.testing.assert_close(
                old_parameter.grad,
                torch.full_like(old_parameter, local_old_gradient),
            )
            torch.testing.assert_close(
                grown_parameter.grad,
                torch.full_like(grown_parameter, 4.0),
            )
            optimizer.step()
            torch.testing.assert_close(
                old_parameter,
                torch.full_like(old_parameter, 5.0 - 0.1 * local_old_gradient),
            )
            torch.testing.assert_close(
                grown_parameter,
                torch.full_like(grown_parameter, 6.6),
            )

            gathered_old_parameters = [
                torch.zeros_like(old_parameter) for _ in range(world_size)
            ]
            gathered_grown_parameters = [
                torch.zeros_like(grown_parameter) for _ in range(world_size)
            ]
            torch.distributed.all_gather(gathered_old_parameters, old_parameter)
            torch.distributed.all_gather(gathered_grown_parameters, grown_parameter)
            assert not torch.equal(
                gathered_old_parameters[0],
                gathered_old_parameters[1],
            )
            torch.testing.assert_close(
                gathered_grown_parameters[0],
                gathered_grown_parameters[1],
            )
            assert (
                "cluster.cluster.neuron_2_1_1.nucleus.model.weight"
                in (optimizer.param_groups[0]["param_names"])
            )
            callback.on_fit_end(trainer, host)
    finally:
        torch.distributed.destroy_process_group()


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


@unittest.skipUnless(
    torch.distributed.is_available() and torch.distributed.is_gloo_available(),
    "gloo process group support is required",
)
class TestRealPostWrapSync(NeuronTestCase):
    def test_callback_averages_only_post_wrap_parameters(self) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            init_file = os.path.join(temporary_directory, "process_group_init")
            torch.multiprocessing.spawn(
                _real_post_wrap_sync_worker,
                args=(2, init_file, config),
                nprocs=2,
                join=True,
            )
