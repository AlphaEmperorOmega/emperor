from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from emperor.neuron import (
    NeuronClusterConfig,
    NeuronClusterOptimizerSyncCallback,
)
from emperor.neuron._distributed_gradients import (
    _average_gradient,
    average_post_wrap_gradients,
)
from unit.test_neuron import NeuronTestCase


class _DistributedGrowingModule(nn.Module):
    def __init__(self, config: NeuronClusterConfig) -> None:
        super().__init__()
        self.cluster = config.build()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output, auxiliary_loss = self.cluster(input_tensor)
        return output.square().mean() + auxiliary_loss


class _DistributedAtrophyHistoryProbe(nn.Module):
    def __init__(self, config: NeuronClusterConfig) -> None:
        super().__init__()
        self.cluster = config.build()
        self.grown_name = "neuron_5_1_1"
        self.cluster.cluster[self.grown_name] = self.cluster._initialize_neuron(
            5,
            1,
            1,
        )
        self.forward_index = 0

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        used_on_this_rank = (
            self.forward_index == 0 and torch.distributed.get_rank() == 1
        )
        self.cluster._neurons_called_this_forward = (
            {self.grown_name} if used_on_this_rank else set()
        )
        self.cluster._check_neuron_atrophy()
        self.forward_index += 1
        parameter_anchor = next(self.cluster.parameters()).reshape(-1)[0]
        return input_tensor + parameter_anchor * 0.0


class _DistributedGrowthHistoryProbe(nn.Module):
    def __init__(self, config: NeuronClusterConfig) -> None:
        super().__init__()
        self.cluster = config.build()
        self.parent_name = next(iter(self.cluster.cluster))
        self.grown_name = "neuron_2_1_1"
        self.forward_index = 0

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        baseline = self.cluster._capture_growth_counter_baseline()
        contributes_this_forward = (
            self.forward_index > 0 or torch.distributed.get_rank() == 1
        )
        if contributes_this_forward:
            self.cluster.cluster[self.parent_name].batch_counter += 1
        self.cluster._neurons_called_this_forward = {self.parent_name}
        self.cluster._check_neuron_growth(baseline)
        self.forward_index += 1
        parameter_anchor = next(self.cluster.parameters()).reshape(-1)[0]
        return input_tensor + parameter_anchor * 0.0


class TestConditionalDDPConfiguration(NeuronTestCase):
    def build_cluster_module(self) -> _DistributedGrowingModule:
        return _DistributedGrowingModule(
            NeuronClusterConfig(
                x_axis_total_neurons=5,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                initial_x_axis_total_neurons=5,
                initial_y_axis_total_neurons=1,
                initial_z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                neuron_config=self.neuron_config(),
            )
        )

    def test_fit_setup_enables_unused_parameter_detection_before_ddp_wrap(
        self,
    ) -> None:
        strategy = DDPStrategy()

        NeuronClusterOptimizerSyncCallback().setup(
            SimpleNamespace(strategy=strategy),
            self.build_cluster_module(),
            "fit",
        )

        self.assertIs(strategy._ddp_kwargs["find_unused_parameters"], True)

    def test_fit_setup_rejects_explicitly_disabled_unused_parameter_detection(
        self,
    ) -> None:
        strategy = DDPStrategy(find_unused_parameters=False)

        with self.assertRaisesRegex(
            RuntimeError,
            "find_unused_parameters=True",
        ):
            NeuronClusterOptimizerSyncCallback().setup(
                SimpleNamespace(strategy=strategy),
                self.build_cluster_module(),
                "fit",
            )

    def test_fit_setup_rejects_a_strategy_without_ddp_configuration(self) -> None:
        strategy = DDPStrategy()
        del strategy._ddp_kwargs

        with self.assertRaisesRegex(
            RuntimeError,
            "does not expose the DDP configuration",
        ):
            NeuronClusterOptimizerSyncCallback().setup(
                SimpleNamespace(strategy=strategy),
                self.build_cluster_module(),
                "fit",
            )

    def test_non_fit_setup_does_not_reconfigure_ddp(self) -> None:
        strategy = DDPStrategy(find_unused_parameters=False)

        NeuronClusterOptimizerSyncCallback().setup(
            SimpleNamespace(strategy=strategy),
            self.build_cluster_module(),
            "validate",
        )

        self.assertIs(strategy._ddp_kwargs["find_unused_parameters"], False)

    def test_fit_setup_configures_ddp_before_lazy_module_cluster_setup(self) -> None:
        strategy = DDPStrategy()
        module = nn.Module()

        NeuronClusterOptimizerSyncCallback().setup(
            SimpleNamespace(strategy=strategy),
            module,
            "fit",
        )
        module.cluster = self.build_cluster_module().cluster

        self.assertIs(strategy._ddp_kwargs["find_unused_parameters"], True)

    def test_fit_setup_rejects_static_graph_ddp(self) -> None:
        strategy = DDPStrategy(static_graph=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "static_graph=False",
        ):
            NeuronClusterOptimizerSyncCallback().setup(
                SimpleNamespace(strategy=strategy),
                self.build_cluster_module(),
                "fit",
            )

    def test_fit_setup_rejects_skipping_unused_parameter_reduction(self) -> None:
        strategy = DDPStrategy(skip_all_reduce_unused_params=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "skip_all_reduce_unused_params=False",
        ):
            NeuronClusterOptimizerSyncCallback().setup(
                SimpleNamespace(strategy=strategy),
                self.build_cluster_module(),
                "fit",
            )


class TestPostWrapGradientAveraging(unittest.TestCase):
    def test_noops_without_an_active_multi_rank_process_group(self) -> None:
        parameter = nn.Parameter(torch.tensor([2.0]))
        parameter.grad = torch.tensor([3.0])
        module = nn.ParameterList([parameter])
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        cases = (
            ({"available": False, "initialized": False, "world_size": 2}, 0),
            ({"available": True, "initialized": False, "world_size": 2}, 0),
            ({"available": True, "initialized": True, "world_size": 1}, 1),
        )
        for process_group, expected_world_size_calls in cases:
            with self.subTest(process_group=process_group):
                with (
                    patch(
                        "torch.distributed.is_available",
                        return_value=process_group["available"],
                    ),
                    patch(
                        "torch.distributed.is_initialized",
                        return_value=process_group["initialized"],
                    ),
                    patch(
                        "torch.distributed.get_world_size",
                        return_value=process_group["world_size"],
                    ) as get_world_size,
                ):
                    average_post_wrap_gradients(
                        module,
                        optimizer,
                        {id(parameter)},
                    )

                self.assertEqual(
                    get_world_size.call_count,
                    expected_world_size_calls,
                )
                torch.testing.assert_close(parameter.grad, torch.tensor([3.0]))

    def test_averages_only_tracked_parameters_owned_by_the_optimizer(self) -> None:
        module = nn.Sequential(nn.Linear(2, 1), nn.Linear(1, 1))
        untracked_parameter = module[0].weight
        other_optimizer_parameter = module[0].bias
        owned_parameter = module[1].weight
        optimizer = torch.optim.SGD(
            [owned_parameter, untracked_parameter],
            lr=0.1,
        )
        owned_parameter.grad = torch.ones_like(owned_parameter)
        untracked_parameter.grad = torch.full_like(untracked_parameter, 2.0)
        other_optimizer_parameter.grad = torch.full_like(
            other_optimizer_parameter,
            3.0,
        )

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce") as all_reduce,
        ):
            average_post_wrap_gradients(
                module,
                optimizer,
                {id(owned_parameter), id(other_optimizer_parameter)},
            )

        self.assertEqual(all_reduce.call_count, 3)
        torch.testing.assert_close(
            owned_parameter.grad,
            torch.full_like(owned_parameter, 0.5),
        )
        torch.testing.assert_close(
            untracked_parameter.grad,
            torch.full_like(untracked_parameter, 2.0),
        )
        torch.testing.assert_close(
            other_optimizer_parameter.grad,
            torch.full_like(other_optimizer_parameter, 3.0),
        )

    def test_local_sparse_gradient_is_rejected_after_shared_sparse_flag(self) -> None:
        parameter = nn.Parameter(torch.zeros(3))
        parameter.grad = torch.sparse_coo_tensor(
            torch.tensor([[1]]),
            torch.tensor([2.0]),
            size=(3,),
            is_coalesced=True,
        )

        with (
            patch("torch.distributed.all_reduce") as all_reduce,
            self.assertRaisesRegex(
                RuntimeError,
                "Distributed Neuron growth does not support sparse gradients",
            ),
        ):
            _average_gradient(parameter, world_size=2)

        all_reduce.assert_called_once()

    def test_remote_sparse_gradient_is_rejected_on_every_rank_before_other_collectives(
        self,
    ) -> None:
        parameter = nn.Parameter(torch.zeros(3))
        parameter.grad = torch.ones_like(parameter)

        def report_remote_sparse(tensor: torch.Tensor) -> None:
            tensor.fill_(1)

        with (
            patch(
                "torch.distributed.all_reduce",
                side_effect=report_remote_sparse,
            ) as all_reduce,
            self.assertRaisesRegex(
                RuntimeError,
                "Distributed Neuron growth does not support sparse gradients",
            ),
        ):
            _average_gradient(parameter, world_size=2)

        all_reduce.assert_called_once()

    def test_globally_inactive_parameter_keeps_no_gradient(self) -> None:
        parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))

        with patch("torch.distributed.all_reduce") as all_reduce:
            _average_gradient(parameter, world_size=2)

        self.assertEqual(all_reduce.call_count, 2)
        self.assertIsNone(parameter.grad)

    def test_remote_only_gradient_is_materialized_as_the_world_average(self) -> None:
        parameter = nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))

        collective_count = 0

        def remote_collective(tensor: torch.Tensor) -> None:
            nonlocal collective_count
            collective_count += 1
            if tensor.dtype == torch.int64:
                tensor.fill_(0 if collective_count == 1 else 1)
            else:
                tensor.copy_(torch.tensor([6.0, -4.0], dtype=tensor.dtype))

        with patch(
            "torch.distributed.all_reduce",
            side_effect=remote_collective,
        ) as all_reduce:
            _average_gradient(parameter, world_size=2)

        self.assertEqual(all_reduce.call_count, 3)
        torch.testing.assert_close(
            parameter.grad,
            torch.tensor([3.0, -2.0], dtype=torch.float64),
        )
        self.assertEqual(parameter.grad.dtype, parameter.dtype)
        self.assertEqual(parameter.grad.device, parameter.device)

    def test_local_gradient_storage_receives_the_world_average(self) -> None:
        parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        original_gradient = torch.tensor([2.0], dtype=torch.float64)
        parameter.grad = original_gradient

        collective_count = 0

        def collective(tensor: torch.Tensor) -> None:
            nonlocal collective_count
            collective_count += 1
            if tensor.dtype == torch.int64:
                tensor.fill_(0 if collective_count == 1 else 2)
            else:
                tensor.add_(4.0)

        with patch("torch.distributed.all_reduce", side_effect=collective):
            _average_gradient(parameter, world_size=2)

        self.assertIs(parameter.grad, original_gradient)
        torch.testing.assert_close(
            parameter.grad,
            torch.tensor([3.0], dtype=torch.float64),
        )


def _distributed_grown_gradient_worker(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(17 + rank)
        module = _DistributedGrowingModule(config)
        distributed_module = DistributedDataParallel(module)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        trainer = SimpleNamespace(optimizers=[optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)

        optimizer.zero_grad()
        first_input = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        distributed_module(first_input + rank).backward()
        optimizer.step()
        callback.on_train_batch_end(trainer, module, None, None, 0)
        self_name = "neuron_2_1_1"
        if self_name not in module.cluster.cluster:
            raise AssertionError(f"rank {rank} did not grow {self_name}")

        optimizer.zero_grad()
        second_input = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        distributed_module(second_input / 10 + rank * 0.75).backward()
        parameter = module.cluster.cluster[self_name].nucleus.model.weight
        if parameter.grad is None:
            raise AssertionError(f"rank {rank} has no grown-parameter gradient")
        local_gradient = parameter.grad.detach().clone()
        gathered_local_gradients = [torch.zeros_like(local_gradient) for _ in range(2)]
        torch.distributed.all_gather(gathered_local_gradients, local_gradient)
        expected_average = torch.stack(gathered_local_gradients).mean(dim=0)

        callback.on_before_optimizer_step(trainer, module, optimizer)

        torch.testing.assert_close(parameter.grad, expected_average)
        if parameter.grad.dtype != parameter.dtype:
            raise AssertionError("grown gradient dtype differs from its parameter")
        if parameter.grad.device != parameter.device:
            raise AssertionError("grown gradient device differs from its parameter")
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError("grown gradient is non-finite")
        optimizer.step()
        gathered_parameters = [torch.zeros_like(parameter) for _ in range(2)]
        torch.distributed.all_gather(gathered_parameters, parameter.detach())
        torch.testing.assert_close(gathered_parameters[0], gathered_parameters[1])
    finally:
        torch.distributed.destroy_process_group()


def _distributed_conditional_routing_worker(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(31)
        module = _DistributedGrowingModule(config)
        strategy = DDPStrategy()
        callback = NeuronClusterOptimizerSyncCallback()
        callback.setup(SimpleNamespace(strategy=strategy), module, "fit")
        distributed_module = DistributedDataParallel(
            module,
            **strategy._ddp_kwargs,
        )
        optimizer = torch.optim.SGD(
            module.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.5,
        )
        input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10

        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            parameters_before_step = {
                name: parameter.detach().clone()
                for name, parameter in module.named_parameters()
            }
            loss = distributed_module(input_tensor)
            loss.backward()
            globally_inactive_parameters = {
                name: parameter
                for name, parameter in module.named_parameters()
                if parameter.grad is None
            }
            if not globally_inactive_parameters:
                raise AssertionError(
                    "conditional routing did not leave a globally inactive parameter"
                )
            if not any(parameter.grad is not None for parameter in module.parameters()):
                raise AssertionError("conditional routing produced no active gradient")

            optimizer.step()
            for name, parameter in globally_inactive_parameters.items():
                torch.testing.assert_close(
                    parameter,
                    parameters_before_step[name],
                    rtol=0.0,
                    atol=0.0,
                )
                if optimizer.state.get(parameter):
                    raise AssertionError(
                        f"optimizer state was created for inactive parameter {name}"
                    )

        for parameter in module.parameters():
            gathered_parameters = [
                torch.zeros_like(parameter) for _ in range(world_size)
            ]
            torch.distributed.all_gather(
                gathered_parameters,
                parameter.detach(),
            )
            for other_rank_parameter in gathered_parameters[1:]:
                torch.testing.assert_close(
                    other_rank_parameter,
                    gathered_parameters[0],
                    rtol=0.0,
                    atol=0.0,
                )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_pruning_optimizer_worker(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(31)
        module = _DistributedGrowingModule(config)
        cluster = module.cluster
        pruned_name = "neuron_2_1_1"
        cluster.cluster[pruned_name] = cluster._initialize_neuron(2, 1, 1)
        pruned_parameters = tuple(cluster.cluster[pruned_name].parameters())
        pruned_parameter_ids = {id(parameter) for parameter in pruned_parameters}
        strategy = DDPStrategy()
        callback = NeuronClusterOptimizerSyncCallback()
        trainer = SimpleNamespace(
            strategy=strategy,
            optimizers=[],
            lr_scheduler_configs=[],
        )
        callback.setup(trainer, module, "fit")
        distributed_module = DistributedDataParallel(
            module,
            **strategy._ddp_kwargs,
        )
        optimizer = torch.optim.SGD(
            module.parameters(),
            lr=0.01,
            momentum=0.9,
        )
        trainer.optimizers = [optimizer]
        callback.on_fit_start(trainer, module)
        input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10

        optimizer.zero_grad(set_to_none=True)
        distributed_module(input_tensor).backward()
        optimizer.step()
        callback.on_train_batch_end(trainer, module, None, None, 0)

        if pruned_name in cluster.cluster:
            raise AssertionError(f"rank {rank} did not prune {pruned_name}")
        optimized_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimized_parameter_ids & pruned_parameter_ids:
            raise AssertionError("pruned parameters remain in the optimizer")
        if any(parameter in optimizer.state for parameter in pruned_parameters):
            raise AssertionError("pruned parameter state remains in the optimizer")

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            distributed_module(input_tensor).backward()
            optimizer.step()
            callback.on_train_batch_end(trainer, module, None, None, 0)

        gathered_topologies = [None] * world_size
        torch.distributed.all_gather_object(
            gathered_topologies,
            tuple(cluster.cluster.keys()),
        )
        if gathered_topologies != [tuple(cluster.cluster.keys())] * world_size:
            raise AssertionError(
                f"pruned topologies differ across ranks: {gathered_topologies}"
            )
        for parameter in module.parameters():
            gathered_parameters = [
                torch.zeros_like(parameter) for _ in range(world_size)
            ]
            torch.distributed.all_gather(
                gathered_parameters,
                parameter.detach(),
            )
            for other_rank_parameter in gathered_parameters[1:]:
                torch.testing.assert_close(
                    other_rank_parameter,
                    gathered_parameters[0],
                    rtol=0.0,
                    atol=0.0,
                )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_ddp_atrophy_history_worker(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        module = _DistributedAtrophyHistoryProbe(config)
        distributed_module = DistributedDataParallel(
            module,
            broadcast_buffers=True,
        )

        distributed_module(torch.ones(1))
        first_counter = int(module.cluster.cluster[module.grown_name].atrophy_counter)
        distributed_module(torch.ones(1))
        retained = module.grown_name in module.cluster.cluster
        second_counter = (
            int(module.cluster.cluster[module.grown_name].atrophy_counter)
            if retained
            else -1
        )
        local_result = (first_counter, retained, second_counter)
        gathered_results = [None] * world_size
        torch.distributed.all_gather_object(gathered_results, local_result)
        assert gathered_results == [(0, True, 1)] * world_size, (
            f"DDP buffer broadcast corrupted global atrophy history: {gathered_results}"
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_ddp_growth_history_worker(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        module = _DistributedGrowthHistoryProbe(config)
        distributed_module = DistributedDataParallel(
            module,
            broadcast_buffers=True,
        )

        distributed_module(torch.ones(1))
        first_counter = int(module.cluster.cluster[module.parent_name].batch_counter)
        grew_on_first_forward = module.grown_name in module.cluster.cluster
        distributed_module(torch.ones(1))
        grew_on_second_forward = module.grown_name in module.cluster.cluster
        parent_counter_after_growth = int(
            module.cluster.cluster[module.parent_name].batch_counter
        )
        local_result = (
            first_counter,
            grew_on_first_forward,
            grew_on_second_forward,
            parent_counter_after_growth,
        )
        gathered_results = [None] * world_size
        torch.distributed.all_gather_object(gathered_results, local_result)
        assert gathered_results == [(1, False, True, 0)] * world_size, (
            f"DDP buffer broadcast corrupted global growth history: {gathered_results}"
        )
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(
    torch.distributed.is_available() and torch.distributed.is_gloo_available(),
    "gloo process group support is required",
)
class TestNeuronDistributedOptimizerSync(NeuronTestCase):
    def test_lightning_ddp_configuration_supports_repeated_conditional_routes(
        self,
    ) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=5,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "conditional_process_group_init")
            torch.multiprocessing.spawn(
                _distributed_conditional_routing_worker,
                args=(2, init_file, config),
                nprocs=2,
                join=True,
            )

    def test_ddp_pruning_removes_optimizer_state_and_allows_later_backwards(
        self,
    ) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            pruning_threshold=1,
            neuron_config=self.neuron_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "pruning_process_group_init")
            torch.multiprocessing.spawn(
                _distributed_pruning_optimizer_worker,
                args=(2, init_file, config),
                nprocs=2,
                join=True,
            )

    def test_grown_parameter_gradient_is_averaged_outside_ddp_reducer(self) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            max_total_growths=1,
            neuron_config=self.full_sampler_neuron_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "process_group_init")
            torch.multiprocessing.spawn(
                _distributed_grown_gradient_worker,
                args=(2, init_file, config),
                nprocs=2,
                join=True,
            )

    def test_ddp_broadcast_preserves_cross_rank_atrophy_history(self) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            pruning_threshold=2,
            neuron_config=self.full_sampler_neuron_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "atrophy_process_group_init")
            torch.multiprocessing.spawn(
                _distributed_ddp_atrophy_history_worker,
                args=(2, init_file, config),
                nprocs=2,
                join=True,
            )

    def test_ddp_broadcast_preserves_cross_rank_growth_history(self) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=3,
            max_total_growths=1,
            neuron_config=self.full_sampler_neuron_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "growth_process_group_init")
            torch.multiprocessing.spawn(
                _distributed_ddp_growth_history_worker,
                args=(2, init_file, config),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
