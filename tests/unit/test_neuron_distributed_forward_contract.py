from __future__ import annotations

import os
import tempfile
import unittest

import torch
from torch import Tensor, nn

from emperor.neuron import NeuronClusterConfig
from unit.test_neuron import NeuronTestCase, ScriptedNeuron, ScriptedSampler


def _initialize_process_group(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _assert_equal_across_ranks(world_size: int, value, label: str) -> None:
    gathered_values = [None] * world_size
    torch.distributed.all_gather_object(gathered_values, value)
    assert all(value == gathered_values[0] for value in gathered_values), (
        f"{label} diverged across ranks: {gathered_values}"
    )


def _assert_tensors_equal_across_ranks(
    world_size: int,
    tensors: dict[str, Tensor],
    label: str,
) -> None:
    gathered_tensors = [None] * world_size
    torch.distributed.all_gather_object(gathered_tensors, tensors)
    reference = gathered_tensors[0]
    for rank_index, rank_tensors in enumerate(gathered_tensors[1:], start=1):
        assert rank_tensors.keys() == reference.keys(), (
            f"{label} keys diverged on rank {rank_index}"
        )
        for name, reference_tensor in reference.items():
            assert torch.equal(rank_tensors[name], reference_tensor), (
                f"{label} diverged for {name} on rank {rank_index}"
            )


def _distributed_forward_worker(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _initialize_process_group(rank, world_size, init_file)
    try:
        torch.manual_seed(0)
        model = config.build()
        parent_delta = [1.0, 2.0, 3.0, 4.0]
        parent = ScriptedNeuron(
            routes=[[2, 1, 1]],
            probabilities=[1.0],
            delta=parent_delta,
        )
        model.cluster = nn.ModuleDict({"neuron_1_1_1": parent})
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])

        torch.manual_seed(100 + rank)
        input_batch = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
            ]
        )
        expected_pre_growth_output = input_batch + torch.tensor(parent_delta)

        output, auxiliary_loss = model(input_batch)

        torch.testing.assert_close(output, expected_pre_growth_output)
        torch.testing.assert_close(auxiliary_loss, torch.tensor(0.0))
        assert sorted(model.cluster) == ["neuron_1_1_1", "neuron_2_1_1"]
        assert model._growth_counters_are_global
        assert int(model.cluster["neuron_1_1_1"].batch_counter) == 0
        assert int(model.cluster["neuron_2_1_1"].batch_counter) == 0

        grown_neuron_state = {
            name: value.detach().cpu().clone()
            for name, value in model.cluster["neuron_2_1_1"].state_dict().items()
        }
        _assert_tensors_equal_across_ranks(
            world_size,
            grown_neuron_state,
            "seeded grown-neuron state",
        )
        _assert_equal_across_ranks(
            world_size,
            sorted(model.cluster),
            "post-growth topology",
        )

        model(input_batch)

        second_topology = sorted(model.cluster)
        second_counters = {
            name: int(neuron.batch_counter) for name, neuron in model.cluster.items()
        }
        assert second_topology == ["neuron_1_1_1", "neuron_2_1_1"]
        _assert_equal_across_ranks(
            world_size,
            second_topology,
            "no-growth topology",
        )
        _assert_equal_across_ranks(
            world_size,
            second_counters,
            "global batch counters",
        )
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(
    torch.distributed.is_available() and torch.distributed.is_gloo_available(),
    "gloo process group support is required",
)
class TestDistributedTopologyForward(NeuronTestCase):
    def test_real_cluster_growth_is_seeded_identically_and_ordered(self) -> None:
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

        with tempfile.TemporaryDirectory() as temporary_directory:
            init_file = os.path.join(temporary_directory, "process_group_init")
            torch.multiprocessing.spawn(
                _distributed_forward_worker,
                args=(2, init_file, config),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
