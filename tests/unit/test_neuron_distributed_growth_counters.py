import unittest
from unittest.mock import patch

import torch
from torch import nn

from emperor.neuron._cluster.plasticity import _NeuronClusterPlasticityMixin


class _CounterNeuron(nn.Module):
    def __init__(self, *, batch_counter: int = 0, atrophy_counter: int = 0) -> None:
        super().__init__()
        self.register_buffer("batch_counter", torch.tensor(batch_counter))
        self.register_buffer("atrophy_counter", torch.tensor(atrophy_counter))


class TestDistributedNeuronAtrophyCounters(unittest.TestCase):
    def test_reduced_atrophy_counters_are_persisted_on_every_rank(self) -> None:
        plasticity = _NeuronClusterPlasticityMixin()
        plasticity.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": _CounterNeuron(atrophy_counter=8),
                "neuron_2_1_1": _CounterNeuron(atrophy_counter=5),
            }
        )
        synchronize = (
            plasticity
            ._NeuronClusterPlasticityMixin__synchronize_atrophy_counters_across_ranks
        )

        def reduce_to_global_minimum(
            counters: torch.Tensor,
            *,
            op: torch.distributed.ReduceOp,
        ) -> None:
            self.assertIs(op, torch.distributed.ReduceOp.MIN)
            counters.copy_(torch.tensor([3, 4]))

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_reduce", side_effect=reduce_to_global_minimum),
        ):
            synchronized = synchronize()

        self.assertEqual(
            synchronized,
            {"neuron_1_1_1": 3, "neuron_2_1_1": 4},
        )
        self.assertEqual(
            [int(neuron.atrophy_counter) for neuron in plasticity.cluster.values()],
            [3, 4],
        )
