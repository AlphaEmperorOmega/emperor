import unittest
from unittest.mock import patch

import torch
from torch import nn

from emperor.neuron._cluster.model import NeuronCluster
from emperor.neuron._cluster.plasticity import _NeuronClusterPlasticityMixin

_SYNC_BATCH_COUNTERS = (
    "_NeuronClusterPlasticityMixin__synchronize_batch_counters_across_ranks"
)
_SYNC_ESCAPE_COUNTS = (
    "_NeuronClusterPlasticityMixin__synchronize_escape_counts_across_ranks"
)


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


class TestDistributedNeuronGrowthCounters(unittest.TestCase):
    def setUp(self) -> None:
        self.plasticity = _NeuronClusterPlasticityMixin()
        self.plasticity.growth_threshold = 10_000
        self.plasticity.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": _CounterNeuron(batch_counter=5),
                "neuron_2_1_1": _CounterNeuron(batch_counter=7),
            }
        )
        self.plasticity.escape_counts = torch.tensor([7])
        self.plasticity._growth_counters_are_global = True

    def test_global_history_adds_each_rank_delta_once(self) -> None:
        baseline = self.plasticity._capture_growth_counter_baseline()
        self.plasticity.cluster["neuron_1_1_1"].batch_counter.add_(1)
        self.plasticity.cluster["neuron_2_1_1"].batch_counter.add_(2)
        self.plasticity.escape_counts.add_(1)
        synchronize_batch = getattr(self.plasticity, _SYNC_BATCH_COUNTERS)
        synchronize_escape = getattr(self.plasticity, _SYNC_ESCAPE_COUNTS)

        def add_remote_contribution(
            counters: torch.Tensor,
            *,
            op: torch.distributed.ReduceOp,
        ) -> None:
            self.assertIs(op, torch.distributed.ReduceOp.SUM)
            if counters.shape == torch.Size([2]):
                counters.add_(torch.tensor([3, 4]))
            else:
                counters.add_(torch.tensor([2]))

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.all_reduce", side_effect=add_remote_contribution),
        ):
            synchronized_batch = synchronize_batch(baseline)
            synchronized_escape = synchronize_escape(baseline)

        self.assertEqual(
            synchronized_batch,
            {"neuron_1_1_1": 9, "neuron_2_1_1": 13},
        )
        self.assertEqual(
            [int(neuron.batch_counter) for neuron in self.plasticity.cluster.values()],
            [9, 13],
        )
        torch.testing.assert_close(synchronized_escape, torch.tensor([10]))
        torch.testing.assert_close(self.plasticity.escape_counts, torch.tensor([10]))

    def test_loaded_growth_counters_are_marked_global(self) -> None:
        self.plasticity._growth_counters_are_global = False

        self.plasticity._mark_growth_counters_global_after_load(None, None)

        self.assertTrue(self.plasticity._growth_counters_are_global)

    def test_forward_passes_the_captured_baseline_to_growth(self) -> None:
        model = NeuronCluster.__new__(NeuronCluster)
        nn.Module.__init__(model)
        model.beam_width = 1
        model.input_dim = 2
        model.train()
        baseline = object()
        input_batch = torch.ones(2, 2)

        with (
            patch.object(
                model,
                "_propagate_signal_through_recurrent_routes",
                return_value=(input_batch.clone(), torch.zeros(()), None),
            ),
            patch.object(
                model,
                "_capture_growth_counter_baseline",
                return_value=baseline,
            ),
            patch.object(model, "_advance_grown_neuron_warmup"),
            patch.object(model, "_check_neuron_growth") as check_growth,
            patch.object(model, "_check_neuron_atrophy"),
        ):
            model(input_batch)

        check_growth.assert_called_once_with(baseline)
