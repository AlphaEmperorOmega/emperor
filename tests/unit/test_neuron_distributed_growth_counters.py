import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from emperor.neuron import NeuronCluster
from emperor.neuron._cluster.plasticity import ClusterPlasticityDelegate
from emperor.neuron._cluster.recurrent_routes import ClusterRoutingDelegate
from emperor.neuron._cluster.state import _NeuronClusterForwardContext
from emperor.neuron._cluster.topology import ClusterTopologyDelegate

_SYNC_BATCH_COUNTERS = (
    "_ClusterPlasticityDelegate__synchronize_batch_counters_across_ranks"
)
_SYNC_ESCAPE_COUNTS = (
    "_ClusterPlasticityDelegate__synchronize_escape_counts_across_ranks"
)
_FIND_GROWTH_POSITION = "_ClusterPlasticityDelegate__find_closest_empty_connection"
_INITIALIZE_GROWN_NEURON = (
    "_ClusterPlasticityDelegate__initialize_grown_neuron_with_synchronized_rng"
)


class _CounterNeuron(nn.Module):
    def __init__(self, *, batch_counter: int = 0, atrophy_counter: int = 0) -> None:
        super().__init__()
        self.register_buffer("batch_counter", torch.tensor(batch_counter))
        self.register_buffer("atrophy_counter", torch.tensor(atrophy_counter))


class TestDistributedNeuronAtrophyCounters(unittest.TestCase):
    def test_reduced_atrophy_counters_are_persisted_on_every_rank(self) -> None:
        owner = SimpleNamespace()
        plasticity = ClusterPlasticityDelegate(owner, ClusterTopologyDelegate(owner))
        owner.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": _CounterNeuron(atrophy_counter=8),
                "neuron_2_1_1": _CounterNeuron(atrophy_counter=5),
            }
        )
        synchronize = plasticity._ClusterPlasticityDelegate__synchronize_atrophy_counters_across_ranks

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
            [int(neuron.atrophy_counter) for neuron in owner.cluster.values()],
            [3, 4],
        )


class TestDistributedNeuronGrowthCounters(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = SimpleNamespace()
        self.plasticity = ClusterPlasticityDelegate(
            self.owner, ClusterTopologyDelegate(self.owner)
        )
        self.owner.growth_threshold = 10_000
        self.owner.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": _CounterNeuron(batch_counter=5),
                "neuron_2_1_1": _CounterNeuron(batch_counter=7),
            }
        )
        self.owner.escape_counts = torch.tensor([7])
        self.owner._growth_counters_are_global = True

    def test_global_history_adds_each_rank_delta_once(self) -> None:
        baseline = self.plasticity.capture_growth_counter_baseline()
        self.owner.cluster["neuron_1_1_1"].batch_counter.add_(1)
        self.owner.cluster["neuron_2_1_1"].batch_counter.add_(2)
        self.owner.escape_counts.add_(1)
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
            [int(neuron.batch_counter) for neuron in self.owner.cluster.values()],
            [9, 13],
        )
        torch.testing.assert_close(synchronized_escape, torch.tensor([10]))
        torch.testing.assert_close(self.owner.escape_counts, torch.tensor([10]))

    def test_loaded_growth_counters_are_marked_global(self) -> None:
        self.owner._growth_counters_are_global = False

        self.plasticity.mark_growth_counters_global_after_load(None, None)

        self.assertTrue(self.owner._growth_counters_are_global)

    def test_forward_passes_the_captured_baseline_to_growth(self) -> None:
        model = NeuronCluster.__new__(NeuronCluster)
        nn.Module.__init__(model)
        model.beam_width = 1
        model.input_dim = 2
        model.train()
        topology = ClusterTopologyDelegate(model)
        plasticity = ClusterPlasticityDelegate(model, topology)
        routing = ClusterRoutingDelegate(model, topology, plasticity)
        model._NeuronCluster__plasticity = plasticity
        model._NeuronCluster__routing = routing
        baseline = object()
        input_batch = torch.ones(2, 2)

        with (
            patch.object(
                routing,
                "propagate",
                return_value=(input_batch.clone(), torch.zeros(()), None),
            ),
            patch.object(
                plasticity,
                "capture_growth_counter_baseline",
                return_value=baseline,
            ),
            patch.object(plasticity, "advance_grown_neuron_warmup"),
            patch.object(plasticity, "check_neuron_growth") as check_growth,
            patch.object(plasticity, "check_neuron_atrophy") as check_atrophy,
        ):
            model(input_batch)

        check_growth.assert_called_once()
        check_atrophy.assert_called_once()
        growth_baseline, growth_context = check_growth.call_args.args
        (atrophy_context,) = check_atrophy.call_args.args
        self.assertIs(growth_baseline, baseline)
        self.assertIs(growth_context, atrophy_context)
        self.assertIsInstance(growth_context, _NeuronClusterForwardContext)


class TestNeuronGrowthCounterTransactions(unittest.TestCase):
    def test_failed_growth_preserves_the_saturated_counter(self) -> None:
        owner = SimpleNamespace()
        plasticity = ClusterPlasticityDelegate(owner, ClusterTopologyDelegate(owner))
        saturated_neuron = _CounterNeuron(batch_counter=5)
        owner.growth_threshold = 1
        owner.cluster = nn.ModuleDict({"neuron_1_1_1": saturated_neuron})
        owner.escape_counts = None
        owner.total_growth_count = None
        owner.forwards_since_last_growth = None
        owner._growth_counters_are_global = False
        owner._add_neuron = lambda _cluster, _name, _neuron: None

        with (
            patch.object(
                plasticity,
                _FIND_GROWTH_POSITION,
                return_value=(2, 1, 1),
            ),
            patch.object(
                plasticity,
                _INITIALIZE_GROWN_NEURON,
                side_effect=RuntimeError("initializer failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "initializer failed"),
        ):
            plasticity.check_neuron_growth(
                None,
                _NeuronClusterForwardContext(),
            )

        self.assertEqual(int(saturated_neuron.batch_counter), 5)
