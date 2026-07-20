import unittest

from torch import nn

from emperor.neuron import NeuronClusterMonitorCallback
from emperor.neuron._cluster.model import NeuronCluster


def _cluster_stub() -> NeuronCluster:
    cluster = NeuronCluster.__new__(NeuronCluster)
    nn.Module.__init__(cluster)
    cluster.beam_width = 1
    cluster.cluster = nn.ModuleDict()
    return cluster


class _ClusterHost(nn.Module):
    def __init__(self, **clusters: NeuronCluster) -> None:
        super().__init__()
        for name, cluster in clusters.items():
            self.add_module(name, cluster)
        self.global_step = 0
        self.logger = None


class TestNeuronMonitorOwnership(unittest.TestCase):
    def test_second_monitor_cannot_replace_an_existing_wrapper(self) -> None:
        cluster = _cluster_stub()
        host = _ClusterHost(cluster=cluster)
        first = NeuronClusterMonitorCallback()
        second = NeuronClusterMonitorCallback()
        original_forward = cluster.forward

        first.on_fit_start(trainer=None, pl_module=host)
        first_wrapper = cluster.forward
        with self.assertRaisesRegex(RuntimeError, "already monitored"):
            second.on_fit_start(trainer=None, pl_module=host)

        self.assertIs(cluster.forward, first_wrapper)
        second.on_fit_end(trainer=None, pl_module=host)
        self.assertIs(cluster.forward, first_wrapper)
        first.on_fit_end(trainer=None, pl_module=host)
        self.assertEqual(cluster.forward, original_forward)
        self.assertNotIn(first._OWNER_ATTRIBUTE, cluster.__dict__)

    def test_partial_setup_failure_restores_previously_wrapped_clusters(self) -> None:
        first_cluster = _cluster_stub()
        second_cluster = _cluster_stub()
        host = _ClusterHost(first=first_cluster, second=second_cluster)
        callback = NeuronClusterMonitorCallback()
        foreign_owner = object()
        second_cluster.__dict__[callback._OWNER_ATTRIBUTE] = foreign_owner

        with self.assertRaisesRegex(RuntimeError, "already monitored"):
            callback.on_fit_start(trainer=None, pl_module=host)

        self.assertNotIn("forward", first_cluster.__dict__)
        self.assertNotIn(callback._OWNER_ATTRIBUTE, first_cluster.__dict__)
        self.assertIs(
            second_cluster.__dict__[callback._OWNER_ATTRIBUTE],
            foreign_owner,
        )
