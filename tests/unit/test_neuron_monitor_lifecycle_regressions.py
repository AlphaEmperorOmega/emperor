import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from emperor.neuron import NeuronCluster, NeuronClusterMonitorCallback


def _cluster_stub() -> NeuronCluster:
    cluster = NeuronCluster.__new__(NeuronCluster)
    nn.Module.__init__(cluster)
    cluster.beam_width = 1
    cluster.cluster = nn.ModuleDict()
    cluster.x_axis_total_neurons = 1
    cluster.y_axis_total_neurons = 1
    cluster.z_axis_total_neurons = 1
    cluster.growth_threshold = None
    cluster.pruning_threshold = None
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


class TestNeuronMonitorForwardInterface(unittest.TestCase):
    def test_wrapper_preserves_the_public_input_keyword(self) -> None:
        cluster = _cluster_stub()

        def instance_forward(input, return_trace=False):
            del return_trace
            return input, input.new_zeros(())

        cluster.forward = instance_forward
        host = _ClusterHost(cluster=cluster)
        host.eval()
        callback = NeuronClusterMonitorCallback()
        callback.on_fit_start(trainer=None, pl_module=host)
        input_batch = torch.ones(2, 3)

        try:
            output, auxiliary_loss = cluster(input=input_batch)
        finally:
            callback.on_fit_end(trainer=None, pl_module=host)

        self.assertIs(output, input_batch)
        torch.testing.assert_close(auxiliary_loss, torch.zeros(()))


class TestNeuronMonitorStepCadence(unittest.TestCase):
    def test_resumed_fit_captures_and_emits_each_completed_step_once(self) -> None:
        cluster = _cluster_stub()

        def instance_forward(input, return_trace=False):
            auxiliary_loss = input.new_zeros(())
            if return_trace:
                return input, auxiliary_loss, SimpleNamespace()
            return input, auxiliary_loss

        cluster.forward = instance_forward
        host = _ClusterHost(cluster=cluster)
        host.global_step = 6
        callback = NeuronClusterMonitorCallback(log_every_n_steps=3)
        callback.on_fit_start(trainer=None, pl_module=host)
        tracked_contexts = []

        try:
            with (
                patch(
                    "emperor.neuron._monitoring.callback."
                    "_NeuronDiagnostics.calculate_route",
                    return_value=None,
                ),
                patch(
                    "emperor.neuron._monitoring.callback."
                    "_NeuronDiagnostics.calculate_entry_routing",
                    return_value=None,
                ),
                patch.object(
                    callback,
                    "_NeuronClusterMonitorCallback__track_neuron_cluster_diagnostics",
                    side_effect=tracked_contexts.append,
                ),
            ):
                callback.on_train_batch_end(None, host, None, None, 5)
                host.global_step = 8
                cluster(torch.ones(2, 3))
                self.assertEqual(callback._latest_observation_steps["cluster"], 9)

                host.global_step = 9
                callback.on_train_batch_end(None, host, None, None, 8)
                callback.on_train_batch_end(None, host, None, None, 8)
        finally:
            callback.on_fit_end(trainer=None, pl_module=host)

        self.assertEqual(len(tracked_contexts), 1)
        self.assertIsNotNone(tracked_contexts[0].observation)


class TestNeuronMonitorTopologyEvents(unittest.TestCase):
    def test_changes_between_emissions_are_accumulated(self) -> None:
        cluster = _cluster_stub()
        cluster.cluster["a"] = nn.Identity()
        host = _ClusterHost(cluster=cluster)
        callback = NeuronClusterMonitorCallback(log_every_n_steps=3)
        callback.on_fit_start(trainer=None, pl_module=host)
        tracked_contexts = []

        try:
            with patch.object(
                callback,
                "_NeuronClusterMonitorCallback__track_neuron_cluster_diagnostics",
                side_effect=tracked_contexts.append,
            ):
                cluster.cluster["b"] = nn.Identity()
                host.global_step = 1
                callback.on_train_batch_end(None, host, None, None, 0)

                del cluster.cluster["a"]
                cluster.cluster["c"] = nn.Identity()
                host.global_step = 2
                callback.on_train_batch_end(None, host, None, None, 1)

                host.global_step = 3
                callback.on_train_batch_end(None, host, None, None, 2)
        finally:
            callback.on_fit_end(trainer=None, pl_module=host)

        self.assertEqual(len(tracked_contexts), 1)
        self.assertEqual(tracked_contexts[0].growth_events, 2.0)
        self.assertEqual(tracked_contexts[0].pruning_events, 1.0)
