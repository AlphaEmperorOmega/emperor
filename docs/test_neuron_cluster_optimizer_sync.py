import unittest

from unittest import mock

import torch
import torch.nn as nn

from docs.test_neuron import NeuronTestCase
from emperor.neuron import NeuronClusterConfig
from emperor.neuron.core import NeuronClusterOptimizerSyncCallback


class FakeLightningModule(nn.Module):
    def __init__(self, cluster):
        super().__init__()
        self.neuron_cluster = cluster
        self.other = nn.Linear(cluster.input_dim, cluster.input_dim)


class FakeTrainer:
    def __init__(self, optimizers):
        self.optimizers = optimizers


class TestNeuronClusterOptimizerSyncCallback(NeuronTestCase):
    def build_growing_cluster(self):
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=4,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

    def grow_once(self, cluster):
        initial_keys = set(cluster.cluster.keys())
        self.assertEqual(len(initial_keys), 4)

        cluster.train()
        cluster(torch.randn(self.batch_size, self.input_dim))

        added_keys = set(cluster.cluster.keys()) - initial_keys
        self.assertEqual(len(cluster.cluster), 5)
        self.assertEqual(len(added_keys), 1)
        return cluster.cluster[added_keys.pop()]

    def optimizer_param_ids(self, optimizer) -> set[int]:
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    def test_sync_adds_grown_neuron_params_to_cluster_optimizer(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=0.012,
            weight_decay=0.034,
        )
        trainer = FakeTrainer([optimizer])

        new_neuron = self.grow_once(cluster)
        new_params = [
            parameter
            for parameter in new_neuron.parameters()
            if parameter.requires_grad
        ]
        self.assertGreater(len(new_params), 0)
        before_sync_param_ids = self.optimizer_param_ids(optimizer)
        self.assertTrue(
            all(
                id(parameter) not in before_sync_param_ids
                for parameter in new_params
            )
        )

        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        after_sync_param_ids = self.optimizer_param_ids(optimizer)
        self.assertTrue(
            all(id(parameter) in after_sync_param_ids for parameter in new_params)
        )

    def test_sync_is_idempotent(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        self.grow_once(cluster)
        callback = NeuronClusterOptimizerSyncCallback()

        callback.sync_optimizers(trainer, module)
        param_group_count = len(optimizer.param_groups)
        param_count = sum(len(group["params"]) for group in optimizer.param_groups)
        callback.sync_optimizers(trainer, module)

        self.assertEqual(len(optimizer.param_groups), param_group_count)
        self.assertEqual(
            sum(len(group["params"]) for group in optimizer.param_groups),
            param_count,
        )

    def test_sync_copies_reference_param_group_options(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=0.012,
            weight_decay=0.034,
            betas=(0.8, 0.9),
        )
        trainer = FakeTrainer([optimizer])
        self.grow_once(cluster)

        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        self.assertEqual(len(optimizer.param_groups), 2)
        copied_group = optimizer.param_groups[-1]
        self.assertEqual(copied_group["lr"], 0.012)
        self.assertEqual(copied_group["weight_decay"], 0.034)
        self.assertEqual(copied_group["betas"], (0.8, 0.9))

    def test_sync_skips_optimizers_without_existing_cluster_params(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        cluster_optimizer = torch.optim.Adam(module.neuron_cluster.parameters())
        other_optimizer = torch.optim.Adam(module.other.parameters())
        trainer = FakeTrainer([cluster_optimizer, other_optimizer])

        new_neuron = self.grow_once(cluster)
        new_param_ids = {
            id(parameter)
            for parameter in new_neuron.parameters()
            if parameter.requires_grad
        }

        NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        self.assertTrue(
            new_param_ids.issubset(self.optimizer_param_ids(cluster_optimizer))
        )
        self.assertTrue(
            new_param_ids.isdisjoint(self.optimizer_param_ids(other_optimizer))
        )

    def test_sync_warns_when_no_optimizer_owns_cluster_params(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        other_optimizer = torch.optim.Adam(module.other.parameters())
        trainer = FakeTrainer([other_optimizer])

        with self.assertWarns(UserWarning):
            NeuronClusterOptimizerSyncCallback().sync_optimizers(trainer, module)

        self.assertEqual(len(other_optimizer.param_groups), 1)

    def test_batch_hooks_skip_rescan_without_growth(self):
        cluster = self.build_growing_cluster()
        module = FakeLightningModule(cluster)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.012)
        trainer = FakeTrainer([optimizer])
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)

        with mock.patch.object(
            callback,
            "sync_optimizers",
            wraps=callback.sync_optimizers,
        ) as sync_spy:
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            callback.on_train_batch_end(
                trainer, module, outputs=None, batch=None, batch_idx=0
            )
            self.assertEqual(sync_spy.call_count, 0)

            new_neuron = self.grow_once(cluster)
            callback.on_train_batch_end(
                trainer, module, outputs=None, batch=None, batch_idx=1
            )
            self.assertEqual(sync_spy.call_count, 1)

        new_param_ids = {
            id(parameter)
            for parameter in new_neuron.parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            new_param_ids.issubset(self.optimizer_param_ids(optimizer))
        )


if __name__ == "__main__":
    unittest.main()
