from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from emperor.neuron import NeuronClusterConfig, NeuronClusterMonitorCallback
from unit.test_neuron import NeuronTestCase


class _MonitoredNeuronModule(LightningModule):
    def __init__(self, cluster_config: NeuronClusterConfig) -> None:
        super().__init__()
        self.cluster = cluster_config.build()
        self.monitor_logs: list[tuple[int, str]] = []

    def training_step(self, batch, batch_idx):
        output, auxiliary_loss = self.cluster(batch[0])
        return output.square().mean() + auxiliary_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def log(self, name, value, *args, **kwargs):
        self.monitor_logs.append((int(self.global_step), name))
        return super().log(name, value, *args, **kwargs)


class NeuronMonitorLifecycleIntegrationTests(NeuronTestCase):
    def test_real_trainer_captures_and_emits_route_metrics_on_step_two(self) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            neuron_config=self.full_sampler_neuron_config(),
        )
        model = _MonitoredNeuronModule(config)
        inputs = (
            torch.arange(8 * self.input_dim, dtype=torch.float32).reshape(
                8, self.input_dim
            )
            / 100
        )
        loader = DataLoader(TensorDataset(inputs), batch_size=4, shuffle=False)
        callback = NeuronClusterMonitorCallback(log_every_n_steps=2)

        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=2,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[callback],
        )
        trainer.fit(model, train_dataloaders=loader)

        route_tag = "cluster/cluster/route/depth_mean"
        self.assertEqual(
            [entry for entry in model.monitor_logs if entry[1] == route_tag],
            [(2, route_tag)],
        )
        self.assertNotIn("forward", model.cluster.__dict__)


if __name__ == "__main__":
    unittest.main()
