from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader, TensorDataset

from emperor.neuron import NeuronClusterConfig, NeuronClusterMonitorCallback
from unit.test_neuron import NeuronTestCase, ScriptedNeuron, ScriptedSampler


class _MonitoredNeuronModule(LightningModule):
    def __init__(self, cluster_config: NeuronClusterConfig) -> None:
        super().__init__()
        self.cluster = cluster_config.build()
        self.monitor_logs: list[tuple[int, str]] = []
        self.monitor_values: list[tuple[int, str, object]] = []

    def training_step(self, batch, batch_idx):
        output, auxiliary_loss = self.cluster(batch[0])
        return output.square().mean() + auxiliary_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def log(self, name, value, *args, **kwargs):
        self.monitor_logs.append((int(self.global_step), name))
        recorded_value = (
            value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        )
        self.monitor_values.append((int(self.global_step), name, recorded_value))
        return super().log(name, value, *args, **kwargs)


class _CheckpointedMonitoredNeuronModule(_MonitoredNeuronModule):
    def __init__(self, cluster_config: NeuronClusterConfig) -> None:
        super().__init__(cluster_config)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.cluster.cluster = torch.nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0, 2.0, 3.0, 4.0],
                )
            }
        )
        self.cluster.entry_sampler = ScriptedSampler(
            indices=[0],
            probabilities=[1.0],
        )

    def training_step(self, batch, batch_idx):
        output, auxiliary_loss = self.cluster(batch[0])
        return output.square().mean() * self.scale + auxiliary_loss


class _FitStartRecordingMonitor(NeuronClusterMonitorCallback):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fit_start_global_steps: list[int] = []
        self.fit_start_emission_steps: list[int] = []
        self.train_start_global_steps: list[int] = []
        self.train_start_emission_steps: list[int] = []

    def on_fit_start(self, trainer, pl_module) -> None:
        super().on_fit_start(trainer, pl_module)
        self.fit_start_global_steps.append(int(pl_module.global_step))
        self.fit_start_emission_steps.append(self._last_emitted_step)

    def on_train_start(self, trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        self.train_start_global_steps.append(int(pl_module.global_step))
        self.train_start_emission_steps.append(self._last_emitted_step)


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

    def test_real_trainer_resume_emits_once_per_documented_step(self) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        )
        inputs = (
            torch.arange(
                8 * self.input_dim,
                dtype=torch.float32,
            ).reshape(8, self.input_dim)
            / 100
        )
        loader = DataLoader(TensorDataset(inputs), batch_size=2, shuffle=False)
        route_depth_tag = "cluster/cluster/route/depth_mean"
        route_histogram_tag = "cluster/cluster/histogram/route_depth"
        utilization_tag = "cluster/cluster/heatmap/neuron_utilization"

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_model = _CheckpointedMonitoredNeuronModule(config)
            source_callback = _FitStartRecordingMonitor(log_every_n_steps=2)
            source_logger = TensorBoardLogger(
                temporary_directory,
                name="source",
                version="run",
            )
            source_trainer = Trainer(
                accelerator="cpu",
                default_root_dir=temporary_directory,
                max_epochs=1,
                limit_train_batches=2,
                logger=source_logger,
                enable_checkpointing=True,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                callbacks=[source_callback],
            )
            source_trainer.fit(source_model, train_dataloaders=loader)
            checkpoint = Path(temporary_directory) / "monitor.ckpt"
            source_trainer.save_checkpoint(checkpoint)

            resumed_model = _CheckpointedMonitoredNeuronModule(config)
            resumed_callback = _FitStartRecordingMonitor(log_every_n_steps=2)
            resumed_logger = TensorBoardLogger(
                temporary_directory,
                name="resumed",
                version="run",
            )
            resumed_trainer = Trainer(
                accelerator="cpu",
                default_root_dir=temporary_directory,
                max_epochs=2,
                limit_train_batches=4,
                logger=resumed_logger,
                enable_checkpointing=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                callbacks=[resumed_callback],
            )
            resumed_trainer.fit(
                resumed_model,
                train_dataloaders=loader,
                ckpt_path=checkpoint,
            )

            event_size_guidance = {"histograms": 0, "images": 0}
            source_events = EventAccumulator(
                source_logger.log_dir,
                size_guidance=event_size_guidance,
            ).Reload()
            resumed_events = EventAccumulator(
                resumed_logger.log_dir,
                size_guidance=event_size_guidance,
            ).Reload()

        self.assertEqual(source_callback.fit_start_global_steps, [0])
        self.assertEqual(source_callback.fit_start_emission_steps, [0])
        self.assertEqual(resumed_callback.fit_start_global_steps, [0])
        self.assertEqual(resumed_callback.fit_start_emission_steps, [0])
        self.assertEqual(source_callback.train_start_global_steps, [0])
        self.assertEqual(source_callback.train_start_emission_steps, [0])
        self.assertEqual(resumed_callback.train_start_global_steps, [2])
        self.assertEqual(resumed_callback.train_start_emission_steps, [2])
        self.assertEqual(
            [
                (step, name)
                for step, name in source_model.monitor_logs
                if name == route_depth_tag
            ],
            [(2, route_depth_tag)],
        )
        self.assertEqual(
            [
                (step, name)
                for step, name in resumed_model.monitor_logs
                if name == route_depth_tag
            ],
            [(4, route_depth_tag), (6, route_depth_tag)],
        )
        for step, name, value in resumed_model.monitor_values:
            if name == route_depth_tag:
                self.assertIn(step, (4, 6))
                torch.testing.assert_close(value, torch.tensor(2.0))
        self.assertEqual(
            [event.step for event in source_events.Histograms(route_histogram_tag)],
            [2],
        )
        self.assertEqual(
            [event.step for event in resumed_events.Histograms(route_histogram_tag)],
            [4, 6],
        )
        self.assertEqual(
            [event.step for event in resumed_events.Images(utilization_tag)],
            [4, 6],
        )
        self.assertNotIn("forward", source_model.cluster.__dict__)
        self.assertNotIn("forward", resumed_model.cluster.__dict__)


if __name__ == "__main__":
    unittest.main()
