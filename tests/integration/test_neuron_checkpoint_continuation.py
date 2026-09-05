from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pytest
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader, TensorDataset

from emperor.neuron import NeuronClusterConfig, NeuronClusterOptimizerSyncCallback
from model_runtime.runs.checkpoints import (
    CheckpointContinuation,
    CheckpointContinuationLifecycle,
)
from unit.test_neuron import NeuronTestCase

OPTIMIZER_LAYOUT_CHECKPOINT_KEY = "emperor_neuron_optimizer_layout"


@dataclass(frozen=True)
class _SingleRunPlan:
    runs: tuple[object, ...] = (object(),)


@dataclass(frozen=True)
class _TargetTrainingRun:
    num_epochs: int = 3


class _GrowingNeuronModule(LightningModule):
    def __init__(self, cluster_config: NeuronClusterConfig) -> None:
        super().__init__()
        self.cluster = cluster_config.build()

    def training_step(self, batch, batch_idx):
        output, auxiliary_loss = self.cluster(batch[0])
        return output.square().mean() + auxiliary_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def configure_callbacks(self):
        return [NeuronClusterOptimizerSyncCallback()]


class _ReversedGrowingNeuronModule(_GrowingNeuronModule):
    def configure_optimizers(self):
        return torch.optim.Adam(reversed(list(self.parameters())), lr=0.01)


class _DelayedCheckpointStrategy(SingleDeviceStrategy):
    @property
    def restore_checkpoint_after_setup(self) -> bool:
        return True


class _StopAfterTwoUpdates(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        trainer.should_stop = trainer.global_step >= 2


class _GrownParameterContinuationProbe(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.restored_parameter: torch.Tensor | None = None
        self.restored_state: dict[str, object] | None = None
        self.updated_parameter: torch.Tensor | None = None
        self.updated_state: dict[str, object] | None = None

    @staticmethod
    def _child_parameter(module: _GrowingNeuronModule) -> torch.nn.Parameter:
        return module.cluster.cluster["neuron_2_1_1"].nucleus.model.weight

    @staticmethod
    def _clone_state(state: dict[str, object]) -> dict[str, object]:
        return {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }

    def on_train_start(self, trainer, pl_module) -> None:
        parameter = self._child_parameter(pl_module)
        optimizer = trainer.optimizers[0]
        self.restored_parameter = parameter.detach().clone()
        self.restored_state = self._clone_state(optimizer.state[parameter])

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        parameter = self._child_parameter(pl_module)
        self.updated_parameter = parameter.detach().clone()
        self.updated_state = self._clone_state(trainer.optimizers[0].state[parameter])


class _OptimizerStateIdentityProbe(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.restored_exp_avg: dict[str, torch.Tensor] = {}
        self.updated_exp_avg: dict[str, torch.Tensor] = {}
        self.restored_group_count = 0

    @staticmethod
    def _capture(trainer, pl_module) -> dict[str, torch.Tensor]:
        optimizer = trainer.optimizers[0]
        return {
            name: optimizer.state[parameter]["exp_avg"].detach().clone()
            for name, parameter in pl_module.named_parameters()
        }

    def on_train_start(self, trainer, pl_module) -> None:
        self.restored_group_count = len(trainer.optimizers[0].param_groups)
        self.restored_exp_avg = self._capture(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        self.updated_exp_avg = self._capture(trainer, pl_module)


class NeuronCheckpointContinuationIntegrationTests(NeuronTestCase):
    @pytest.mark.training
    def test_reused_trainer_continues_adam_with_early_and_delayed_restore(self) -> None:
        for delayed in (False, True):
            with (
                self.subTest(delayed=delayed),
                tempfile.TemporaryDirectory() as directory,
            ):
                torch.manual_seed(17)
                module = _GrowingNeuronModule(self.build_config())
                trainer = Trainer(
                    accelerator="cpu",
                    strategy=_DelayedCheckpointStrategy(device="cpu")
                    if delayed
                    else "auto",
                    default_root_dir=directory,
                    max_epochs=3,
                    max_steps=3,
                    limit_train_batches=2,
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    num_sanity_val_steps=0,
                    callbacks=[_StopAfterTwoUpdates()],
                )
                loader = self.build_loader()
                trainer.fit(module, train_dataloaders=loader)
                self.assertEqual(trainer.global_step, 2)
                previous_optimizer = trainer.optimizers[0]
                source_child = module.cluster.cluster[
                    "neuron_2_1_1"
                ].nucleus.model.weight
                expected_parameter = source_child.detach().clone()
                expected_state = self.clone_optimizer_state(
                    previous_optimizer.state[source_child]
                )
                checkpoint_path = Path(directory) / "resume.ckpt"
                trainer.save_checkpoint(checkpoint_path)
                resumed_module = _GrowingNeuronModule(self.build_config())
                probe = _GrownParameterContinuationProbe()
                trainer.callbacks.append(probe)
                trainer.fit(
                    resumed_module, train_dataloaders=loader, ckpt_path=checkpoint_path
                )
                self.assertIsNot(trainer.optimizers[0], previous_optimizer)
                self.assertEqual(trainer.global_step, 3)
                torch.testing.assert_close(probe.restored_parameter, expected_parameter)
                for name, value in expected_state.items():
                    torch.testing.assert_close(probe.restored_state[name], value)
                self.assertFalse(
                    torch.equal(probe.updated_parameter, expected_parameter)
                )
                self.assertEqual(
                    probe.updated_state["step"], expected_state["step"] + 1
                )
                self.assertTrue(torch.isfinite(probe.updated_state["exp_avg"]).all())

    def test_late_restoration_preserves_original_ddp_registration_snapshot(
        self,
    ) -> None:
        source = _GrowingNeuronModule(self.build_config())
        with torch.no_grad():
            source.cluster(torch.ones(3, self.input_dim))
        source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
        saved_states = [source_optimizer.state_dict()]
        checkpoint = {
            "optimizer_states": saved_states,
            "lr_schedulers": [],
        }
        NeuronClusterOptimizerSyncCallback().on_save_checkpoint(
            type("SourceTrainer", (), {"optimizers": [source_optimizer]})(),
            source,
            checkpoint,
        )
        module = _GrowingNeuronModule(self.build_config())
        optimizer = torch.optim.Adam(module.parameters(), lr=0.02)
        trainer = type(
            "TrainerStub", (), {"optimizers": [optimizer], "lr_scheduler_configs": []}
        )()
        callback = NeuronClusterOptimizerSyncCallback()
        callback.on_fit_start(trainer, module)
        original_registered_ids = set(callback._ddp_registered_param_ids)
        module.load_state_dict(source.state_dict())
        callback.on_load_checkpoint(trainer, module, checkpoint)
        child_ids = {
            id(parameter)
            for parameter in module.cluster.cluster["neuron_2_1_1"].parameters()
        }
        self.assertEqual(callback._ddp_registered_param_ids, original_registered_ids)
        self.assertTrue(child_ids.issubset(callback._post_wrap_param_ids))
        with self.assertRaisesRegex(RuntimeError, "partial Neuron optimizer"):
            callback.on_train_start(trainer, module)
        optimizer.load_state_dict(saved_states[0])
        callback.on_train_start(trainer, module)
        self.assertTrue(child_ids.issubset(callback._post_wrap_param_ids))

    def test_reused_trainer_binds_checkpoint_to_current_optimizer_generation(
        self,
    ) -> None:
        module = _GrowingNeuronModule(self.build_config())
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                default_root_dir=directory,
            )
            module.trainer = trainer
            trainer.state.fn = TrainerFn.FITTING
            trainer.strategy.connect(module)
            trainer.strategy.setup_optimizers(trainer)
            previous_optimizer = trainer.optimizers[0]
            callback = NeuronClusterOptimizerSyncCallback()
            callback.setup(trainer, module, "fit")
            callback.on_fit_start(trainer, module)
            checkpoint = {
                "optimizer_states": [previous_optimizer.state_dict()],
                "lr_schedulers": [],
            }
            callback.on_save_checkpoint(trainer, module, checkpoint)
            callback.on_fit_end(trainer, module)

            callback.setup(trainer, module, "fit")
            callback.on_load_checkpoint(trainer, module, checkpoint)
            self.assertFalse(previous_optimizer._optimizer_load_state_dict_post_hooks)
            trainer.strategy.setup_optimizers(trainer)
            current_optimizer = trainer.optimizers[0]
            self.assertIsNot(current_optimizer, previous_optimizer)
            callback.on_fit_start(trainer, module)
            current_optimizer.load_state_dict(checkpoint["optimizer_states"][0])
            callback.on_train_start(trainer, module)
            self.assertFalse(current_optimizer._optimizer_load_state_dict_post_hooks)
            self.assertFalse(previous_optimizer._optimizer_load_state_dict_post_hooks)
            callback.on_fit_end(trainer, module)

    def build_config(self) -> NeuronClusterConfig:
        return NeuronClusterConfig(
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

    def build_loader(self) -> DataLoader:
        inputs = (
            torch.arange(24 * self.input_dim, dtype=torch.float32).reshape(
                24, self.input_dim
            )
            / 100
        )
        return DataLoader(TensorDataset(inputs), batch_size=3, shuffle=False)

    @staticmethod
    def clone_optimizer_state(state: dict[str, object]) -> dict[str, object]:
        return {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }

    @pytest.mark.training
    def test_grown_parameter_and_adam_state_restore_then_continue_updating(
        self,
    ) -> None:
        config = self.build_config()
        loader = self.build_loader()
        with tempfile.TemporaryDirectory() as directory:
            source_model = _GrowingNeuronModule(config)
            source_trainer = Trainer(
                accelerator="cpu",
                default_root_dir=directory,
                max_epochs=1,
                limit_train_batches=2,
                logger=False,
                enable_model_summary=False,
                enable_checkpointing=True,
                num_sanity_val_steps=0,
            )
            source_trainer.fit(source_model, train_dataloaders=loader)
            checkpoint = Path(directory) / "last.ckpt"
            source_trainer.save_checkpoint(checkpoint)
            source_child = source_model.cluster.cluster[
                "neuron_2_1_1"
            ].nucleus.model.weight
            source_value = source_child.detach().clone()
            source_state = self.clone_optimizer_state(
                source_trainer.optimizers[0].state[source_child]
            )

            resumed_model = _GrowingNeuronModule(config)
            with CheckpointContinuationLifecycle.admit(
                CheckpointContinuation(checkpoint),
                _SingleRunPlan(),
            ) as lifecycle:
                execution_options = lifecycle.bind_training_runs([_TargetTrainingRun()])
                model_validator = execution_options.strict_model_preloader
                assert model_validator is not None
                model_validator(resumed_model)
                probe = _GrownParameterContinuationProbe()
                resumed_trainer = Trainer(
                    accelerator="cpu",
                    default_root_dir=directory,
                    max_epochs=3,
                    limit_train_batches=3,
                    limit_val_batches=0,
                    logger=False,
                    enable_model_summary=False,
                    enable_checkpointing=False,
                    num_sanity_val_steps=0,
                    callbacks=[probe],
                )
                resumed_trainer.fit(
                    resumed_model,
                    train_dataloaders=loader,
                    ckpt_path=execution_options.checkpoint_path,
                    weights_only=True,
                )

        self.assertIn("neuron_2_1_1", resumed_model.cluster.cluster)
        torch.testing.assert_close(probe.restored_parameter, source_value)
        self.assertIsNotNone(probe.restored_state)
        for key, expected in source_state.items():
            actual = probe.restored_state[key]
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected)
            else:
                self.assertEqual(actual, expected)
        self.assertFalse(torch.equal(probe.updated_parameter, probe.restored_parameter))
        self.assertGreater(probe.updated_state["step"], probe.restored_state["step"])
        self.assertFalse(
            torch.equal(
                probe.updated_state["exp_avg"],
                probe.restored_state["exp_avg"],
            )
        )
        for value in probe.updated_state.values():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.isfinite(value).all())

    def test_checkpoint_without_named_optimizer_layout_is_rejected(self) -> None:
        config = self.build_config()
        model = _GrowingNeuronModule(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        callback = NeuronClusterOptimizerSyncCallback()
        trainer = type(
            "TrainerStub",
            (),
            {"state": None, "optimizers": [optimizer]},
        )()

        with self.assertRaisesRegex(RuntimeError, "retired optimizer layout"):
            callback.on_load_checkpoint(
                trainer,
                model,
                {"optimizer_states": [optimizer.state_dict()]},
            )

    @pytest.mark.training
    def test_named_layout_restores_custom_parameter_order_by_identity(self) -> None:
        config = self.build_config()
        loader = self.build_loader()
        with tempfile.TemporaryDirectory() as directory:
            source_model = _ReversedGrowingNeuronModule(config)
            source_trainer = Trainer(
                accelerator="cpu",
                default_root_dir=directory,
                max_epochs=1,
                limit_train_batches=2,
                logger=False,
                enable_model_summary=False,
                enable_checkpointing=True,
                num_sanity_val_steps=0,
            )
            source_trainer.fit(source_model, train_dataloaders=loader)
            optimizer = source_trainer.optimizers[0]
            expected_exp_avg = {}
            for index, (name, parameter) in enumerate(
                source_model.named_parameters(),
                start=1,
            ):
                sentinel = index / 100.0
                expected_exp_avg[name] = torch.full_like(parameter, sentinel)
                optimizer.state[parameter] = {
                    "step": torch.tensor(7.0),
                    "exp_avg": expected_exp_avg[name].clone(),
                    "exp_avg_sq": torch.full_like(parameter, sentinel + 1.0),
                }

            checkpoint = Path(directory) / "named-layout.ckpt"
            source_trainer.save_checkpoint(checkpoint)
            checkpoint_payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            self.assertIn(OPTIMIZER_LAYOUT_CHECKPOINT_KEY, checkpoint_payload)

            resumed_model = _ReversedGrowingNeuronModule(config)
            probe = _OptimizerStateIdentityProbe()
            resumed_trainer = Trainer(
                accelerator="cpu",
                default_root_dir=directory,
                max_epochs=2,
                limit_train_batches=3,
                limit_val_batches=0,
                logger=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                num_sanity_val_steps=0,
                callbacks=[probe],
            )
            resumed_trainer.fit(
                resumed_model,
                train_dataloaders=loader,
                ckpt_path=checkpoint,
            )

        self.assertEqual(probe.restored_group_count, 1)
        self.assertEqual(set(probe.restored_exp_avg), set(expected_exp_avg))
        for name, expected in expected_exp_avg.items():
            torch.testing.assert_close(probe.restored_exp_avg[name], expected)
        self.assertTrue(probe.updated_exp_avg)


if __name__ == "__main__":
    unittest.main()
