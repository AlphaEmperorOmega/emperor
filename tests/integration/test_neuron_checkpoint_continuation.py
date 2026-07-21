from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, TensorDataset

from emperor.neuron import (
    NeuronClusterConfig,
    NeuronClusterOptimizerSyncCallback,
)
from model_runtime.runs.checkpoints import (
    CheckpointContinuation,
    _LoadedCheckpointContinuation,
    validate_model_state,
)
from unit.test_neuron import NeuronTestCase

OPTIMIZER_LAYOUT_CHECKPOINT_KEY = "emperor_neuron_optimizer_layout"


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

            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            resumed_model = _GrowingNeuronModule(config)
            validate_model_state(
                _LoadedCheckpointContinuation(
                    request=CheckpointContinuation(checkpoint),
                    state_dict=checkpoint_payload["state_dict"],
                    epoch=int(checkpoint_payload["epoch"]),
                    global_step=int(checkpoint_payload["global_step"]),
                ),
                resumed_model,
            )
            probe = _GrownParameterContinuationProbe()
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

    def test_legacy_appended_optimizer_group_restores_each_named_state(self) -> None:
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
            current_checkpoint = Path(directory) / "current.ckpt"
            source_trainer.save_checkpoint(current_checkpoint)
            checkpoint_payload = torch.load(
                current_checkpoint,
                map_location="cpu",
                weights_only=True,
            )

            optimizer = source_trainer.optimizers[0]
            self.assertEqual(len(optimizer.param_groups), 1)
            optimizer_parameters = optimizer.param_groups[0]["params"]
            serialized_group = checkpoint_payload["optimizer_states"][0][
                "param_groups"
            ][0]
            serialized_parameter_ids = serialized_group["params"]
            self.assertEqual(len(optimizer_parameters), len(serialized_parameter_ids))

            parameter_names = {
                id(parameter): name
                for name, parameter in source_model.named_parameters()
            }
            serialized_id_by_name = {
                parameter_names[id(parameter)]: serialized_id
                for parameter, serialized_id in zip(
                    optimizer_parameters,
                    serialized_parameter_ids,
                    strict=True,
                )
            }
            grown_prefix = "cluster.cluster.neuron_2_1_1."
            base_parameter_ids = [
                serialized_id_by_name[name]
                for name in serialized_id_by_name
                if not name.startswith(grown_prefix)
            ]
            grown_parameter_ids = [
                serialized_id_by_name[name]
                for name in serialized_id_by_name
                if name.startswith(grown_prefix)
            ]
            self.assertTrue(base_parameter_ids)
            self.assertTrue(grown_parameter_ids)

            optimizer_state = checkpoint_payload["optimizer_states"][0]
            base_group = copy.deepcopy(serialized_group)
            base_group["params"] = base_parameter_ids
            grown_group = copy.deepcopy(serialized_group)
            grown_group["params"] = grown_parameter_ids
            optimizer_state["param_groups"] = [base_group, grown_group]
            checkpoint_payload.pop(OPTIMIZER_LAYOUT_CHECKPOINT_KEY)

            expected_exp_avg: dict[str, torch.Tensor] = {}
            parameters_by_name = dict(source_model.named_parameters())
            for index, (name, serialized_id) in enumerate(
                serialized_id_by_name.items(),
                start=1,
            ):
                parameter = parameters_by_name[name]
                sentinel = index / 100.0
                expected_exp_avg[name] = torch.full_like(parameter, sentinel)
                optimizer_state["state"][serialized_id] = {
                    "step": torch.tensor(7.0),
                    "exp_avg": expected_exp_avg[name].clone(),
                    "exp_avg_sq": torch.full_like(parameter, sentinel + 1.0),
                }

            legacy_checkpoint = Path(directory) / "legacy.ckpt"
            torch.save(checkpoint_payload, legacy_checkpoint)

            resumed_model = _GrowingNeuronModule(config)
            validate_model_state(
                _LoadedCheckpointContinuation(
                    request=CheckpointContinuation(legacy_checkpoint),
                    state_dict=checkpoint_payload["state_dict"],
                    epoch=int(checkpoint_payload["epoch"]),
                    global_step=int(checkpoint_payload["global_step"]),
                ),
                resumed_model,
            )
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
                ckpt_path=legacy_checkpoint,
            )

            named_continuation = Path(directory) / "legacy-resaved-named.ckpt"
            resumed_trainer.save_checkpoint(named_continuation)
            twice_resumed_model = _GrowingNeuronModule(config)
            twice_resumed_probe = _OptimizerStateIdentityProbe()
            twice_resumed_trainer = Trainer(
                accelerator="cpu",
                default_root_dir=directory,
                max_epochs=3,
                limit_train_batches=4,
                limit_val_batches=0,
                logger=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                num_sanity_val_steps=0,
                callbacks=[twice_resumed_probe],
            )
            twice_resumed_trainer.fit(
                twice_resumed_model,
                train_dataloaders=loader,
                ckpt_path=named_continuation,
            )

        self.assertEqual(probe.restored_group_count, 2)
        self.assertEqual(set(probe.restored_exp_avg), set(expected_exp_avg))
        for name, expected in expected_exp_avg.items():
            torch.testing.assert_close(probe.restored_exp_avg[name], expected)
        grown_weight_name = "cluster.cluster.neuron_2_1_1.nucleus.model.weight"
        self.assertFalse(
            torch.equal(
                probe.updated_exp_avg[grown_weight_name],
                probe.restored_exp_avg[grown_weight_name],
            )
        )
        self.assertEqual(twice_resumed_probe.restored_group_count, 2)
        self.assertEqual(
            set(twice_resumed_probe.restored_exp_avg),
            set(probe.updated_exp_avg),
        )
        for name, expected in probe.updated_exp_avg.items():
            torch.testing.assert_close(
                twice_resumed_probe.restored_exp_avg[name],
                expected,
            )

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
