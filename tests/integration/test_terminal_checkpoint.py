from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pytest
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, Timer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from model_runtime.runs import (
    FilesystemRunArtifacts,
    RunRequest,
    execute_runs,
    plan_runs,
)
from model_runtime.runs.experiment import ExperimentBase
from models.catalog import model_package


class _TwoBatchDataModule(LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 0

    def setup(self, stage: str | None = None) -> None:
        generator = torch.Generator().manual_seed(11)
        features = torch.randn(4, 1, 28, 28, generator=generator)
        targets = torch.tensor([0, 1, 2, 3])
        self.dataset = TensorDataset(features, targets)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )


class _VectorDataModule(LightningDataModule):
    def setup(self, stage: str | None = None) -> None:
        self.dataset = TensorDataset(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                ]
            ),
            torch.tensor([1.0, 1.0, 2.0, 3.0]),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=2, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=2, shuffle=False)


class _FullStateModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        recurrent_layer = RecurrentLayerConfig(
            input_dim=2,
            output_dim=2,
            max_steps=100,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=LayerConfig(
                input_dim=2,
                output_dim=2,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=2,
                    output_dim=2,
                    bias_flag=False,
                ),
            ),
        ).build()
        self.recurrent_iteration_schedule = recurrent_layer.recurrent_iteration_schedule

    def training_step(self, batch, batch_index):
        self.recurrent_iteration_schedule.record_successful_forward()
        features, targets = batch
        predictions = self.linear(features).squeeze(-1)
        return nn.functional.mse_loss(predictions, targets)

    def validation_step(self, batch, batch_index) -> None:
        features, targets = batch
        predictions = self.linear(features).squeeze(-1)
        loss = nn.functional.mse_loss(predictions, targets)
        self.log("validation/loss", loss, on_epoch=True, batch_size=len(targets))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class _ContinuationStateProbe(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.restored_global_step: int | None = None
        self.restored_optimizer_step: int | None = None
        self.restored_scheduler_epoch: int | None = None
        self.restored_recurrent_progress: int | None = None
        self.updated_optimizer_step: int | None = None

    @staticmethod
    def _optimizer_step(trainer: Trainer) -> int:
        steps = [
            state["step"]
            for state in trainer.optimizers[0].state.values()
            if "step" in state
        ]
        return max(
            int(step.item()) if isinstance(step, torch.Tensor) else int(step)
            for step in steps
        )

    def on_train_start(self, trainer: Trainer, model: _FullStateModule) -> None:
        self.restored_global_step = trainer.global_step
        self.restored_optimizer_step = self._optimizer_step(trainer)
        self.restored_scheduler_epoch = trainer.lr_scheduler_configs[
            0
        ].scheduler.last_epoch
        self.restored_recurrent_progress = (
            model.recurrent_iteration_schedule.snapshot().forward_call_progress
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        model: _FullStateModule,
        outputs,
        batch,
        batch_index,
    ) -> None:
        self.updated_optimizer_step = self._optimizer_step(trainer)


def _linears_linear():
    package = model_package("linears/linear")
    if package is None:
        raise AssertionError("Expected the linears/linear Model Package.")
    return package


class TerminalCheckpointIntegrationTests(unittest.TestCase):
    @staticmethod
    def _fit_through_runtime(
        trainer: Trainer,
        model: LightningModule,
        data_module: LightningDataModule,
        checkpoint: ModelCheckpoint,
        *,
        checkpoint_path: Path | None = None,
    ) -> None:
        state = SimpleNamespace(
            options=SimpleNamespace(ckpt_path=checkpoint_path),
        )
        runtime = SimpleNamespace(
            model=model,
            dataset=data_module,
            runtime_config={"run_test_after_fit": False},
            terminal_checkpoint_callback=checkpoint,
        )
        ExperimentBase._fit_and_test_training(state, trainer, runtime)

    @pytest.mark.training
    def test_clean_mid_epoch_stop_rewrites_exact_terminal_last_checkpoint(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={
                    "BATCH_SIZE": 2,
                    "HIDDEN_DIM": 4,
                    "STACK_NUM_LAYERS": 1,
                    "NUM_EPOCHS": 3,
                    "CALLBACK_CHECKPOINT_FLAG": True,
                    "CALLBACK_EARLY_STOPPING_METRIC": "validation/loss",
                    "TRAINER_ENABLE_CHECKPOINTING": True,
                    "TRAINER_MAX_STEPS": 3,
                    "TRAINER_BENCHMARK": False,
                    "TRAINER_NUM_SANITY_VAL_STEPS": 0,
                    "TRAINER_LOG_EVERY_N_STEPS": 1,
                    "RUN_TEST_AFTER_FIT": False,
                    "DATA_NUM_WORKERS": 0,
                    "SEED": 11,
                },
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(
                root=Path(tmp) / "logs",
                clock=lambda: datetime(2026, 8, 13, 12, 0, 0),
            )
            with patch.object(
                ExperimentBase,
                "_build_dataset",
                lambda _self, run: _TwoBatchDataModule(run.config.batch_size),
            ):
                result = execute_runs(package, plan, artifacts=artifacts)[0]

            checkpoint_dir = Path(result.log_dir) / "checkpoints"
            terminal_path = checkpoint_dir / "last.ckpt"
            terminal = torch.load(
                terminal_path,
                map_location="cpu",
                weights_only=False,
            )
            best_paths = tuple(
                path
                for path in checkpoint_dir.glob("*.ckpt")
                if path.name != "last.ckpt"
            )

            self.assertEqual(len(best_paths), 1)
            best = torch.load(
                best_paths[0],
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(best["global_step"], 2)
            self.assertEqual(terminal["global_step"], 3)
            self.assertFalse((checkpoint_dir / "last-v1.ckpt").exists())

    @pytest.mark.training
    def test_terminal_full_state_resumes_optimizer_scheduler_and_recurrence(
        self,
    ) -> None:
        torch.manual_seed(23)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_checkpoint_dir = root / "source" / "checkpoints"
            source_checkpoint = ModelCheckpoint(
                dirpath=source_checkpoint_dir,
                monitor="validation/loss",
                save_top_k=1,
                save_last=True,
                mode="min",
            )
            source_timer = Timer()
            source_model = _FullStateModule()
            source_trainer = Trainer(
                accelerator="cpu",
                devices=1,
                default_root_dir=root / "source",
                max_epochs=3,
                max_steps=3,
                callbacks=[source_checkpoint, source_timer],
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                deterministic=True,
            )
            self._fit_through_runtime(
                source_trainer,
                source_model,
                _VectorDataModule(),
                source_checkpoint,
            )

            source_path = source_checkpoint_dir / "last.ckpt"
            source_bytes = source_path.read_bytes()
            source_payload = torch.load(
                source_path,
                map_location="cpu",
                weights_only=False,
            )
            recurrent_key = (
                "recurrent_iteration_schedule.forward_call_progress"
            )

            self.assertEqual(source_trainer.global_step, 3)
            self.assertEqual(source_payload["global_step"], 3)
            self.assertTrue(source_payload["optimizer_states"])
            self.assertTrue(source_payload["lr_schedulers"])
            self.assertIn("loops", source_payload)
            self.assertIn("callbacks", source_payload)
            self.assertIn(source_timer.state_key, source_payload["callbacks"])
            self.assertEqual(
                source_payload["state_dict"][recurrent_key].item(),
                3,
            )
            self.assertEqual(
                source_payload["callbacks"][source_checkpoint.state_key][
                    "last_model_path"
                ],
                str(source_path),
            )

            resumed_checkpoint_dir = root / "resumed" / "checkpoints"
            resumed_checkpoint = ModelCheckpoint(
                dirpath=resumed_checkpoint_dir,
                monitor="validation/loss",
                save_top_k=1,
                save_last=True,
                mode="min",
            )
            resumed_timer = Timer()
            probe = _ContinuationStateProbe()
            resumed_model = _FullStateModule()
            resumed_trainer = Trainer(
                accelerator="cpu",
                devices=1,
                default_root_dir=root / "resumed",
                max_epochs=3,
                max_steps=4,
                callbacks=[resumed_checkpoint, resumed_timer, probe],
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                deterministic=True,
            )
            self._fit_through_runtime(
                resumed_trainer,
                resumed_model,
                _VectorDataModule(),
                resumed_checkpoint,
                checkpoint_path=source_path,
            )

            resumed_path = resumed_checkpoint_dir / "last.ckpt"
            resumed_payload = torch.load(
                resumed_path,
                map_location="cpu",
                weights_only=False,
            )
            source_optimizer_step = max(
                int(state["step"].item())
                for state in source_payload["optimizer_states"][0]["state"].values()
            )

            self.assertEqual(probe.restored_global_step, 3)
            self.assertEqual(probe.restored_optimizer_step, source_optimizer_step)
            self.assertEqual(
                probe.restored_scheduler_epoch,
                source_payload["lr_schedulers"][0]["last_epoch"],
            )
            self.assertEqual(probe.restored_recurrent_progress, 3)
            self.assertGreater(
                probe.updated_optimizer_step,
                probe.restored_optimizer_step,
            )
            self.assertEqual(resumed_trainer.global_step, 4)
            self.assertEqual(resumed_payload["global_step"], 4)
            self.assertEqual(
                resumed_payload["state_dict"][recurrent_key].item(),
                4,
            )
            self.assertNotEqual(resumed_path, source_path)
            self.assertEqual(source_path.read_bytes(), source_bytes)


if __name__ == "__main__":
    unittest.main()
