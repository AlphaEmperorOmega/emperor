from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from lightning import LightningDataModule
from lightning.pytorch.callbacks import Callback
from models.catalog import model_package
from torch.utils.data import DataLoader, TensorDataset

from model_runtime.runs import (
    CheckpointContinuation,
    FilesystemRunArtifacts,
    RunRequest,
    execute_runs,
    plan_runs,
)


class _InMemoryMnist(LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 0

    def setup(self, stage: str | None = None) -> None:
        generator = torch.Generator().manual_seed(7)
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


class _ContinuationProbe(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []
        self.start_epoch: int | None = None
        self.start_step: int | None = None
        self.optimizer_state_restored = False
        self.end_step: int | None = None

    def write_event(self, event: dict) -> None:
        self.events.append(dict(event))

    def on_train_start(self, trainer, pl_module) -> None:
        self.start_epoch = trainer.current_epoch
        self.start_step = trainer.global_step
        optimizer_state = trainer.optimizers[0].state_dict()["state"]
        self.optimizer_state_restored = bool(optimizer_state)

    def on_train_end(self, trainer, pl_module) -> None:
        self.end_step = trainer.global_step


def _linears_linear():
    package = model_package("linears/linear")
    if package is None:
        raise AssertionError("Expected the linears/linear Model Package.")
    return package


class CheckpointContinuationIntegrationTests(unittest.TestCase):
    def test_real_lightning_continues_complete_state_into_new_run_artifact(
        self,
    ) -> None:
        package = _linears_linear()
        common_overrides = {
            "BATCH_SIZE": 4,
            "HIDDEN_DIM": 4,
            "STACK_NUM_LAYERS": 1,
            "CALLBACK_CHECKPOINT_FLAG": True,
            "TRAINER_ENABLE_CHECKPOINTING": True,
            "TRAINER_BENCHMARK": False,
            "TRAINER_NUM_SANITY_VAL_STEPS": 0,
            "TRAINER_LOG_EVERY_N_STEPS": 1,
            "RUN_TEST_AFTER_FIT": False,
            "DATA_NUM_WORKERS": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(
                root=Path(tmp) / "logs",
                clock=lambda: datetime(2026, 7, 13, 12, 0, 0),
            )
            source_plan = plan_runs(
                package,
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    overrides={**common_overrides, "NUM_EPOCHS": 1},
                ),
            )
            source_probe = _ContinuationProbe()
            with patch.object(
                package.experiment_type,
                "_build_dataset",
                lambda _self, run: _InMemoryMnist(run.config.batch_size),
            ):
                source_result = execute_runs(
                    package,
                    source_plan,
                    artifacts=artifacts,
                    progress=source_probe,
                )[0]

                checkpoint = Path(source_result.log_dir) / "checkpoints" / "last.ckpt"
                self.assertTrue(checkpoint.is_file())
                source_bytes = checkpoint.read_bytes()
                source_checkpoint = torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertTrue(source_checkpoint["optimizer_states"])

                continuation_plan = plan_runs(
                    package,
                    RunRequest(
                        presets=("baseline",),
                        datasets=("Mnist",),
                        overrides={**common_overrides, "NUM_EPOCHS": 2},
                    ),
                )
                continuation_probe = _ContinuationProbe()
                continued_result = execute_runs(
                    package,
                    continuation_plan,
                    artifacts=artifacts,
                    progress=continuation_probe,
                    continuation=CheckpointContinuation(checkpoint),
                )[0]

            self.assertEqual(continuation_probe.start_epoch, 1)
            self.assertEqual(
                continuation_probe.start_step,
                source_checkpoint["global_step"],
            )
            self.assertTrue(continuation_probe.optimizer_state_restored)
            self.assertGreater(
                continuation_probe.end_step,
                continuation_probe.start_step,
            )
            self.assertNotEqual(continued_result.log_dir, source_result.log_dir)
            self.assertEqual(
                continued_result.payload["resumedFrom"],
                {
                    "checkpoint": "last.ckpt",
                    "epoch": 0,
                    "globalStep": source_checkpoint["global_step"],
                },
            )
            written_result = json.loads(
                Path(continued_result.log_dir, "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                written_result["resumedFrom"],
                continued_result.payload["resumedFrom"],
            )
            self.assertEqual(checkpoint.read_bytes(), source_bytes)


if __name__ == "__main__":
    unittest.main()
