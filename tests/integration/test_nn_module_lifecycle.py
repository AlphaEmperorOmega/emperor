from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from emperor.nn import Module


class _TinyRegressionModule(Module):
    def __init__(self) -> None:
        super().__init__(plotProgress=False)
        self.net = torch.nn.Linear(1, 1, bias=False)
        self.auxiliary_scale = torch.nn.Parameter(torch.tensor(0.5))
        self.lr = 0.1
        with torch.no_grad():
            self.net.weight.zero_()

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return self.net(inputs), self.auxiliary_scale.square()

    def loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return F.mse_loss(predictions, targets)


class ModuleLightningLifecycleTests(unittest.TestCase):
    def test_base_training_step_runs_real_automatic_optimization(self) -> None:
        dataset = TensorDataset(
            torch.tensor([[1.0]]),
            torch.tensor([[2.0]]),
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        module = _TinyRegressionModule()

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                default_root_dir=Path(temporary_directory),
                deterministic=True,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                logger=False,
                max_epochs=1,
                num_sanity_val_steps=0,
            )
            trainer.fit(
                module,
                train_dataloaders=loader,
                val_dataloaders=loader,
            )
            validation_results = trainer.validate(module, dataloaders=loader)
            test_results = trainer.test(module, dataloaders=loader)

        self.assertEqual(trainer.global_step, 1)
        torch.testing.assert_close(
            module.net.weight.detach(),
            torch.tensor([[0.4]]),
        )
        torch.testing.assert_close(
            module.auxiliary_scale.detach(),
            torch.tensor(0.4),
        )
        self.assertEqual(validation_results, [{}])
        self.assertEqual(test_results, [{}])


if __name__ == "__main__":
    unittest.main()
