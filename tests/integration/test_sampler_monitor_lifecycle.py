from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from emperor.sampler import SamplerConfig, SamplerModel, SamplerMonitorCallback

USAGE_SUFFIXES = (
    "active_experts",
    "usage_entropy",
    "usage_coefficient_of_variation",
    "max_usage_fraction",
    "min_usage_fraction",
    "max_probability_mass",
    "min_probability_mass",
)


def sampler_config(
    *,
    top_k: int,
    num_experts: int = 4,
    normalize: bool,
    auxiliary_weight: float,
) -> SamplerConfig:
    return SamplerConfig(
        top_k=top_k,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=normalize,
        noisy_topk_flag=False,
        num_experts=num_experts,
        coefficient_of_variation_loss_weight=(
            0.0 if top_k == num_experts else auxiliary_weight
        ),
        switch_loss_weight=0.0 if top_k == num_experts else auxiliary_weight,
        zero_centred_loss_weight=0.0 if top_k == num_experts else auxiliary_weight,
        mutual_information_loss_weight=0.0,
        router_config=None,
    )


class _SamplerTrainingModule(LightningModule):
    def __init__(
        self,
        *,
        fail_after_forward: bool = False,
        learning_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.samplers = nn.ModuleDict(
            {
                "sparse": SamplerModel(
                    sampler_config(
                        top_k=1,
                        normalize=False,
                        auxiliary_weight=0.1,
                    )
                ),
                "topk": SamplerModel(
                    sampler_config(
                        top_k=2,
                        normalize=True,
                        auxiliary_weight=0.1,
                    )
                ),
                "full": SamplerModel(
                    sampler_config(
                        top_k=4,
                        normalize=True,
                        auxiliary_weight=0.0,
                    )
                ),
                "single": SamplerModel(
                    sampler_config(
                        top_k=1,
                        num_experts=1,
                        normalize=False,
                        auxiliary_weight=0.0,
                    )
                ),
            }
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.fail_after_forward = fail_after_forward
        self.learning_rate = learning_rate
        self.logged_calls: list[tuple[int, str, Tensor]] = []
        self.last_outputs: dict[
            str,
            tuple[Tensor, Tensor | None, Tensor | None, Tensor],
        ] = {}

    def training_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        (base_logits,) = batch
        logits = base_logits * self.logit_scale
        skip_mask = logits.new_ones((logits.shape[0], 1))
        outputs = {
            "sparse": self.samplers["sparse"].sample_probabilities_and_indices(
                logits,
                skip_mask,
            ),
            "topk": self.samplers["topk"].sample_probabilities_and_indices(
                logits,
                skip_mask,
            ),
            "full": self.samplers["full"].sample_probabilities_and_indices(
                logits,
                skip_mask,
            ),
            "single": self.samplers["single"].sample_probabilities_and_indices(
                logits[:, :1],
                skip_mask,
            ),
        }
        self.last_outputs = {
            name: (
                probabilities.detach().clone(),
                indices.detach().clone() if indices is not None else None,
                updated_skip_mask.detach().clone()
                if updated_skip_mask is not None
                else None,
                auxiliary_loss.detach().clone(),
            )
            for name, (
                probabilities,
                indices,
                updated_skip_mask,
                auxiliary_loss,
            ) in outputs.items()
        }
        if self.fail_after_forward:
            raise RuntimeError("deliberate sampler lifecycle failure")

        losses = []
        for probabilities, _, _, auxiliary_loss in outputs.values():
            losses.append(probabilities.square().mean() + auxiliary_loss)
        return torch.stack(losses).sum()

    def log(
        self,
        name: str,
        value: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if torch.is_tensor(value):
            self.logged_calls.append(
                (int(self.global_step), name, value.detach().clone())
            )
        super().log(name, value, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


class _HistoryObserver(Callback):
    def __init__(self, monitor: SamplerMonitorCallback) -> None:
        super().__init__()
        self.monitor = monitor
        self.lengths: list[int] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        history = self.monitor._usage_history["samplers.topk"]
        self.lengths.append(len(history))


def loader(num_batches: int = 1) -> DataLoader:
    logits = torch.tensor(
        [
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, -1.0, -2.0],
            [-2.0, -1.0, 1.0, 4.0],
            [2.0, 0.0, 3.0, 1.0],
            [1.0, 3.0, 0.0, 2.0],
        ]
    )
    return DataLoader(
        TensorDataset(logits[: num_batches * 2]),
        batch_size=2,
        shuffle=False,
    )


def trainer(
    root: Path,
    callbacks: list[Callback],
    *,
    logger: bool | TensorBoardLogger = False,
) -> Trainer:
    return Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        deterministic=True,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=root,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def expected_usage(
    probabilities: Tensor,
    indices: Tensor | None,
    num_experts: int,
) -> tuple[Tensor, Tensor]:
    if indices is None:
        flat_probabilities = probabilities.reshape(-1, num_experts)
        return (
            (flat_probabilities > 0).sum(dim=0).float(),
            flat_probabilities.sum(dim=0),
        )
    flat_indices = indices.reshape(-1)
    counts = torch.bincount(flat_indices, minlength=num_experts).float()
    mass = probabilities.new_zeros(num_experts)
    mass.scatter_add_(0, flat_indices, probabilities.reshape(-1))
    return counts, mass


class SamplerMonitorLifecycleTests(unittest.TestCase):
    def assert_usage_metrics(
        self,
        logged: dict[str, Tensor],
        prefix: str,
        scope: str,
        counts: Tensor,
        mass: Tensor,
    ) -> None:
        usage_fraction = counts / counts.sum().clamp_min(1.0)
        mass_fraction = mass / mass.sum().clamp_min(1e-6)
        expected = {
            "active_experts": (counts > 0).sum().float(),
            "usage_entropy": -(
                usage_fraction.clamp_min(1e-6).log() * usage_fraction
            ).sum(),
            "usage_coefficient_of_variation": (
                counts.std(unbiased=False) / counts.mean().clamp_min(1e-6)
            ),
            "max_usage_fraction": usage_fraction.max(),
            "min_usage_fraction": usage_fraction.min(),
            "max_probability_mass": mass_fraction.max(),
            "min_probability_mass": mass_fraction.min(),
        }
        for suffix, expected_value in expected.items():
            metric_name = f"{prefix}/{scope}/{suffix}"
            self.assertIn(metric_name, logged)
            torch.testing.assert_close(logged[metric_name], expected_value)
            self.assertTrue(torch.isfinite(logged[metric_name]), metric_name)

    def test_real_trainer_logs_exact_metrics_for_every_variant_and_updates(
        self,
    ) -> None:
        torch.manual_seed(17)
        model = _SamplerTrainingModule()
        monitor = SamplerMonitorCallback(log_every_n_steps=1)
        initial_scale = model.logit_scale.detach().clone()

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), [monitor])
            fit_trainer.fit(model, train_dataloaders=loader())

        logged = dict(fit_trainer.logged_metrics)
        for name, sampler in model.samplers.items():
            probabilities, indices, updated_skip_mask, auxiliary_loss = (
                model.last_outputs[name]
            )
            counts, mass = expected_usage(
                probabilities,
                indices,
                sampler.num_experts,
            )
            prefix = f"samplers.{name}"
            self.assert_usage_metrics(logged, prefix, "batch", counts, mass)
            self.assert_usage_metrics(logged, prefix, "cumulative", counts, mass)
            torch.testing.assert_close(
                logged[f"{prefix}/capacity/retention_fraction"],
                updated_skip_mask.float().mean(),
            )
            torch.testing.assert_close(
                logged[f"{prefix}/capacity/drop_fraction"],
                1.0 - updated_skip_mask.float().mean(),
            )
            torch.testing.assert_close(
                logged[f"{prefix}/loss/auxiliary_loss"],
                auxiliary_loss.float().mean(),
            )
            self.assertNotIn("_usage_tracker", sampler._modules)

        self.assertFalse(torch.equal(model.logit_scale.detach(), initial_scale))
        self.assertIsNone(monitor._tracker_manager)
        self.assertEqual(monitor._sampler_modules, [])
        self.assertEqual(monitor._usage_history, {})
        self.assertEqual(monitor._mass_history, {})

    def test_real_trainer_uses_cadence_and_bounded_history(self) -> None:
        model = _SamplerTrainingModule(learning_rate=0.0)
        monitor = SamplerMonitorCallback(log_every_n_steps=2, history_size=1)
        observer = _HistoryObserver(monitor)

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="sampler-cadence",
            )
            fit_trainer = trainer(
                Path(temporary_directory),
                [monitor, observer],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=loader(num_batches=3))

        self.assertEqual({step for step, _, _ in model.logged_calls}, {1, 3})
        self.assertEqual(observer.lengths, [1, 1, 1])

    def test_real_tensorboard_logger_receives_histograms_and_heatmaps(self) -> None:
        model = _SamplerTrainingModule(learning_rate=0.0)
        monitor = SamplerMonitorCallback(log_every_n_steps=1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="sampler",
            )
            fit_trainer = trainer(
                Path(temporary_directory),
                [monitor],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=loader())
            tensorboard_logger.experiment.flush()

            events = EventAccumulator(tensorboard_logger.log_dir)
            events.Reload()
            tags = events.Tags()

        prefix = "samplers.topk"
        usage_histogram = f"{prefix}/histogram/usage_fraction"
        mass_histogram = f"{prefix}/histogram/probability_mass"
        usage_heatmap = f"{prefix}/heatmap/usage_fraction"
        mass_heatmap = f"{prefix}/heatmap/probability_mass"
        self.assertIn(usage_histogram, tags["histograms"])
        self.assertIn(mass_histogram, tags["histograms"])
        self.assertIn(usage_heatmap, tags["images"])
        self.assertIn(mass_heatmap, tags["images"])
        self.assertEqual(events.Histograms(usage_histogram)[0].step, 1)
        self.assertEqual(events.Histograms(mass_histogram)[0].step, 1)
        self.assertEqual(events.Images(usage_heatmap)[0].step, 1)
        self.assertEqual(events.Images(mass_heatmap)[0].step, 1)

    def test_real_trainer_exception_invokes_monitor_cleanup(self) -> None:
        model = _SamplerTrainingModule(
            fail_after_forward=True,
            learning_rate=0.0,
        )
        monitor = SamplerMonitorCallback(log_every_n_steps=1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), [monitor])
            with self.assertRaisesRegex(
                RuntimeError,
                "^deliberate sampler lifecycle failure$",
            ):
                fit_trainer.fit(model, train_dataloaders=loader())

        for sampler in model.samplers.values():
            self.assertNotIn("_usage_tracker", sampler._modules)
        self.assertIsNone(monitor._tracker_manager)
        self.assertEqual(monitor._sampler_modules, [])
        self.assertEqual(monitor._usage_history, {})
        self.assertEqual(monitor._mass_history, {})


if __name__ == "__main__":
    unittest.main()
