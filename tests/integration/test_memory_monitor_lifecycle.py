from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from emperor.memory import (
    AttentionDynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryMonitorCallback,
    WeightedDynamicMemoryConfig,
)
from unit.test_memory import (
    make_memory_config,
    set_constant_output,
    set_scaled_identity,
    set_slot_multipliers,
)

BASE_SUFFIXES = (
    "memory/output_mean",
    "memory/output_var",
    "memory/output_l2_norm",
    "memory/contribution/delta_mean",
    "memory/contribution/delta_var",
    "memory/contribution/delta_norm",
    "memory/contribution/relative_delta_norm",
)
GATE_SUFFIXES = (
    "memory/gate/open_mean",
    "memory/gate/open_fraction",
)
SATURATION_SUFFIX = "memory/gate/saturation_fraction"


def configured_memory(
    config_type: type[
        GatedResidualDynamicMemoryConfig
        | WeightedDynamicMemoryConfig
        | ElementWiseWeightedDynamicMemoryConfig
        | AttentionDynamicMemoryConfig
    ],
) -> nn.Module:
    memory = make_memory_config(
        config_cls=config_type,
        input_dim=2,
        output_dim=2,
        num_memory_slots=2,
    ).build()
    if config_type is AttentionDynamicMemoryConfig:
        set_slot_multipliers(memory.memory_model, [1.0, 2.0])
        set_scaled_identity(memory.query_model, 1.0)
        set_scaled_identity(memory.key_model, 1.0)
        set_scaled_identity(memory.value_model, 1.0)
        set_scaled_identity(memory.output_model, 1.0)
        set_constant_output(memory.memory_gate_model, [-10.0, 10.0])
    elif config_type is GatedResidualDynamicMemoryConfig:
        set_scaled_identity(memory.memory_model, 2.0)
        set_constant_output(memory.memory_gate_model, [-10.0, 1.0])
    elif config_type is WeightedDynamicMemoryConfig:
        set_scaled_identity(memory.memory_model, 3.0)
        set_constant_output(memory.memory_weight_model, [0.0, 2.0])
    elif config_type is ElementWiseWeightedDynamicMemoryConfig:
        set_scaled_identity(memory.memory_model, 4.0)
        set_constant_output(memory.memory_weight_model, [-10.0, 10.0])
    else:
        raise AssertionError(f"Unexpected memory type: {type(memory).__name__}")
    return memory


class _MemoryTrainingModule(LightningModule):
    def __init__(
        self,
        *,
        variants: tuple[str, ...] = (
            "gated",
            "weighted",
            "element_wise",
            "attention",
        ),
        fail_after_forward: bool = False,
        learning_rate: float = 0.05,
    ) -> None:
        super().__init__()
        available = {
            "gated": configured_memory(GatedResidualDynamicMemoryConfig),
            "weighted": configured_memory(WeightedDynamicMemoryConfig),
            "element_wise": configured_memory(ElementWiseWeightedDynamicMemoryConfig),
            "attention": configured_memory(AttentionDynamicMemoryConfig),
        }
        self.memories = nn.ModuleDict({name: available[name] for name in variants})
        self.fail_after_forward = fail_after_forward
        self.learning_rate = learning_rate
        self.logged_calls: list[tuple[int, str, Tensor]] = []
        self.last_inputs: Tensor | None = None
        self.last_outputs: dict[str, Tensor] = {}

    def training_step(
        self,
        batch: tuple[Tensor],
        batch_idx: int,
    ) -> Tensor:
        (inputs,) = batch
        self.last_inputs = inputs.detach().clone()
        outputs = {name: memory(inputs) for name, memory in self.memories.items()}
        self.last_outputs = {
            name: output.detach().clone() for name, output in outputs.items()
        }
        if self.fail_after_forward:
            raise RuntimeError("deliberate memory lifecycle failure")
        return torch.stack(
            [output.square().mean() for output in outputs.values()]
        ).sum()

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


def loader(num_batches: int = 1) -> DataLoader:
    values = torch.tensor(
        [
            [1.0, -2.0],
            [0.5, 3.0],
            [-1.0, 0.25],
            [2.0, -0.5],
            [0.75, 1.25],
            [-0.25, -1.5],
        ]
    )
    return DataLoader(
        TensorDataset(values[: num_batches * 2]),
        batch_size=2,
        shuffle=False,
    )


def trainer(root: Path, callback: MemoryMonitorCallback) -> Trainer:
    return Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        deterministic=True,
        callbacks=[callback],
        logger=False,
        default_root_dir=root,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


class MemoryMonitorLifecycleTests(unittest.TestCase):
    def assert_logged_close(
        self,
        logged: dict[str, Tensor],
        name: str,
        expected: Tensor,
    ) -> None:
        self.assertIn(name, logged)
        torch.testing.assert_close(logged[name], expected)

    def test_real_trainer_logs_exact_metrics_for_every_variant_and_updates(
        self,
    ) -> None:
        torch.manual_seed(17)
        model = _MemoryTrainingModule()
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        initial_parameter = (
            model.memories["gated"].memory_model[0].model.weight_params.detach().clone()
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), callback)
            fit_trainer.fit(model, train_dataloaders=loader())

        logged = dict(fit_trainer.logged_metrics)
        self.assertIsNotNone(model.last_inputs)
        inputs = model.last_inputs
        for module_name, output in model.last_outputs.items():
            prefix = f"memories.{module_name}"
            delta = output - inputs
            self.assert_logged_close(
                logged,
                f"{prefix}/memory/output_mean",
                output.mean(),
            )
            self.assert_logged_close(
                logged,
                f"{prefix}/memory/output_var",
                output.var(unbiased=False),
            )
            self.assert_logged_close(
                logged,
                f"{prefix}/memory/output_l2_norm",
                output.norm(),
            )
            self.assert_logged_close(
                logged,
                f"{prefix}/memory/contribution/delta_mean",
                delta.mean(),
            )
            self.assert_logged_close(
                logged,
                f"{prefix}/memory/contribution/delta_var",
                delta.var(unbiased=False),
            )
            self.assert_logged_close(
                logged,
                f"{prefix}/memory/contribution/delta_norm",
                delta.norm(),
            )
            self.assert_logged_close(
                logged,
                f"{prefix}/memory/contribution/relative_delta_norm",
                delta.norm() / inputs.norm().clamp_min(1e-6),
            )

        gate_logits = {
            "gated": torch.tensor([-10.0, 1.0]),
            "weighted": torch.tensor([0.0, 2.0]),
            "element_wise": torch.tensor([-10.0, 10.0]),
            "attention": torch.tensor([-10.0, 10.0]),
        }
        for module_name, logits in gate_logits.items():
            prefix = f"memories.{module_name}/memory/gate"
            if module_name == "weighted":
                values = torch.softmax(logits, dim=-1)[-1]
                self.assertNotIn(f"{prefix}/saturation_fraction", logged)
            else:
                values = torch.sigmoid(logits)
                self.assert_logged_close(
                    logged,
                    f"{prefix}/saturation_fraction",
                    ((values < 0.01) | (values > 0.99)).float().mean(),
                )
            self.assert_logged_close(
                logged,
                f"{prefix}/open_mean",
                values.mean(),
            )
            self.assert_logged_close(
                logged,
                f"{prefix}/open_fraction",
                (values > 0.5).float().mean(),
            )

        self.assertFalse(
            torch.equal(
                model.memories["gated"].memory_model[0].model.weight_params.detach(),
                initial_parameter,
            )
        )
        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._latest_gate_logits, {})

    def test_real_trainer_uses_forward_global_step_cadence_without_duplicates(
        self,
    ) -> None:
        model = _MemoryTrainingModule(
            variants=("gated",),
            learning_rate=0.0,
        )
        callback = MemoryMonitorCallback(log_every_n_steps=2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), callback)
            fit_trainer.fit(model, train_dataloaders=loader(num_batches=3))

        self.assertEqual({step for step, _, _ in model.logged_calls}, {0, 2})
        expected_names = {
            f"memories.gated/{suffix}"
            for suffix in (*BASE_SUFFIXES, *GATE_SUFFIXES, SATURATION_SUFFIX)
        }
        for step in (0, 2):
            self.assertEqual(
                {
                    name
                    for logged_step, name, _ in model.logged_calls
                    if logged_step == step
                },
                expected_names,
            )

    def test_real_trainer_exception_removes_every_monitor_hook(self) -> None:
        model = _MemoryTrainingModule(
            variants=("gated",),
            fail_after_forward=True,
            learning_rate=0.0,
        )
        callback = MemoryMonitorCallback(log_every_n_steps=1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), callback)
            with self.assertRaisesRegex(
                RuntimeError,
                "^deliberate memory lifecycle failure$",
            ):
                fit_trainer.fit(model, train_dataloaders=loader())

        memory = model.memories["gated"]
        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._latest_gate_logits, {})
        self.assertEqual(memory._forward_pre_hooks, {})
        self.assertEqual(memory._forward_hooks, {})
        self.assertEqual(memory.memory_gate_model._forward_hooks, {})


if __name__ == "__main__":
    unittest.main()
