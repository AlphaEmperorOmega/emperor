import unittest

import torch
from emperor.memory.config import (
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from emperor.memory.core._validator import DynamicMemoryValidator
from emperor.memory.core.base import DynamicMemoryAbstract
from emperor.memory.core.monitor import MemoryMonitorCallback
from torch import nn

from support.monitor import orchestration_calls
from unit.test_memory import (
    MEMORY_CASES,
    ConstantLastDimModule,
    IdentityModule,
    make_memory_config,
)


class FakeTrainer:
    def __init__(self, global_step: int = 0):
        self.global_step = global_step


class FakeLightningModule(nn.Module):
    def __init__(self, memory: DynamicMemoryAbstract, global_step: int = 0):
        super().__init__()
        self.memory = memory
        self.global_step = global_step
        self.logged_scalars = []

    def log(self, name, value, *args, **kwargs):
        self.logged_scalars.append((name, value))


class MultiMemoryLightningModule(nn.Module):
    def __init__(
        self,
        first_memory: DynamicMemoryAbstract,
        second_memory: DynamicMemoryAbstract,
        global_step: int = 0,
    ):
        super().__init__()
        self.first_memory = first_memory
        self.second_memory = second_memory
        self.global_step = global_step
        self.logged_scalars = []

    def log(self, name, value, *args, **kwargs):
        self.logged_scalars.append((name, value))


class GateFreeMemory(DynamicMemoryAbstract):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        DynamicMemoryValidator.validate_forward_inputs(logits, self.memory_dim)
        return logits + 1.0


def build_memory(config_cls=GatedResidualDynamicMemoryConfig) -> DynamicMemoryAbstract:
    return make_memory_config(
        config_cls=config_cls,
        input_dim=4,
        output_dim=4,
    ).build()


class TestMemoryMonitorCallback(unittest.TestCase):
    def test_tracking_orchestration_lists_each_tracked_fact(self):
        orchestration = (
            MemoryMonitorCallback._MemoryMonitorCallback__track_memory_diagnostics
        )

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_output_mean",
                "__track_output_variance",
                "__track_output_l2_norm",
                "__track_contribution_delta_mean",
                "__track_contribution_delta_variance",
                "__track_contribution_delta_norm",
                "__track_contribution_relative_delta_norm",
                "__track_gate_open_mean",
                "__track_gate_open_fraction",
                "__track_gate_saturation_fraction",
            ),
        )

    def assert_logged_close(
        self,
        scalars: dict[str, torch.Tensor],
        name: str,
        expected: torch.Tensor,
    ) -> None:
        self.assertIn(name, scalars)
        torch.testing.assert_close(scalars[name], expected)

    def test_init_rejects_invalid_log_interval_types(self):
        for bad in (True, 1.5, "1"):
            with self.subTest(log_every_n_steps=bad):
                with self.assertRaises(TypeError):
                    MemoryMonitorCallback(log_every_n_steps=bad)

    def test_init_rejects_non_positive_log_interval(self):
        for bad in (0, -1):
            with self.subTest(log_every_n_steps=bad):
                with self.assertRaises(ValueError):
                    MemoryMonitorCallback(log_every_n_steps=bad)

    def test_forward_hook_respects_trainer_global_step(self):
        skipped = FakeLightningModule(build_memory(), global_step=10)
        skipped_callback = MemoryMonitorCallback(log_every_n_steps=10)
        skipped_trainer = FakeTrainer(global_step=5)
        skipped_callback.on_fit_start(skipped_trainer, skipped)
        skipped.memory(torch.randn(2, 4))
        self.assertEqual(skipped.logged_scalars, [])
        skipped_callback.on_fit_end(skipped_trainer, skipped)

        logged = FakeLightningModule(build_memory(), global_step=5)
        logged_callback = MemoryMonitorCallback(log_every_n_steps=10)
        logged_trainer = FakeTrainer(global_step=10)
        logged_callback.on_fit_start(logged_trainer, logged)
        logged.memory(torch.randn(2, 4))

        keys = {name for name, _ in logged.logged_scalars}
        self.assertIn("memory/memory/output_mean", keys)
        self.assertIn("memory/memory/contribution/relative_delta_norm", keys)
        logged_callback.on_fit_end(logged_trainer, logged)

    def test_logs_finite_metrics_for_all_memory_variants(self):
        for config_cls, _ in MEMORY_CASES:
            with self.subTest(config_cls=config_cls.__name__):
                module = FakeLightningModule(build_memory(config_cls))
                callback = MemoryMonitorCallback(log_every_n_steps=1)
                trainer = FakeTrainer(global_step=0)

                callback.on_fit_start(trainer, module)
                module.memory(torch.randn(3, 4))

                scalars = dict(module.logged_scalars)
                for name in (
                    "memory/memory/output_mean",
                    "memory/memory/output_var",
                    "memory/memory/output_l2_norm",
                    "memory/memory/contribution/delta_mean",
                    "memory/memory/contribution/delta_var",
                    "memory/memory/contribution/delta_norm",
                    "memory/memory/contribution/relative_delta_norm",
                    "memory/memory/gate/open_mean",
                    "memory/memory/gate/open_fraction",
                ):
                    self.assertIn(name, scalars)
                    self.assertTrue(
                        torch.isfinite(scalars[name]).all(),
                        f"{name} not finite",
                    )
                callback.on_fit_end(trainer, module)

    def test_hook_cleanup_removes_forward_hooks_and_state(self):
        module = FakeLightningModule(build_memory())
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        self.assertTrue(callback._hooks)

        callback.on_fit_end(trainer, module)

        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._latest_gate_logits, {})
        module.logged_scalars.clear()
        module.memory(torch.randn(2, 4))
        self.assertEqual(module.logged_scalars, [])

    def test_repeated_fit_start_does_not_duplicate_hooks(self):
        module = FakeLightningModule(build_memory())
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_fit_start(trainer, module)
        module.memory(torch.randn(2, 4))

        logged_names = [name for name, _ in module.logged_scalars]
        self.assertEqual(len(logged_names), len(set(logged_names)))
        callback.on_fit_end(trainer, module)

    def test_logs_exact_output_and_contribution_formulas(self):
        memory = GateFreeMemory(make_memory_config(input_dim=4, output_dim=4))
        module = FakeLightningModule(memory)
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)
        logits = torch.tensor([[1.0, -2.0, 3.0, 0.0], [0.5, 2.0, -1.0, 4.0]])
        output = logits + 1.0
        delta = output - logits

        callback.on_fit_start(trainer, module)
        module.memory(logits)

        scalars = dict(module.logged_scalars)
        prefix = "memory/memory"
        self.assert_logged_close(scalars, f"{prefix}/output_mean", output.mean())
        self.assert_logged_close(
            scalars,
            f"{prefix}/output_var",
            output.var(unbiased=False),
        )
        self.assert_logged_close(scalars, f"{prefix}/output_l2_norm", output.norm())
        self.assert_logged_close(
            scalars,
            f"{prefix}/contribution/delta_mean",
            delta.mean(),
        )
        self.assert_logged_close(
            scalars,
            f"{prefix}/contribution/delta_var",
            delta.var(unbiased=False),
        )
        self.assert_logged_close(
            scalars,
            f"{prefix}/contribution/delta_norm",
            delta.norm(),
        )
        self.assert_logged_close(
            scalars,
            f"{prefix}/contribution/relative_delta_norm",
            delta.norm() / logits.norm().clamp_min(1e-6),
        )
        callback.on_fit_end(trainer, module)

    def test_logs_exact_sigmoid_gate_metrics(self):
        memory = build_memory(GatedResidualDynamicMemoryConfig)
        gate_logits = torch.tensor([-10.0, 0.0, 10.0, 1.0])
        memory.memory_model = IdentityModule()
        memory.memory_gate_model = ConstantLastDimModule(gate_logits)
        module = FakeLightningModule(memory)
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        module.memory(torch.ones(2, 4))

        gate = torch.sigmoid(gate_logits)
        scalars = dict(module.logged_scalars)
        prefix = "memory/memory/gate"
        self.assert_logged_close(scalars, f"{prefix}/open_mean", gate.mean())
        self.assert_logged_close(
            scalars,
            f"{prefix}/open_fraction",
            (gate > 0.5).float().mean(),
        )
        self.assert_logged_close(
            scalars,
            f"{prefix}/saturation_fraction",
            ((gate < 0.01) | (gate > 0.99)).float().mean(),
        )
        self.assertEqual(callback._latest_gate_logits, {})
        callback.on_fit_end(trainer, module)

    def test_logs_exact_weighted_memory_share_metrics(self):
        memory = build_memory(WeightedDynamicMemoryConfig)
        weight_logits = torch.tensor([0.0, 2.0])
        memory.memory_model = IdentityModule()
        memory.memory_weight_model = ConstantLastDimModule(weight_logits)
        module = FakeLightningModule(memory)
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        module.memory(torch.ones(2, 4))

        memory_share = torch.softmax(weight_logits, dim=-1)[-1]
        scalars = dict(module.logged_scalars)
        prefix = "memory/memory/gate"
        self.assert_logged_close(scalars, f"{prefix}/open_mean", memory_share)
        self.assert_logged_close(
            scalars,
            f"{prefix}/open_fraction",
            (memory_share > 0.5).float(),
        )
        self.assertNotIn(f"{prefix}/saturation_fraction", scalars)
        callback.on_fit_end(trainer, module)

    def test_logs_distinct_prefixes_for_multiple_memory_modules(self):
        module = MultiMemoryLightningModule(
            GateFreeMemory(make_memory_config(input_dim=4, output_dim=4)),
            GateFreeMemory(make_memory_config(input_dim=4, output_dim=4)),
        )
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        module.first_memory(torch.zeros(1, 4))
        module.second_memory(torch.zeros(1, 4))

        keys = {name for name, _ in module.logged_scalars}
        self.assertIn("first_memory/memory/output_mean", keys)
        self.assertIn("second_memory/memory/output_mean", keys)
        callback.on_fit_end(trainer, module)

    def test_logs_output_and_contribution_without_gate_submodule(self):
        memory = GateFreeMemory(make_memory_config(input_dim=4, output_dim=4))
        module = FakeLightningModule(memory)
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        module.memory(torch.zeros(2, 4))

        keys = {name for name, _ in module.logged_scalars}
        self.assertIn("memory/memory/output_mean", keys)
        self.assertIn("memory/memory/contribution/delta_norm", keys)
        self.assertFalse(any("/gate/" in key for key in keys))
        for name, value in module.logged_scalars:
            self.assertTrue(torch.isfinite(value).all(), f"{name} not finite")
        callback.on_fit_end(trainer, module)


if __name__ == "__main__":
    unittest.main()
