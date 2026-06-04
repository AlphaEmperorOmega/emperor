import torch
import unittest

from torch import nn

from emperor.linears.core.config import LinearLayerConfig
from emperor.linears.core.layers import LinearLayer
from emperor.linears.core.monitor import LinearMonitorCallback


class FakeTrainer:
    def __init__(self, global_step: int = 0):
        self.global_step = global_step


class FakeLightningModule(nn.Module):
    def __init__(self, linear: LinearLayer, global_step: int = 0):
        super().__init__()
        self.linear = linear
        self.global_step = global_step
        self.logged_scalars = []

    def log(self, name, value, *args, **kwargs):
        self.logged_scalars.append((name, value))


def build_module(
    input_dim: int = 4,
    output_dim: int = 4,
    bias_flag: bool = True,
    global_step: int = 0,
) -> FakeLightningModule:
    layer = LinearLayer(
        LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
        )
    )
    return FakeLightningModule(layer, global_step=global_step)


class TestLinearMonitorCallback(unittest.TestCase):
    def test_init_rejects_non_positive_log_interval(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                LinearMonitorCallback(log_every_n_steps=bad)

    def test_gradient_stats_are_finite_with_zero_weights(self):
        module = build_module(global_step=0)
        module.linear.weight_params.data.zero_()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        module.linear(torch.randn(3, 4)).sum().backward()
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        self.assertTrue(module.logged_scalars)
        for name, value in module.logged_scalars:
            self.assertTrue(
                torch.isfinite(value).all(), f"{name} not finite: {value}"
            )
        update_ratios = [
            value
            for name, value in module.logged_scalars
            if name.endswith("update_ratio")
        ]
        self.assertTrue(update_ratios)
        self.assertTrue(torch.isfinite(update_ratios[0]).all())
        callback.on_fit_end(trainer, module)

    def test_single_element_bias_var_is_finite(self):
        module = build_module(input_dim=1, output_dim=1, bias_flag=True)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        bias_vars = [
            value
            for name, value in module.logged_scalars
            if name.endswith("bias/var")
        ]
        self.assertTrue(bias_vars)
        self.assertTrue(torch.isfinite(bias_vars[0]).all())
        callback.on_fit_end(trainer, module)

    def test_forward_hook_respects_cadence(self):
        trainer = FakeTrainer()

        skipped = build_module(global_step=5)
        skipped_callback = LinearMonitorCallback(log_every_n_steps=10)
        skipped_callback.on_fit_start(trainer, skipped)
        skipped.linear(torch.randn(2, 4))
        self.assertEqual(skipped.logged_scalars, [])
        skipped_callback.on_fit_end(trainer, skipped)

        logged = build_module(global_step=10)
        logged_callback = LinearMonitorCallback(log_every_n_steps=10)
        logged_callback.on_fit_start(trainer, logged)
        logged.linear(torch.randn(2, 4))
        keys = {name for name, _ in logged.logged_scalars}
        for key in (
            "linear/input/mean",
            "linear/input/var",
            "linear/output/mean",
            "linear/output/var",
        ):
            self.assertIn(key, keys)
        logged_callback.on_fit_end(trainer, logged)

    def test_batch_end_gates_on_trainer_global_step(self):
        # batch_idx is a multiple but global_step is not -> must not log,
        # proving the cadence now follows trainer.global_step.
        skipped = build_module()
        skipped_callback = LinearMonitorCallback(log_every_n_steps=10)
        skipped_trainer = FakeTrainer(global_step=5)
        skipped_callback.on_fit_start(skipped_trainer, skipped)
        skipped_callback.on_train_batch_end(
            skipped_trainer, skipped, None, None, batch_idx=0
        )
        self.assertEqual(skipped.logged_scalars, [])
        skipped_callback.on_fit_end(skipped_trainer, skipped)

        logged = build_module()
        logged_callback = LinearMonitorCallback(log_every_n_steps=10)
        logged_trainer = FakeTrainer(global_step=10)
        logged_callback.on_fit_start(logged_trainer, logged)
        logged_callback.on_train_batch_end(
            logged_trainer, logged, None, None, batch_idx=3
        )
        self.assertTrue(logged.logged_scalars)
        logged_callback.on_fit_end(logged_trainer, logged)


if __name__ == "__main__":
    unittest.main()
