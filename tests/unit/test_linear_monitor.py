import unittest

import torch
from torch import nn

from emperor.linears import LinearLayer, LinearLayerConfig, LinearMonitorCallback
from support.monitor import orchestration_calls


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
    def test_forward_tracking_orchestration_lists_each_tracked_fact(self):
        orchestration = (
            LinearMonitorCallback._LinearMonitorCallback__track_forward_diagnostics
        )

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_input_mean",
                "__track_input_variance",
                "__track_output_mean",
                "__track_output_variance",
            ),
        )

    def test_training_tracking_orchestration_lists_each_tracked_fact(self):
        cls = LinearMonitorCallback
        orchestration = cls._LinearMonitorCallback__track_linear_training_diagnostics
        parameter_calls = (
            "__track_parameter_mean",
            "__track_parameter_variance",
            "__track_parameter_l2_norm",
            "__track_parameter_delta_norm",
            "__track_relative_parameter_delta_norm",
        )

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                *parameter_calls,
                *parameter_calls,
                "__record_parameter_snapshots",
                "__track_gradient_mean",
                "__track_gradient_variance",
                "__track_gradient_norm",
                "__track_update_ratio",
                "__track_gradient_mean",
                "__track_gradient_variance",
                "__track_gradient_norm",
                "__track_dead_input_fraction",
                "__track_dead_output_fraction",
                "__track_spectral_norm",
                "__track_condition_number",
                "__track_effective_rank",
            ),
        )

    def test_init_uses_safe_monitoring_defaults(self):
        callback = LinearMonitorCallback()

        self.assertEqual(callback.log_every_n_steps, 100)
        self.assertIs(callback.log_weight_conditioning, False)

    def test_init_rejects_invalid_options(self):
        for bad_interval in (True, 1.5, float("nan"), float("inf")):
            with self.subTest(log_every_n_steps=bad_interval):
                with self.assertRaisesRegex(
                    TypeError,
                    "log_every_n_steps must be an int",
                ):
                    LinearMonitorCallback(log_every_n_steps=bad_interval)

        for bad_interval in (0, -1):
            with self.subTest(log_every_n_steps=bad_interval):
                with self.assertRaisesRegex(
                    ValueError,
                    "log_every_n_steps must be greater than 0",
                ):
                    LinearMonitorCallback(log_every_n_steps=bad_interval)

        for bad_flag in (0, 1, "false", None):
            with self.subTest(log_weight_conditioning=bad_flag):
                with self.assertRaisesRegex(
                    TypeError,
                    "log_weight_conditioning must be a bool",
                ):
                    LinearMonitorCallback(log_weight_conditioning=bad_flag)

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
            self.assertTrue(torch.isfinite(value).all(), f"{name} not finite: {value}")
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
            value for name, value in module.logged_scalars if name.endswith("bias/var")
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

    def test_logs_parameter_delta_metrics_after_second_sampled_step(self):
        module = build_module()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)
        baseline_names = {name for name, _ in module.logged_scalars}
        self.assertNotIn("linear/weights/delta_norm", baseline_names)
        self.assertNotIn("linear/bias/delta_norm", baseline_names)

        module.logged_scalars.clear()
        trainer.global_step = 1
        module.linear.weight_params.data.add_(0.25)
        module.linear.bias_params.data.add_(0.5)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=1)

        scalars = dict(module.logged_scalars)
        for name in (
            "linear/weights/delta_norm",
            "linear/weights/relative_delta_norm",
            "linear/bias/delta_norm",
            "linear/bias/relative_delta_norm",
        ):
            self.assertIn(name, scalars)
            self.assertTrue(torch.isfinite(scalars[name]).all(), f"{name} not finite")
            self.assertGreater(scalars[name].item(), 0.0)
        callback.on_fit_end(trainer, module)

    def test_does_not_log_bias_delta_metrics_when_bias_disabled(self):
        module = build_module(bias_flag=False)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)
        module.logged_scalars.clear()
        trainer.global_step = 1
        module.linear.weight_params.data.add_(0.25)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=1)

        names = {name for name, _ in module.logged_scalars}
        self.assertIn("linear/weights/delta_norm", names)
        self.assertNotIn("linear/bias/delta_norm", names)
        self.assertNotIn("linear/bias/relative_delta_norm", names)
        callback.on_fit_end(trainer, module)

    def test_delta_metrics_stay_finite_with_zero_parameters(self):
        module = build_module(global_step=0)
        module.linear.weight_params.data.zero_()
        module.linear.bias_params.data.zero_()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)
        module.logged_scalars.clear()
        trainer.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=1)

        scalars = dict(module.logged_scalars)
        for name in (
            "linear/weights/delta_norm",
            "linear/weights/relative_delta_norm",
            "linear/bias/delta_norm",
            "linear/bias/relative_delta_norm",
        ):
            self.assertIn(name, scalars)
            self.assertTrue(torch.isfinite(scalars[name]).all(), f"{name} not finite")
            self.assertEqual(scalars[name].item(), 0.0)
        callback.on_fit_end(trainer, module)

    def test_logs_weight_conditioning_metrics(self):
        module = build_module(input_dim=2, output_dim=2, global_step=0)
        with torch.no_grad():
            module.linear.weight_params.copy_(
                torch.tensor(
                    [
                        [3.0, 0.0],
                        [0.0, 1.0],
                    ]
                )
            )
        callback = LinearMonitorCallback(
            log_every_n_steps=1,
            log_weight_conditioning=True,
        )
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        scalars = dict(module.logged_scalars)
        singular_values = torch.tensor([3.0, 1.0])
        normalized_spectrum = singular_values / singular_values.sum()
        expected_effective_rank = torch.exp(
            -(normalized_spectrum * normalized_spectrum.log()).sum()
        )

        self.assertAlmostEqual(
            scalars["linear/weights/spectral_norm"].item(), 3.0, places=5
        )
        self.assertAlmostEqual(
            scalars["linear/weights/condition_number"].item(), 3.0, places=5
        )
        self.assertAlmostEqual(
            scalars["linear/weights/effective_rank"].item(),
            expected_effective_rank.item(),
            places=5,
        )
        callback.on_fit_end(trainer, module)

    def test_fit_end_removes_forward_hooks_and_clears_state(self):
        module = build_module(global_step=0)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        module.linear(torch.randn(2, 4))
        self.assertTrue(module.logged_scalars)

        module.logged_scalars.clear()
        callback.on_fit_end(trainer, module)

        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._linear_modules, [])
        self.assertEqual(callback._parameter_snapshots, {})
        module.linear(torch.randn(2, 4))
        self.assertEqual(module.logged_scalars, [])

    def test_fit_start_replaces_existing_hooks_when_callback_is_reused(self):
        module = build_module(global_step=0)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_fit_start(trainer, module)
        module.linear(torch.randn(2, 4))

        names = [name for name, _ in module.logged_scalars]
        self.assertEqual(names.count("linear/input/mean"), 1)
        self.assertEqual(names.count("linear/input/var"), 1)
        self.assertEqual(names.count("linear/output/mean"), 1)
        self.assertEqual(names.count("linear/output/var"), 1)
        callback.on_fit_end(trainer, module)

    def test_logs_dead_feature_fractions(self):
        module = build_module(global_step=0)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        names = {name for name, _ in module.logged_scalars}
        self.assertIn("linear/weights/dead_input_fraction", names)
        self.assertIn("linear/weights/dead_output_fraction", names)
        callback.on_fit_end(trainer, module)

    def test_dead_output_fraction_detects_zeroed_column(self):
        module = build_module(input_dim=4, output_dim=4, global_step=0)
        module.linear.weight_params.data.fill_(1.0)
        module.linear.weight_params.data[:, 2] = 0.0
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        scalars = dict(module.logged_scalars)
        self.assertAlmostEqual(
            scalars["linear/weights/dead_output_fraction"].item(), 0.25, places=5
        )
        self.assertEqual(scalars["linear/weights/dead_input_fraction"].item(), 0.0)
        callback.on_fit_end(trainer, module)

    def test_weight_conditioning_can_be_disabled(self):
        module = build_module(global_step=0)
        callback = LinearMonitorCallback(
            log_every_n_steps=1, log_weight_conditioning=False
        )
        trainer = FakeTrainer(global_step=0)

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        names = {name for name, _ in module.logged_scalars}
        self.assertNotIn("linear/weights/spectral_norm", names)
        self.assertNotIn("linear/weights/condition_number", names)
        self.assertIn("linear/weights/dead_output_fraction", names)
        callback.on_fit_end(trainer, module)


if __name__ == "__main__":
    unittest.main()
