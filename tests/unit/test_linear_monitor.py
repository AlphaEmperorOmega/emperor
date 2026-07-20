import os
import tempfile
import unittest
from collections.abc import Callable
from unittest.mock import patch

import torch
from torch import nn
from torch.multiprocessing.spawn import ProcessRaisedException

from emperor.linears import LinearLayer, LinearLayerConfig, LinearMonitorCallback
from emperor.linears._monitoring.diagnostics import _LinearDiagnostics, _TensorMoments


class FakeTrainer:
    def __init__(self, global_step: int = 0, training: bool = True):
        self.global_step = global_step
        self.training = training


class FakeLightningModule(nn.Module):
    def __init__(self, linear: LinearLayer, global_step: int = 0):
        super().__init__()
        self.linear = linear
        self.global_step = global_step
        self.logged_scalars: list[tuple[str, torch.Tensor]] = []
        self.logged_options: list[dict[str, object]] = []

    def log(
        self,
        name: str,
        value: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.logged_scalars.append((name, value))
        self.logged_options.append(kwargs)


class TupleOutputLinear(LinearLayer):
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor]:
        return (super().forward(X),)


class NoInputLinear(LinearLayer):
    def forward(self, X: torch.Tensor | None = None) -> torch.Tensor:
        return self.weight_params.sum().reshape(1)


def build_layer(
    input_dim: int = 4,
    output_dim: int = 4,
    bias_flag: bool = True,
) -> LinearLayer:
    return LinearLayer(
        LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
        )
    )


def build_module(
    input_dim: int = 4,
    output_dim: int = 4,
    bias_flag: bool = True,
    global_step: int = 0,
) -> FakeLightningModule:
    return FakeLightningModule(
        build_layer(input_dim, output_dim, bias_flag),
        global_step=global_step,
    )


def complete_optimizer_step(
    callback: LinearMonitorCallback,
    trainer: FakeTrainer,
    module: FakeLightningModule,
    update: Callable[[], None] | None = None,
) -> None:
    callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
    if update is not None:
        update()
    batch_idx = trainer.global_step
    trainer.global_step += 1
    module.global_step = trainer.global_step
    callback.on_train_batch_end(
        trainer,
        module,
        outputs=None,
        batch=None,
        batch_idx=batch_idx,
    )


def _distributed_monitor_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    module = build_module(input_dim=1, output_dim=1, bias_flag=False)
    callback = LinearMonitorCallback(log_every_n_steps=1)
    trainer = FakeTrainer()
    try:
        with torch.no_grad():
            module.linear.weight_params.fill_(1.0)
        callback.on_fit_start(trainer, module)
        if rank == 0:
            module.linear(torch.tensor([[1.0], [3.0]]))
            module.linear.weight_params.grad = torch.ones_like(
                module.linear.weight_params
            )

        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
        trainer.global_step = 1
        module.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        scalar_values = {
            name: value.detach().cpu().item() for name, value in module.logged_scalars
        }
        payload = (
            tuple(sorted(scalar_values)),
            scalar_values["linear/input/mean"],
            scalar_values["linear/input/var"],
            scalar_values["linear/output/mean"],
            scalar_values["linear/weights/grad_norm"],
            all(options.get("sync_dist") is True for options in module.logged_options),
        )
        gathered_payloads: list[object | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered_payloads, payload)
        assert gathered_payloads == [payload] * world_size
        for actual, expected in zip(payload[1:-1], (2.0, 1.0, 2.0, 1.0), strict=True):
            assert abs(actual - expected) < 1e-6
        assert payload[-1] is True
    finally:
        callback.on_fit_end(trainer, module)
        torch.distributed.destroy_process_group()


class TestLinearDiagnostics(unittest.TestCase):
    def test_single_element_summary_has_population_variance(self):
        summary = _LinearDiagnostics.summarize(torch.tensor([2.0]))

        self.assertEqual(summary.mean.item(), 2.0)
        self.assertEqual(summary.variance.item(), 0.0)
        self.assertEqual(summary.norm.item(), 2.0)

    def test_stable_norm_preserves_complex_magnitude(self):
        norm = _LinearDiagnostics.stable_norm(torch.tensor([3.0 + 4.0j]))

        self.assertEqual(norm.item(), 5.0)

    def test_complex_moments_use_real_magnitudes(self):
        moments = _TensorMoments()
        moments.add(torch.tensor([1.0 + 1.0j]))
        moments.add(torch.tensor([2.0 + 2.0j]))

        summary = moments.summarize()
        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary.mean.item(), 1.5 * 2.0**0.5, places=6)
        self.assertAlmostEqual(summary.variance.item(), 0.5, places=6)
        self.assertAlmostEqual(summary.norm.item(), 10.0**0.5, places=6)

    def test_low_precision_diagnostics_are_upcast(self):
        summary = _LinearDiagnostics.summarize(
            torch.tensor([1.0, 2.0], dtype=torch.float16)
        )

        self.assertEqual(summary.mean.dtype, torch.float32)
        self.assertEqual(summary.variance.dtype, torch.float32)
        self.assertEqual(summary.norm.dtype, torch.float32)

    def test_moments_keep_zero_variance_for_extreme_float64_constant(self):
        moments = _TensorMoments()
        moments.add(torch.tensor([1e308], dtype=torch.float64))

        summary = moments.summarize()

        self.assertIsNotNone(summary)
        self.assertEqual(summary.mean.item(), 1e308)
        self.assertEqual(summary.variance.item(), 0.0)
        self.assertEqual(summary.norm.item(), 1e308)

    def test_summary_avoids_overflow_for_extreme_float64_constant(self):
        summary = _LinearDiagnostics.summarize(
            torch.tensor([1e308, 1e308], dtype=torch.float64)
        )

        self.assertEqual(summary.mean.item(), 1e308)
        self.assertEqual(summary.variance.item(), 0.0)
        self.assertTrue(torch.isfinite(summary.norm))
        self.assertAlmostEqual(summary.norm.item() / 1e308, 2.0**0.5)

    def test_distributed_activation_summary_combines_counts_and_moments(self):
        local = _TensorMoments()
        local.add(torch.tensor([1.0, 3.0]))
        remote_state = torch.tensor([1.0, 5.0, 0.0], dtype=torch.float64)

        def gather_remote_moments(
            gathered: list[torch.Tensor],
            local_state: torch.Tensor,
        ) -> None:
            gathered[0].copy_(local_state)
            gathered[1].copy_(remote_state)

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                torch.distributed,
                "all_gather",
                side_effect=gather_remote_moments,
            ),
        ):
            summary = _LinearDiagnostics.distributed_moments_summary(
                local,
                reference=torch.zeros(1),
            )

        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary.mean.item(), 3.0, places=6)
        self.assertAlmostEqual(summary.variance.item(), 8.0 / 3.0, places=6)
        self.assertAlmostEqual(summary.norm.item(), 35.0**0.5, places=6)

    def test_distributed_activation_summary_avoids_large_constant_overflow(self):
        local = _TensorMoments()
        local.add(torch.tensor([1e20, 1e20]))
        assert local.mean is not None
        remote_state = torch.tensor(
            [2.0, local.mean.item(), 0.0],
            dtype=torch.float64,
        )

        def gather_remote_moments(
            gathered: list[torch.Tensor],
            local_state: torch.Tensor,
        ) -> None:
            gathered[0].copy_(local_state)
            gathered[1].copy_(remote_state)

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                torch.distributed,
                "all_gather",
                side_effect=gather_remote_moments,
            ),
        ):
            summary = _LinearDiagnostics.distributed_moments_summary(
                local,
                reference=torch.zeros(1),
            )

        self.assertIsNotNone(summary)
        self.assertTrue(torch.isfinite(summary.mean))
        self.assertEqual(summary.variance.item(), 0.0)
        self.assertTrue(torch.isfinite(summary.norm))
        self.assertAlmostEqual(summary.norm.item() / 1e20, 2.0, places=6)

    def test_distributed_gradient_summary_does_not_inflate_replicated_norm(self):
        local = _LinearDiagnostics.summarize(torch.tensor([1.0, 3.0]))

        def add_remote_summary(reduced: torch.Tensor) -> None:
            reduced.add_(
                torch.tensor(
                    [1.0, 2.0, 1.0, 10.0**0.5],
                    dtype=torch.float64,
                )
            )

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=add_remote_summary,
            ),
        ):
            summary = _LinearDiagnostics.distributed_optional_summary(
                local,
                reference=torch.zeros(1),
            )

        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary.mean.item(), 2.0, places=6)
        self.assertAlmostEqual(summary.variance.item(), 1.0, places=6)
        self.assertAlmostEqual(summary.norm.item(), 10.0**0.5, places=6)

    def test_distributed_summaries_remain_absent_when_no_rank_contributes(self):
        def gather_local_state(
            gathered: list[torch.Tensor],
            local_state: torch.Tensor,
        ) -> None:
            gathered[0].copy_(local_state)

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=1),
            patch.object(
                torch.distributed,
                "all_gather",
                side_effect=gather_local_state,
            ),
            patch.object(torch.distributed, "all_reduce"),
        ):
            activation_summary = _LinearDiagnostics.distributed_moments_summary(
                None,
                reference=torch.zeros(1),
            )
            gradient_summary = _LinearDiagnostics.distributed_optional_summary(
                None,
                reference=torch.zeros(1),
            )

        self.assertIsNone(activation_summary)
        self.assertIsNone(gradient_summary)

    def test_conditioning_metrics_are_scale_invariant(self):
        matrix = torch.diag(torch.tensor([3.0, 1.0], dtype=torch.float64))

        baseline = _LinearDiagnostics.weight_conditioning(matrix)
        tiny = _LinearDiagnostics.weight_conditioning(matrix * 1e-20)

        torch.testing.assert_close(tiny.condition_number, baseline.condition_number)
        torch.testing.assert_close(tiny.effective_rank, baseline.effective_rank)

    def test_conditioning_preserves_complex_matrix_structure(self):
        matrix = torch.tensor(
            [[1.0, 1.0j], [1.0j, 1.0]],
            dtype=torch.complex64,
        )

        metrics = _LinearDiagnostics.weight_conditioning(matrix)

        self.assertAlmostEqual(metrics.condition_number.item(), 1.0, places=5)

    def test_conditioning_reports_singular_zero_and_non_finite_weights(self):
        singular = _LinearDiagnostics.weight_conditioning(
            torch.tensor([[2.0, 0.0], [0.0, 0.0]])
        )
        self.assertTrue(torch.isinf(singular.condition_number))
        self.assertEqual(singular.effective_rank.item(), 1.0)

        zero = _LinearDiagnostics.weight_conditioning(torch.zeros(2, 2))
        self.assertEqual(zero.spectral_norm.item(), 0.0)
        self.assertTrue(torch.isinf(zero.condition_number))
        self.assertEqual(zero.effective_rank.item(), 0.0)

        non_finite = _LinearDiagnostics.weight_conditioning(
            torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])
        )
        for metric in (
            non_finite.spectral_norm,
            non_finite.condition_number,
            non_finite.effective_rank,
        ):
            self.assertTrue(torch.isnan(metric))

    def test_conditioning_handles_svd_convergence_failure(self):
        with patch.object(
            torch.linalg,
            "svdvals",
            side_effect=torch.linalg.LinAlgError("did not converge"),
        ):
            metrics = _LinearDiagnostics.weight_conditioning(torch.eye(2))

        self.assertTrue(torch.isnan(metrics.spectral_norm))
        self.assertTrue(torch.isnan(metrics.condition_number))
        self.assertTrue(torch.isnan(metrics.effective_rank))


class TestLinearMonitorCallback(unittest.TestCase):
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

    def test_batch_end_without_optimizer_boundary_does_not_emit(self):
        module = build_module()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        self.assertIsNotNone(callback._discovery_hook)
        module.linear(torch.ones(2, 4))
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        self.assertEqual(module.logged_scalars, [])
        self.assertTrue(callback._activation_moments)
        self.assertIsNone(callback._discovery_hook)
        callback.on_fit_end(trainer, module)

    def test_dynamic_discovery_ignores_non_linears_and_installs_once(self):
        module = build_module()
        callback = LinearMonitorCallback(log_every_n_steps=2)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
        discovery_hook = callback._discovery_hook
        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
        torch.nn.ReLU()(torch.ones(1))

        self.assertIs(callback._discovery_hook, discovery_hook)
        self.assertEqual(set(callback._linear_modules), {"linear"})
        callback.on_fit_end(trainer, module)

    def test_forward_and_parameter_metrics_share_optimizer_step_cadence(self):
        skipped = build_module(global_step=4)
        skipped_trainer = FakeTrainer(global_step=4)
        skipped_callback = LinearMonitorCallback(log_every_n_steps=10)
        skipped_callback.on_fit_start(skipped_trainer, skipped)
        skipped.linear(torch.ones(2, 4))
        complete_optimizer_step(skipped_callback, skipped_trainer, skipped)
        self.assertEqual(skipped.logged_scalars, [])
        skipped_callback.on_fit_end(skipped_trainer, skipped)

        logged = build_module(global_step=9)
        logged_trainer = FakeTrainer(global_step=9)
        logged_callback = LinearMonitorCallback(log_every_n_steps=10)
        logged_callback.on_fit_start(logged_trainer, logged)
        logged.linear(torch.ones(2, 4))
        self.assertEqual(logged.logged_scalars, [])
        complete_optimizer_step(logged_callback, logged_trainer, logged)

        names = {name for name, _ in logged.logged_scalars}
        for name in (
            "linear/input/mean",
            "linear/input/var",
            "linear/output/mean",
            "linear/output/var",
            "linear/weights/mean",
        ):
            self.assertIn(name, names)
        logged_callback.on_fit_end(logged_trainer, logged)

    def test_stale_pending_step_and_activations_are_discarded(self):
        module = build_module(global_step=1)
        callback = LinearMonitorCallback(log_every_n_steps=2)
        trainer = FakeTrainer(global_step=1)

        callback.on_fit_start(trainer, module)
        module.linear(torch.ones(1, 4))
        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
        self.assertTrue(callback._activation_moments)
        self.assertIsNotNone(callback._pending_step)

        trainer.global_step = 4
        module.global_step = 4
        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]

        self.assertEqual(callback._activation_moments, {})
        self.assertIsNone(callback._pending_step)
        callback.on_fit_end(trainer, module)

    def test_repeated_forwards_are_aggregated_once_per_optimizer_step(self):
        module = build_module(input_dim=1, output_dim=1, bias_flag=False)
        with torch.no_grad():
            module.linear.weight_params.fill_(1.0)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        module.linear(torch.tensor([[1.0]]))
        module.linear(torch.tensor([[3.0]]))
        module.linear.weight_params.grad = torch.tensor([[2.0]])

        def update() -> None:
            with torch.no_grad():
                module.linear.weight_params.sub_(0.2)

        complete_optimizer_step(callback, trainer, module, update)

        scalars = dict(module.logged_scalars)
        self.assertEqual(scalars["linear/input/mean"].item(), 2.0)
        self.assertEqual(scalars["linear/input/var"].item(), 1.0)
        self.assertEqual(scalars["linear/output/mean"].item(), 2.0)
        self.assertEqual(scalars["linear/output/var"].item(), 1.0)
        self.assertAlmostEqual(
            scalars["linear/weights/update_ratio"].item(),
            0.2,
            places=6,
        )
        names = [name for name, _ in module.logged_scalars]
        self.assertEqual(names.count("linear/input/mean"), 1)
        self.assertEqual(names.count("linear/output/mean"), 1)
        callback.on_fit_end(trainer, module)

    def test_forward_hook_tracks_keyword_tensor_inputs(self):
        module = build_module()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        module.linear(X=torch.ones(2, 4))
        complete_optimizer_step(callback, trainer, module)

        names = {name for name, _ in module.logged_scalars}
        self.assertIn("linear/input/mean", names)
        self.assertIn("linear/output/mean", names)
        callback.on_fit_end(trainer, module)

    def test_forward_hook_logs_available_channel_for_tuple_output(self):
        module = FakeLightningModule(
            TupleOutputLinear(
                LinearLayerConfig(input_dim=2, output_dim=1, bias_flag=False)
            )
        )
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        module.linear(torch.ones(1, 2))
        complete_optimizer_step(callback, trainer, module)

        names = {name for name, _ in module.logged_scalars}
        self.assertIn("linear/input/mean", names)
        self.assertNotIn("linear/output/mean", names)
        callback.on_fit_end(trainer, module)

    def test_forward_hook_logs_available_channel_without_tensor_input(self):
        module = FakeLightningModule(
            NoInputLinear(LinearLayerConfig(input_dim=2, output_dim=1, bias_flag=False))
        )
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        module.linear()
        complete_optimizer_step(callback, trainer, module)

        names = {name for name, _ in module.logged_scalars}
        self.assertNotIn("linear/input/mean", names)
        self.assertIn("linear/output/mean", names)
        callback.on_fit_end(trainer, module)

    def test_forward_hook_skips_empty_tensors(self):
        module = build_module()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        module.linear(torch.empty(0, 4))
        complete_optimizer_step(callback, trainer, module)

        activation_names = {
            name
            for name, _ in module.logged_scalars
            if "/input/" in name or "/output/" in name
        }
        self.assertEqual(activation_names, set())
        callback.on_fit_end(trainer, module)

    def test_update_ratio_uses_parameter_change_instead_of_raw_gradient(self):
        module = build_module(input_dim=2, output_dim=2, bias_flag=False)
        with torch.no_grad():
            module.linear.weight_params.fill_(1.0)
        module.linear.weight_params.grad = torch.ones_like(module.linear.weight_params)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        complete_optimizer_step(callback, trainer, module)

        scalars = dict(module.logged_scalars)
        self.assertEqual(scalars["linear/weights/update_ratio"].item(), 0.0)
        self.assertEqual(
            scalars["linear/weights/gradient_to_weight_norm_ratio"].item(),
            1.0,
        )
        callback.on_fit_end(trainer, module)

    def test_repeated_optimizer_closures_preserve_the_first_pre_step_values(self):
        module = build_module(input_dim=1, output_dim=1, bias_flag=False)
        with torch.no_grad():
            module.linear.weight_params.fill_(1.0)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
        with torch.no_grad():
            module.linear.weight_params.zero_()
        complete_optimizer_step(callback, trainer, module)

        scalars = dict(module.logged_scalars)
        self.assertEqual(scalars["linear/weights/delta_norm"].item(), 1.0)
        self.assertEqual(scalars["linear/weights/update_ratio"].item(), 1.0)
        callback.on_fit_end(trainer, module)

    def test_multiple_optimizer_steps_in_one_batch_do_not_overwrite_state(self):
        module = build_module(input_dim=1, output_dim=1, bias_flag=False)
        second = build_layer(1, 1, False)
        module.add_module("second", second)
        with torch.no_grad():
            module.linear.weight_params.fill_(1.0)
            second.weight_params.fill_(1.0)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
        with torch.no_grad():
            module.linear.weight_params.sub_(0.2)
        trainer.global_step = 1
        module.global_step = 1

        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
        with torch.no_grad():
            second.weight_params.sub_(0.2)
        trainer.global_step = 2
        module.global_step = 2
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        first_deltas = [
            value.item()
            for name, value in module.logged_scalars
            if name == "linear/weights/delta_norm"
        ]
        second_deltas = [
            value.item()
            for name, value in module.logged_scalars
            if name == "second/weights/delta_norm"
        ]
        self.assertEqual(len(first_deltas), 2)
        self.assertEqual(len(second_deltas), 2)
        self.assertAlmostEqual(first_deltas[0], 0.2, places=6)
        self.assertEqual(first_deltas[1], 0.0)
        self.assertEqual(second_deltas[0], 0.0)
        self.assertAlmostEqual(second_deltas[1], 0.2, places=6)
        callback.on_fit_end(trainer, module)

    def test_stale_activations_are_not_reassigned_to_a_later_step(self):
        module = build_module()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        module.linear(torch.ones(1, 4))
        trainer.global_step = 1
        module.global_step = 1
        complete_optimizer_step(callback, trainer, module)

        names = {name for name, _ in module.logged_scalars}
        self.assertNotIn("linear/input/mean", names)
        self.assertNotIn("linear/output/mean", names)
        self.assertEqual(callback._activation_moments, {})
        self.assertEqual(callback._activation_modules, {})
        callback.on_fit_end(trainer, module)

    def test_manual_optimizer_can_enter_sampled_cadence_after_dynamic_rename(self):
        module = build_module(input_dim=1, output_dim=1, bias_flag=False)
        second = build_layer(1, 1, False)
        module.add_module("second", second)
        first_optimizer = torch.optim.SGD(module.linear.parameters(), lr=0.1)
        second_optimizer = torch.optim.SGD(second.parameters(), lr=0.1)
        callback = LinearMonitorCallback(log_every_n_steps=2)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        self.assertIsNone(callback._discovery_hook)
        callback.on_before_optimizer_step(trainer, module, first_optimizer)
        self.assertIsNotNone(callback._discovery_hook)
        trainer.global_step = 1
        module.global_step = 1

        delattr(module, "second")
        module.add_module("renamed", second)
        second(torch.ones(1, 1))
        callback.on_before_optimizer_step(trainer, module, second_optimizer)
        with torch.no_grad():
            second.weight_params.sub_(0.2)
        trainer.global_step = 2
        module.global_step = 2
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        names = {name for name, _ in module.logged_scalars}
        self.assertIn("renamed/input/mean", names)
        self.assertIn("renamed/weights/delta_norm", names)
        self.assertNotIn("second/input/mean", names)
        self.assertIsNone(callback._discovery_hook)
        callback.on_fit_end(trainer, module)

    def test_float64_capture_preserves_small_parameter_changes(self):
        module = build_module(input_dim=2, output_dim=2, bias_flag=False).double()
        with torch.no_grad():
            module.linear.weight_params.fill_(1.0)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)

        def update() -> None:
            with torch.no_grad():
                module.linear.weight_params.add_(1e-10)

        complete_optimizer_step(callback, trainer, module, update)

        scalars = dict(module.logged_scalars)
        delta = scalars["linear/weights/delta_norm"]
        self.assertGreater(delta.item(), 0.0)
        self.assertEqual(delta.dtype, torch.float64)
        torch.testing.assert_close(
            delta,
            torch.tensor(2e-10, dtype=torch.float64),
            rtol=1e-6,
            atol=0.0,
        )
        callback.on_fit_end(trainer, module)

    def test_parameter_delta_metrics_compare_pre_and_post_step_values(self):
        module = build_module(input_dim=2, output_dim=2)
        with torch.no_grad():
            module.linear.weight_params.fill_(1.0)
            module.linear.bias_params.fill_(1.0)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)

        def update() -> None:
            with torch.no_grad():
                module.linear.weight_params.add_(0.25)
                module.linear.bias_params.add_(0.5)

        complete_optimizer_step(callback, trainer, module, update)

        scalars = dict(module.logged_scalars)
        self.assertAlmostEqual(
            scalars["linear/weights/delta_norm"].item(),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            scalars["linear/weights/relative_delta_norm"].item(),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            scalars["linear/bias/relative_delta_norm"].item(),
            0.5,
            places=6,
        )
        callback.on_fit_end(trainer, module)

    def test_complex_phase_change_is_visible_in_parameter_delta(self):
        module = build_module(input_dim=1, output_dim=1, bias_flag=False)
        module.linear.weight_params = nn.Parameter(torch.tensor([[1.0 + 0.0j]]))
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)

        def update() -> None:
            with torch.no_grad():
                module.linear.weight_params.copy_(torch.tensor([[0.0 + 1.0j]]))

        complete_optimizer_step(callback, trainer, module, update)

        scalars = dict(module.logged_scalars)
        self.assertAlmostEqual(
            scalars["linear/weights/delta_norm"].item(),
            2.0**0.5,
            places=6,
        )
        self.assertAlmostEqual(
            scalars["linear/weights/relative_delta_norm"].item(),
            2.0**0.5,
            places=6,
        )
        callback.on_fit_end(trainer, module)

    def test_replaced_parameter_is_not_compared_with_previous_parameter(self):
        module = build_module(input_dim=2, output_dim=2, bias_flag=False)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
        module.linear.weight_params = nn.Parameter(torch.zeros(2, 2))
        trainer.global_step = 1
        module.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        names = {name for name, _ in module.logged_scalars}
        self.assertNotIn("linear/weights/delta_norm", names)
        self.assertNotIn("linear/weights/relative_delta_norm", names)
        self.assertNotIn("linear/weights/update_ratio", names)
        callback.on_fit_end(trainer, module)

    def test_zero_parameters_and_zero_delta_have_zero_relative_change(self):
        module = build_module()
        with torch.no_grad():
            module.linear.weight_params.zero_()
            module.linear.bias_params.zero_()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        complete_optimizer_step(callback, trainer, module)

        scalars = dict(module.logged_scalars)
        for name in (
            "linear/weights/delta_norm",
            "linear/weights/relative_delta_norm",
            "linear/weights/update_ratio",
            "linear/bias/delta_norm",
            "linear/bias/relative_delta_norm",
        ):
            self.assertEqual(scalars[name].item(), 0.0, name)
        callback.on_fit_end(trainer, module)

    def test_nonzero_update_from_zero_parameters_has_infinite_relative_change(self):
        module = build_module(input_dim=1, output_dim=1, bias_flag=False)
        with torch.no_grad():
            module.linear.weight_params.zero_()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)

        def update() -> None:
            with torch.no_grad():
                module.linear.weight_params.fill_(1.0)

        complete_optimizer_step(callback, trainer, module, update)

        scalars = dict(module.logged_scalars)
        self.assertEqual(scalars["linear/weights/delta_norm"].item(), 1.0)
        self.assertTrue(torch.isinf(scalars["linear/weights/relative_delta_norm"]))
        self.assertTrue(torch.isinf(scalars["linear/weights/update_ratio"]))
        callback.on_fit_end(trainer, module)

    def test_gradient_summaries_are_finite_with_zero_weights(self):
        module = build_module()
        with torch.no_grad():
            module.linear.weight_params.zero_()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        module.linear(torch.ones(3, 4)).sum().backward()
        complete_optimizer_step(callback, trainer, module)

        scalars = dict(module.logged_scalars)
        for name in (
            "linear/weights/grad_mean",
            "linear/weights/grad_var",
            "linear/weights/grad_norm",
        ):
            self.assertTrue(torch.isfinite(scalars[name]).all(), name)
        self.assertTrue(
            torch.isinf(scalars["linear/weights/gradient_to_weight_norm_ratio"]).all()
        )
        self.assertEqual(scalars["linear/weights/update_ratio"].item(), 0.0)
        callback.on_fit_end(trainer, module)

    def test_single_element_bias_variance_is_finite(self):
        module = build_module(input_dim=1, output_dim=1)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        complete_optimizer_step(callback, trainer, module)

        bias_variance = dict(module.logged_scalars)["linear/bias/var"]
        self.assertEqual(bias_variance.item(), 0.0)
        callback.on_fit_end(trainer, module)

    def test_bias_metrics_are_omitted_when_bias_is_disabled(self):
        module = build_module(bias_flag=False)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        complete_optimizer_step(callback, trainer, module)

        names = {name for name, _ in module.logged_scalars}
        self.assertIn("linear/weights/delta_norm", names)
        self.assertFalse(any(name.startswith("linear/bias/") for name in names))
        callback.on_fit_end(trainer, module)

    def test_weight_conditioning_is_explicitly_opt_in(self):
        module = build_module(input_dim=2, output_dim=2)
        with torch.no_grad():
            module.linear.weight_params.copy_(torch.diag(torch.tensor([3.0, 1.0])))
        callback = LinearMonitorCallback(
            log_every_n_steps=1,
            log_weight_conditioning=True,
        )
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        complete_optimizer_step(callback, trainer, module)

        scalars = dict(module.logged_scalars)
        singular_values = torch.tensor([3.0, 1.0])
        probabilities = singular_values / singular_values.sum()
        expected_effective_rank = torch.exp(
            -torch.xlogy(probabilities, probabilities).sum()
        )
        self.assertAlmostEqual(
            scalars["linear/weights/spectral_norm"].item(),
            3.0,
            places=5,
        )
        self.assertAlmostEqual(
            scalars["linear/weights/condition_number"].item(),
            3.0,
            places=5,
        )
        torch.testing.assert_close(
            scalars["linear/weights/effective_rank"],
            expected_effective_rank,
        )
        callback.on_fit_end(trainer, module)

    def test_default_monitor_skips_weight_conditioning_but_logs_health(self):
        module = build_module()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        complete_optimizer_step(callback, trainer, module)

        names = {name for name, _ in module.logged_scalars}
        self.assertNotIn("linear/weights/spectral_norm", names)
        self.assertNotIn("linear/weights/condition_number", names)
        self.assertIn("linear/weights/dead_output_fraction", names)
        callback.on_fit_end(trainer, module)

    def test_dead_feature_fractions_use_each_feature_axis(self):
        module = build_module(input_dim=4, output_dim=4)
        with torch.no_grad():
            module.linear.weight_params.fill_(1.0)
            module.linear.weight_params[:, 2] = 0.0
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        complete_optimizer_step(callback, trainer, module)

        scalars = dict(module.logged_scalars)
        self.assertEqual(
            scalars["linear/weights/dead_output_fraction"].item(),
            0.25,
        )
        self.assertEqual(
            scalars["linear/weights/dead_input_fraction"].item(),
            0.0,
        )
        callback.on_fit_end(trainer, module)

    def test_non_finite_weights_produce_unknown_dead_feature_fractions(self):
        module = build_module(input_dim=2, output_dim=2, bias_flag=False)
        with torch.no_grad():
            module.linear.weight_params.copy_(
                torch.tensor([[float("inf"), 0.0], [0.0, 1.0]])
            )
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        complete_optimizer_step(callback, trainer, module)

        scalars = dict(module.logged_scalars)
        self.assertTrue(torch.isnan(scalars["linear/weights/dead_input_fraction"]))
        self.assertTrue(torch.isnan(scalars["linear/weights/dead_output_fraction"]))
        callback.on_fit_end(trainer, module)

    def test_refresh_replaces_hooks_when_linear_module_changes(self):
        module = build_module(input_dim=2, output_dim=1, bias_flag=False)
        original_layer = module.linear
        replacement_layer = build_layer(2, 1, False)
        untouched_layer = build_layer(2, 1, False)
        module.add_module("untouched", untouched_layer)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        self.assertEqual(len(original_layer._forward_hooks), 1)
        original_layer(torch.ones(1, 2))
        untouched_layer(torch.ones(1, 2))
        self.assertTrue(callback._activation_moments)
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        module.linear = replacement_layer
        replacement_layer(torch.ones(1, 2))

        self.assertEqual(len(original_layer._forward_hooks), 0)
        self.assertEqual(len(replacement_layer._forward_hooks), 1)
        self.assertTrue(callback._activation_moments)
        self.assertEqual(
            callback._activation_moments[(1, id(replacement_layer), "input")].count,
            2,
        )
        complete_optimizer_step(callback, trainer, module)
        names = [name for name, _ in module.logged_scalars]
        self.assertEqual(names.count("linear/input/mean"), 1)
        callback.on_fit_end(trainer, module)

    def test_replacement_after_forward_preserves_the_layer_that_ran(self):
        module = build_module(input_dim=1, output_dim=1, bias_flag=False)
        original_layer = module.linear
        with torch.no_grad():
            original_layer.weight_params.fill_(1.0)
        optimizer = torch.optim.SGD(original_layer.parameters(), lr=0.1)
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        module.linear(torch.tensor([[2.0]])).sum().backward()
        replacement_layer = build_layer(1, 1, False)
        with torch.no_grad():
            replacement_layer.weight_params.fill_(5.0)
        module.linear = replacement_layer

        callback.on_before_optimizer_step(trainer, module, optimizer)
        optimizer.step()
        trainer.global_step = 1
        module.global_step = 1
        callback.on_train_batch_end(trainer, module, None, None, batch_idx=0)

        scalars = dict(module.logged_scalars)
        self.assertEqual(scalars["linear/input/mean"].item(), 2.0)
        self.assertEqual(scalars["linear/output/mean"].item(), 2.0)
        self.assertAlmostEqual(scalars["linear/weights/mean"].item(), 0.8)
        self.assertAlmostEqual(scalars["linear/weights/delta_norm"].item(), 0.2)
        self.assertIs(callback._linear_modules["linear"], replacement_layer)
        callback.on_fit_end(trainer, module)

    def test_discovery_refreshes_namespace_when_tracked_layer_moves(self):
        module = build_module(input_dim=2, output_dim=1, bias_flag=False)
        moved_layer = module.linear
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        delattr(module, "linear")
        module.add_module("renamed", moved_layer)
        moved_layer(torch.ones(1, 2))
        complete_optimizer_step(callback, trainer, module)

        names = {name for name, _ in module.logged_scalars}
        self.assertIn("renamed/input/mean", names)
        self.assertIn("renamed/weights/mean", names)
        self.assertNotIn("linear/input/mean", names)
        self.assertEqual(len(moved_layer._forward_hooks), 1)
        callback.on_fit_end(trainer, module)
        self.assertEqual(len(moved_layer._forward_hooks), 0)

    def test_fit_start_reuse_does_not_duplicate_hooks_or_metrics(self):
        module = build_module()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_fit_start(trainer, module)
        self.assertEqual(len(module.linear._forward_hooks), 1)
        module.linear(torch.ones(2, 4))
        complete_optimizer_step(callback, trainer, module)

        names = [name for name, _ in module.logged_scalars]
        self.assertEqual(names.count("linear/input/mean"), 1)
        self.assertEqual(names.count("linear/output/mean"), 1)
        callback.on_fit_end(trainer, module)

    def test_fit_end_removes_hooks_and_clears_transient_state(self):
        module = build_module()
        callback = LinearMonitorCallback(log_every_n_steps=1)
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        self.assertIsNotNone(callback._discovery_hook)
        module.linear(torch.ones(2, 4))
        callback.on_before_optimizer_step(trainer, module, None)  # type: ignore[arg-type]
        self.assertTrue(callback._activation_moments)
        self.assertIsNotNone(callback._pending_step)

        callback.on_fit_end(trainer, module)

        self.assertEqual(callback._hooks, {})
        self.assertEqual(callback._linear_modules, {})
        self.assertEqual(callback._activation_moments, {})
        self.assertEqual(callback._activation_modules, {})
        self.assertIsNone(callback._pending_step)
        self.assertIsNone(callback._discovery_hook)
        self.assertEqual(len(module.linear._forward_hooks), 0)

    def test_every_metric_requests_distributed_synchronization(self):
        module = build_module(input_dim=1, output_dim=1)
        callback = LinearMonitorCallback(
            log_every_n_steps=1,
            log_weight_conditioning=True,
        )
        trainer = FakeTrainer()

        callback.on_fit_start(trainer, module)
        module.linear(torch.ones(1, 1)).sum().backward()
        complete_optimizer_step(callback, trainer, module)

        self.assertTrue(module.logged_options)
        self.assertTrue(
            all(options.get("sync_dist") is True for options in module.logged_options)
        )
        callback.on_fit_end(trainer, module)

    @unittest.skipUnless(
        torch.distributed.is_available() and torch.distributed.is_gloo_available(),
        "gloo process group support is required",
    )
    def test_distributed_optional_channels_emit_identically_on_every_rank(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as temporary_directory:
            init_file = os.path.join(temporary_directory, "process_group_init")
            try:
                torch.multiprocessing.spawn(
                    _distributed_monitor_worker,
                    args=(world_size, init_file),
                    nprocs=world_size,
                    join=True,
                )
            except ProcessRaisedException as error:
                if "Operation not permitted" in str(error):
                    self.skipTest("loopback sockets are blocked in this environment")
                raise


if __name__ == "__main__":
    unittest.main()
