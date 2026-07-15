import unittest
from dataclasses import dataclass

import torch
from emperor.base.layer import LayerState, RecurrentLayerConfig
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.halting.config import StickBreakingConfig
from emperor.halting.core.monitor import HaltingMonitorCallback
from emperor.halting.core.tracker import (
    HaltingUsageTracker,
    HaltingUsageTrackerManager,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from torch import Tensor, nn

from support.monitor import orchestration_calls

SCALAR_SUFFIXES = [
    "depth/ponder_cost_mean",
    "depth/ponder_cost_std",
    "depth/step_count",
    "halt/halted_fraction",
    "halt/accumulated_halt_prob_mean",
    "halt/remaining_mass_mean",
    "halt/saturation_fraction",
    "loss/ponder_loss",
]


@dataclass
class FakeSoftState:
    continuation_probability: Tensor
    accumulated_ponder_cost: Tensor


class FakeExperiment:
    def __init__(self):
        self.histograms = []
        self.images = []

    def add_histogram(self, tag, values, step):
        self.histograms.append((tag, values.clone(), step))

    def add_image(self, tag, image, step, dataformats):
        self.images.append((tag, image.clone(), step, dataformats))


class StrictHistogramExperiment(FakeExperiment):
    def add_histogram(self, tag, values, step):
        if values.numel() == 0:
            raise ValueError("The input has no element.")
        super().add_histogram(tag, values, step)


class FakeLogger:
    def __init__(self, experiment):
        self.experiment = experiment


class FakeTrainer:
    def __init__(self, global_step: int = 0):
        self.global_step = global_step


class FakeLightningModule(nn.Module):
    def __init__(self, recurrent: nn.Module, experiment=None, global_step: int = 0):
        super().__init__()
        self.recurrent = recurrent
        self.logger = FakeLogger(experiment) if experiment is not None else None
        self.global_step = global_step
        self.logged_scalars = []

    def log(self, name, value, *args, **kwargs):
        self.logged_scalars.append((name, value))


class TestHaltingMonitorCallback(unittest.TestCase):
    def test_tracking_orchestration_lists_each_tracked_fact(self):
        orchestration = (
            HaltingMonitorCallback._HaltingMonitorCallback__track_halting_diagnostics
        )

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_ponder_cost_mean",
                "__track_ponder_cost_standard_deviation",
                "__track_step_count",
                "__track_halted_fraction",
                "__track_accumulated_halt_probability_mean",
                "__track_remaining_mass_mean",
                "__track_saturation_fraction",
                "__track_ponder_loss",
                "__track_survival_history",
                "__track_survival_histogram",
                "__track_ponder_cost_histogram",
                "__track_survival_heatmap",
            ),
        )

    def __layer_config(self, model_config) -> LayerConfig:
        return LayerConfig(
            activation=ActivationOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            layer_model_config=model_config,
        )

    def __stack(self, dim, output_dim, bias_flag, bias_option) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=output_dim,
            num_layers=1,
            last_layer_bias_option=bias_option,
            apply_output_pipeline_flag=False,
            layer_config=self.__layer_config(LinearLayerConfig(bias_flag=bias_flag)),
        )

    def build_recurrent_halting_layer(
        self,
        dim: int = 4,
        max_steps: int = 3,
        threshold: float = 1.0,
    ) -> nn.Module:
        cfg = RecurrentLayerConfig(
            input_dim=dim,
            output_dim=dim,
            max_steps=max_steps,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=self.__stack(dim, dim, True, LastLayerBiasOptions.DEFAULT),
            gate_config=None,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=StickBreakingConfig(
                input_dim=dim,
                threshold=threshold,
                halting_dropout=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=self.__stack(
                    dim, 2, False, LastLayerBiasOptions.DISABLED
                ),
            ),
        )
        layer = cfg.build()
        layer.eval()
        return layer

    def build_module(
        self,
        experiment=None,
        global_step: int = 0,
        **layer_kwargs,
    ) -> FakeLightningModule:
        layer = self.build_recurrent_halting_layer(**layer_kwargs)
        return FakeLightningModule(
            layer, experiment=experiment, global_step=global_step
        )

    def primed_callback(self, module, **callback_kwargs) -> HaltingMonitorCallback:
        callback = HaltingMonitorCallback(**callback_kwargs)
        callback.on_fit_start(trainer=None, pl_module=module)
        return callback

    def drive_forward(self, module: FakeLightningModule, batch_size: int = 5) -> None:
        module.recurrent(LayerState(hidden=torch.randn(batch_size, 4)))

    def test_init_rejects_non_positive_log_interval(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                HaltingMonitorCallback(log_every_n_steps=bad)

    def test_init_rejects_non_positive_history_size(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                HaltingMonitorCallback(history_size=bad)

    def test_on_fit_start_discovers_and_wraps_halting_modules(self):
        module = self.build_module()
        callback = self.primed_callback(module, log_every_n_steps=1)

        self.assertEqual(
            [name for name, _ in callback._halting_layers],
            ["recurrent.halting_model"],
        )
        halting_model = module.recurrent.halting_model
        self.assertIn("update_halting_state", halting_model.__dict__)
        self.assertIn("finalize_weighted_accumulation", halting_model.__dict__)
        self.assertIsNotNone(getattr(halting_model, "_usage_tracker", None))

    def test_repeated_fit_start_replaces_tracker_without_stacking_wrappers(self):
        module = self.build_module()
        callback = self.primed_callback(module, log_every_n_steps=1)
        halting_model = module.recurrent.halting_model
        first_tracker = halting_model._usage_tracker

        callback.on_fit_start(trainer=None, pl_module=module)

        self.assertEqual(len(callback._halting_layers), 1)
        self.assertIsNot(halting_model._usage_tracker, first_tracker)
        callback.on_fit_end(trainer=None, pl_module=module)
        self.assertNotIn("update_halting_state", halting_model.__dict__)
        self.assertNotIn("finalize_weighted_accumulation", halting_model.__dict__)

    def test_logs_expected_keys_all_finite(self):
        module = self.build_module(global_step=10)
        callback = self.primed_callback(module, log_every_n_steps=10)

        self.drive_forward(module)
        callback.on_train_batch_end(FakeTrainer(10), module, None, None, batch_idx=0)

        logged_names = {name for name, _ in module.logged_scalars}
        for suffix in SCALAR_SUFFIXES:
            self.assertIn(f"recurrent.halting_model/{suffix}", logged_names)
        for name, value in module.logged_scalars:
            self.assertTrue(torch.isfinite(value).all(), f"{name} not finite: {value}")

    def test_never_halt_survival_stays_full(self):
        module = self.build_module(global_step=0, max_steps=3, threshold=1.0)
        callback = self.primed_callback(module, log_every_n_steps=1)

        self.drive_forward(module)
        callback.on_train_batch_end(FakeTrainer(0), module, None, None, batch_idx=0)

        scalars = dict(module.logged_scalars)
        self.assertEqual(
            scalars["recurrent.halting_model/halt/halted_fraction"].item(), 0.0
        )
        self.assertEqual(
            scalars["recurrent.halting_model/depth/step_count"].item(), 3.0
        )
        self.assertEqual(
            scalars["recurrent.halting_model/halt/saturation_fraction"].item(), 1.0
        )

    def test_halt_now_survival_drops_and_marks_halted(self):
        torch.manual_seed(0)
        module = self.build_module(global_step=0, max_steps=5, threshold=1e-6)
        callback = self.primed_callback(module, log_every_n_steps=1)

        self.drive_forward(module)
        callback.on_train_batch_end(FakeTrainer(0), module, None, None, batch_idx=0)

        scalars = dict(module.logged_scalars)
        self.assertEqual(
            scalars["recurrent.halting_model/halt/halted_fraction"].item(), 1.0
        )
        self.assertEqual(
            scalars["recurrent.halting_model/depth/step_count"].item(), 1.0
        )
        self.assertEqual(
            scalars["recurrent.halting_model/halt/saturation_fraction"].item(), 0.0
        )

    def test_cadence_gates_on_trainer_global_step(self):
        module = self.build_module()
        callback = self.primed_callback(module, log_every_n_steps=10)

        self.drive_forward(module)
        callback.on_train_batch_end(FakeTrainer(7), module, None, None, batch_idx=0)
        self.assertEqual(module.logged_scalars, [])

        callback.on_train_batch_end(FakeTrainer(20), module, None, None, batch_idx=3)
        self.assertTrue(module.logged_scalars)

    def test_visual_summaries_emit_histogram_and_heatmap(self):
        experiment = FakeExperiment()
        module = self.build_module(experiment=experiment, global_step=0)
        callback = self.primed_callback(module, log_every_n_steps=1)

        self.drive_forward(module)
        callback.on_train_batch_end(FakeTrainer(0), module, None, None, batch_idx=0)
        self.drive_forward(module)
        callback.on_train_batch_end(FakeTrainer(0), module, None, None, batch_idx=0)

        self.assertTrue(
            any("histogram/survival" in tag for tag, _, _ in experiment.histograms)
        )
        self.assertTrue(
            any("histogram/ponder_cost" in tag for tag, _, _ in experiment.histograms)
        )
        self.assertTrue(
            any("heatmap/survival" in tag for tag, _, _, _ in experiment.images)
        )
        _, image, _, dataformats = experiment.images[-1]
        self.assertEqual(dataformats, "CHW")
        self.assertEqual(image.dim(), 3)

    def test_visual_summaries_skip_empty_survival_histogram(self):
        experiment = StrictHistogramExperiment()
        module = self.build_module(experiment=experiment, global_step=0)
        callback = self.primed_callback(module, log_every_n_steps=1)

        callback.on_train_batch_end(FakeTrainer(0), module, None, None, batch_idx=0)

        self.assertEqual(experiment.histograms, [])
        self.assertEqual(experiment.images, [])

    def test_survival_history_is_bounded(self):
        experiment = FakeExperiment()
        module = self.build_module(experiment=experiment)
        callback = self.primed_callback(
            module,
            log_every_n_steps=1,
            history_size=2,
        )

        for global_step in range(3):
            module.global_step = global_step
            self.drive_forward(module)
            callback.on_train_batch_end(
                FakeTrainer(global_step),
                module,
                None,
                None,
                batch_idx=global_step,
            )

        history = callback._survival_history["recurrent.halting_model"]
        self.assertEqual(len(history), 2)
        for tensor in history.tensors:
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)

    def test_on_fit_end_restores_methods_and_clears_state(self):
        module = self.build_module()
        callback = self.primed_callback(module, log_every_n_steps=1)
        halting_model = module.recurrent.halting_model

        callback.on_fit_end(trainer=None, pl_module=module)

        self.assertNotIn("update_halting_state", halting_model.__dict__)
        self.assertNotIn("finalize_weighted_accumulation", halting_model.__dict__)
        self.assertIsNone(getattr(halting_model, "_usage_tracker", None))
        self.assertEqual(callback._halting_layers, [])
        self.assertEqual(callback._survival_history, {})

        # the layer still runs after the monitor detaches
        self.drive_forward(module)

    def test_manager_attach_is_idempotent(self):
        layer = self.build_recurrent_halting_layer()
        manager = HaltingUsageTrackerManager()
        first = manager.attach(layer.halting_model)
        second = manager.attach(layer.halting_model)
        self.assertIs(first, second)

    def test_tracker_handles_soft_state_without_halt_mask(self):
        tracker = HaltingUsageTracker()
        tracker.begin_forward()
        tracker.record_step(
            FakeSoftState(
                continuation_probability=torch.tensor([1.0, 0.5, 0.0]),
                accumulated_ponder_cost=torch.zeros(3),
            )
        )
        tracker.record_final(
            torch.tensor([0.4]),
            FakeSoftState(
                continuation_probability=torch.tensor([0.2, 0.1, 0.0]),
                accumulated_ponder_cost=torch.tensor([2.0, 3.0, 1.0]),
            ),
        )

        self.assertTrue(torch.isfinite(tracker.last_survival).all())
        self.assertEqual(tracker.last_step_count.item(), 1.0)
        self.assertEqual(tracker.last_halted_fraction.item(), 0.0)
        self.assertAlmostEqual(tracker.last_ponder_cost_mean.item(), 2.0, places=5)
        self.assertTrue(
            torch.equal(tracker.last_ponder_cost, torch.tensor([2.0, 3.0, 1.0]))
        )


if __name__ == "__main__":
    unittest.main()
