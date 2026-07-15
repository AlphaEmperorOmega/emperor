import unittest
from dataclasses import dataclass

import torch
from emperor.base.layer import (
    LayerConfig,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.layer.monitor import RecurrentLayerMonitorCallback
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.linears.core.config import LinearLayerConfig

from support.monitor import (
    CaptureLightningModule,
    NoExperimentLightningModule,
    TrainerStub,
    orchestration_calls,
    same_bound_method,
)


@dataclass
class IncrementBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    increment: float | None = optional_field("Increment.")

    def _registry_owner(self) -> type:
        return IncrementBlock


class IncrementBlock(Module):
    def __init__(
        self,
        cfg: IncrementBlockConfig,
        overrides: IncrementBlockConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.increment = self.cfg.increment

    def forward(self, state: LayerState) -> LayerState:
        state.hidden = state.hidden + self.increment
        return state


class ConstantGate(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, state: LayerState) -> LayerState:
        return LayerState(hidden=torch.full_like(state.hidden, self.value))


class TestRecurrentLayerMonitorCallback(unittest.TestCase):
    def test_tracking_orchestration_lists_each_tracked_fact(self):
        cls = RecurrentLayerMonitorCallback
        orchestration = cls._RecurrentLayerMonitorCallback__track_recurrent_diagnostics

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_actual_steps",
                "__track_hidden_delta_mean",
                "__track_maximum_hidden_delta",
                "__track_final_hidden_delta",
                "__track_convergence_ratio",
                "__track_maximum_step_fraction",
                "__track_per_step_hidden_delta_mean",
                "__track_gate_open_mean",
                "__track_gate_open_fraction",
                "__track_gate_saturation_fraction",
                "__track_hidden_delta_history",
                "__track_hidden_delta_histogram",
                "__track_hidden_delta_heatmap",
            ),
        )

    def gate_config(self, dim: int = 4) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=dim,
                output_dim=dim,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    bias_flag=True,
                ),
            ),
        )

    def recurrent(
        self,
        with_gate: bool = True,
        gate_option: LayerGateOptions | None = None,
        gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID,
    ) -> RecurrentLayer:
        cfg = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            max_steps=3,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=IncrementBlockConfig(
                input_dim=4,
                output_dim=4,
                increment=0.25,
            ),
            gate_config=(
                GateConfig(
                    model_config=self.gate_config(),
                    option=gate_option or LayerGateOptions.MULTIPLIER,
                    activation=gate_activation,
                )
                if with_gate
                else None
            ),
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=None,
        )
        return RecurrentLayer(cfg)

    def state(self):
        return LayerState(hidden=torch.zeros(2, 4))

    def test_rejects_non_positive_cadence(self):
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RecurrentLayerMonitorCallback(log_every_n_steps=bad)

    def test_rejects_non_positive_history_size(self):
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RecurrentLayerMonitorCallback(history_size=bad)

    def test_discovers_only_recurrent_layers(self):
        module = CaptureLightningModule(
            recurrent=self.recurrent(), other=torch.nn.Linear(4, 4)
        )
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)

        self.assertEqual(set(callback._delta_history), {"recurrent"})
        callback.on_fit_end(TrainerStub(), module)

    def test_respects_global_step_cadence(self):
        recurrent = self.recurrent()
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=2)
        callback.on_fit_start(TrainerStub(), module)

        module.global_step = 1
        recurrent(self.state())
        self.assertEqual(module.logged, [])

        module.global_step = 2
        recurrent(self.state())
        self.assertIn("recurrent/recurrent/actual_steps", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_repeated_fit_start_replaces_existing_instrumentation(self):
        recurrent = self.recurrent()
        original_forward = recurrent.forward
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        callback.on_fit_start(TrainerStub(), module)
        recurrent(self.state())

        self.assertEqual(len(callback._wrapped_methods), 2)
        self.assertEqual(
            module.logged_tags.count("recurrent/recurrent/actual_steps"),
            1,
        )
        callback.on_fit_end(TrainerStub(), module)
        self.assertTrue(same_bound_method(recurrent.forward, original_forward))

    def test_logs_expected_finite_scalars(self):
        recurrent = self.recurrent()
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        recurrent(self.state())

        expected_tags = {
            "recurrent/recurrent/actual_steps",
            "recurrent/recurrent/hidden_delta_mean",
            "recurrent/recurrent/hidden_delta_max",
            "recurrent/recurrent/hidden_delta_final",
            "recurrent/recurrent/convergence_ratio",
            "recurrent/recurrent/max_step_fraction",
            "recurrent/recurrent/gate/open_mean",
            "recurrent/recurrent/gate/open_fraction",
            "recurrent/recurrent/gate/saturation_fraction",
        }
        self.assertEqual(set(module.logged_tags), expected_tags)
        self.assertNotIn(
            "recurrent/recurrent/preserved_halted_hidden_fraction",
            module.logged_tags,
        )
        for tag in expected_tags:
            self.assertTrue(
                torch.isfinite(torch.as_tensor(module.logged_value(tag))).all(), tag
            )
        callback.on_fit_end(TrainerStub(), module)

    def test_logs_per_step_scalars_when_enabled(self):
        recurrent = self.recurrent()
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(
            log_every_n_steps=1,
            log_per_step_scalars=True,
        )
        callback.on_fit_start(TrainerStub(), module)

        recurrent(self.state())

        step_tags = {
            f"recurrent/recurrent/step_{index}/hidden_delta_mean"
            for index in range(recurrent.max_steps)
        }
        self.assertTrue(step_tags.issubset(set(module.logged_tags)))
        for tag in step_tags:
            self.assertTrue(
                torch.isfinite(torch.as_tensor(module.logged_value(tag))).all(),
                tag,
            )
        callback.on_fit_end(TrainerStub(), module)

    def test_recurrent_gate_monitor_uses_selected_gate_transform(self):
        recurrent = self.recurrent(
            gate_option=LayerGateOptions.MULTIPLIER,
            gate_activation=ActivationOptions.TANH,
        )
        recurrent.recurrent_gate.model = ConstantGate(1.0)
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        recurrent(self.state())

        expected = torch.tanh(torch.tensor(1.0))
        torch.testing.assert_close(
            torch.as_tensor(module.logged_value("recurrent/recurrent/gate/open_mean")),
            expected,
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_skips_missing_recurrent_gate_metrics(self):
        recurrent = self.recurrent(with_gate=False)
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        recurrent(self.state())

        self.assertEqual(callback._hooks, [])
        self.assertIn("recurrent/recurrent/actual_steps", module.logged_tags)
        self.assertFalse(
            any(
                tag.startswith("recurrent/recurrent/gate/")
                for tag in module.logged_tags
            )
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_emits_histograms_and_images_when_experiment_supports_them(self):
        recurrent = self.recurrent()
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        recurrent(self.state())

        experiment = module.logger.experiment
        self.assertTrue(
            any(
                tag == "recurrent/recurrent/histogram/hidden_delta"
                for tag, _, _ in experiment.histograms
            )
        )
        self.assertTrue(
            any(
                tag == "recurrent/recurrent/heatmap/hidden_delta_by_step"
                for tag, _, _, _ in experiment.images
            )
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_skips_visual_summaries_without_experiment(self):
        recurrent = self.recurrent()
        module = NoExperimentLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        recurrent(self.state())

        self.assertIn("recurrent/recurrent/actual_steps", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_delta_history_is_bounded_and_detached(self):
        recurrent = self.recurrent()
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(
            log_every_n_steps=1,
            history_size=2,
        )
        callback.on_fit_start(TrainerStub(), module)

        for global_step in range(3):
            module.global_step = global_step
            recurrent(self.state())

        history = callback._delta_history["recurrent"]
        self.assertEqual(len(history), 2)
        for tensor in history.tensors:
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)
        callback.on_fit_end(TrainerStub(), module)

    def test_restores_wrappers_and_clears_state_on_fit_end(self):
        recurrent = self.recurrent()
        original_forward = recurrent.forward
        original_controller = recurrent._RecurrentLayer__run_controllers
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        self.assertIsNot(recurrent.forward, original_forward)
        self.assertIsNot(
            recurrent._RecurrentLayer__run_controllers,
            original_controller,
        )

        callback.on_fit_end(TrainerStub(), module)

        self.assertTrue(same_bound_method(recurrent.forward, original_forward))
        self.assertTrue(
            same_bound_method(
                recurrent._RecurrentLayer__run_controllers,
                original_controller,
            )
        )
        self.assertEqual(callback._wrapped_methods, [])
        self.assertEqual(callback._observations, {})


if __name__ == "__main__":
    unittest.main()
