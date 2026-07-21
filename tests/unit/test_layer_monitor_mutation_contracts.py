import math
import unittest
from types import SimpleNamespace

import torch

from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerControllerMonitorCallback,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
    RecurrentLayerMonitorCallback,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.layers._monitoring.diagnostics import (
    _LayerActivationTrackingContext,
    _LayerDropoutTrackingContext,
    _LayerGateTrackingContext,
    _LayerNormTrackingContext,
    _LayerResidualTrackingContext,
    _RecurrentDiagnostics,
    _RecurrentObservation,
    _RecurrentTrackingContext,
)
from emperor.layers._recurrent import _RecurrentState
from emperor.linears import LinearLayerConfig
from emperor.monitoring import MonitorEmissionPolicy, MonitorTensorHistory
from support.monitor import CaptureLightningModule, TrainerStub


def _linear_stack_config(dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=_layer_config(dim),
    )


def _layer_config(
    dim: int,
    *,
    activation: ActivationOptions = ActivationOptions.DISABLED,
    residual: ResidualConnectionOptions | None = None,
    dropout: float = 0.0,
    norm: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
    gate_config: GateConfig | None = None,
) -> LayerConfig:
    return LayerConfig(
        input_dim=dim,
        output_dim=dim,
        activation=activation,
        residual_config=None if residual is None else ResidualConfig(option=residual),
        dropout_probability=dropout,
        layer_norm_position=norm,
        gate_config=gate_config,
        halting_config=None,
        memory_config=None,
        layer_model_config=LinearLayerConfig(
            input_dim=dim,
            output_dim=dim,
            bias_flag=True,
        ),
    )


def _recurrent(
    dim: int = 2,
    *,
    max_steps: int = 5,
    gate_config: GateConfig | None = None,
) -> RecurrentLayer:
    recurrent = RecurrentLayer(
        RecurrentLayerConfig(
            input_dim=dim,
            output_dim=dim,
            max_steps=max_steps,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=_layer_config(dim),
            gate_config=gate_config,
            residual_config=None,
            halting_config=None,
            memory_config=None,
        )
    )
    with torch.no_grad():
        recurrent.block_model.model.weight_params.copy_(torch.eye(dim))
        recurrent.block_model.model.bias_params.zero_()
    return recurrent


class TestLayerMonitorMutationContracts(unittest.TestCase):
    def assert_logged_close(
        self,
        module: CaptureLightningModule,
        tag: str,
        expected: float | torch.Tensor,
    ) -> None:
        torch.testing.assert_close(
            torch.as_tensor(module.logged_value(tag)),
            torch.as_tensor(expected, dtype=torch.float32),
        )

    def test_configuration_contract_is_exact(self) -> None:
        default_recurrent_callback = RecurrentLayerMonitorCallback()
        self.assertEqual(default_recurrent_callback.log_every_n_steps, 100)
        self.assertEqual(default_recurrent_callback.history_size, 128)
        self.assertFalse(default_recurrent_callback.log_per_step_scalars)
        recurrent_callback = RecurrentLayerMonitorCallback(
            log_every_n_steps=7,
            history_size=5,
            log_per_step_scalars=True,
        )
        self.assertEqual(recurrent_callback.log_every_n_steps, 7)
        self.assertEqual(recurrent_callback.history_size, 5)
        self.assertTrue(recurrent_callback.log_per_step_scalars)
        self.assertEqual(recurrent_callback._hooks, [])
        self.assertEqual(recurrent_callback._wrapped_methods, [])
        self.assertEqual(recurrent_callback._observations, {})
        self.assertEqual(recurrent_callback._delta_history, {})
        self.assertEqual(recurrent_callback._latest_gate_logits, {})
        self.assertIsInstance(
            recurrent_callback._emission_policy,
            MonitorEmissionPolicy,
        )

        default_layer_callback = LayerControllerMonitorCallback()
        self.assertEqual(default_layer_callback.log_every_n_steps, 100)
        layer_callback = LayerControllerMonitorCallback(log_every_n_steps=7)
        self.assertEqual(layer_callback.log_every_n_steps, 7)
        self.assertEqual(layer_callback._hooks, [])
        self.assertEqual(layer_callback._wrapped_methods, [])
        self.assertEqual(layer_callback._hooked_gate_model_ids, set())

        for option_name, arguments in (
            ("log_every_n_steps", {"log_every_n_steps": 0}),
            ("history_size", {"history_size": 0}),
        ):
            with self.subTest(option_name=option_name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^{option_name} must be greater than 0\.$",
                ):
                    RecurrentLayerMonitorCallback(**arguments)
        with self.assertRaisesRegex(
            ValueError,
            r"^log_every_n_steps must be greater than 0\.$",
        ):
            LayerControllerMonitorCallback(log_every_n_steps=0)

    def test_layer_metric_equations_are_exact_at_boundaries(self) -> None:
        module = CaptureLightningModule()
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        raw_values = torch.tensor([-1.0, -0.99, 0.0, 0.99, 1.0])
        effective_values = torch.tensor([-0.5, 0.0, 0.01, 0.5, 0.99, 1.0])
        callback._LayerControllerMonitorCallback__track_gate_diagnostics(
            _LayerGateTrackingContext(
                pl_module=module,
                module_name="exact",
                raw_values=raw_values,
                effective_values=effective_values,
            )
        )

        self.assert_logged_close(module, "exact/gate/output_mean", raw_values.mean())
        self.assert_logged_close(
            module,
            "exact/gate/output_var",
            raw_values.var(unbiased=False),
        )
        self.assert_logged_close(module, "exact/gate/positive_fraction", 0.4)
        self.assert_logged_close(module, "exact/gate/saturation_fraction", 0.4)
        self.assert_logged_close(
            module,
            "exact/gate/effective_mean",
            effective_values.mean(),
        )
        self.assert_logged_close(
            module,
            "exact/gate/effective_var",
            effective_values.var(unbiased=False),
        )
        self.assert_logged_close(
            module,
            "exact/gate/effective_positive_fraction",
            4.0 / 6.0,
        )
        self.assert_logged_close(
            module,
            "exact/gate/effective_saturation_fraction",
            3.0 / 6.0,
        )

        callback._LayerControllerMonitorCallback__track_dropout_diagnostics(
            _LayerDropoutTrackingContext(
                pl_module=module,
                module_name="exact",
                input_values=torch.tensor([0.0, 1.0, 2.0, 0.0, -3.0]),
                output_values=torch.tensor([0.0, 0.0, 4.0, 0.0, 0.0]),
            )
        )
        self.assert_logged_close(module, "exact/dropout/zero_fraction", 0.8)
        self.assert_logged_close(
            module,
            "exact/dropout/dropped_nonzero_fraction",
            2.0 / 3.0,
        )

        callback._LayerControllerMonitorCallback__track_layer_norm_diagnostics(
            _LayerNormTrackingContext(
                pl_module=module,
                module_name="exact",
                input_values=torch.tensor([3.0, 4.0]),
                output_values=torch.tensor([0.0, 2.0]),
            )
        )
        self.assert_logged_close(module, "exact/layer_norm/output_mean", 1.0)
        self.assert_logged_close(module, "exact/layer_norm/output_var", 1.0)
        self.assert_logged_close(
            module,
            "exact/layer_norm/relative_delta_norm",
            math.sqrt(13.0) / 5.0,
        )

        callback._LayerControllerMonitorCallback__track_activation_diagnostics(
            _LayerActivationTrackingContext(
                pl_module=module,
                module_name="exact",
                activation_values=raw_values,
            )
        )
        self.assert_logged_close(module, "exact/activation/zero_fraction", 0.2)
        self.assert_logged_close(
            module,
            "exact/activation/saturation_fraction",
            0.4,
        )

        callback._LayerControllerMonitorCallback__track_residual_diagnostics(
            _LayerResidualTrackingContext(
                pl_module=module,
                module_name="exact",
                output_values=torch.tensor([4.0, 3.0]),
                input_values=torch.tensor([3.0, 1.0]),
                previous_values=torch.tensor([6.0, 8.0]),
            )
        )
        self.assert_logged_close(
            module,
            "exact/residual/contribution_ratio",
            math.sqrt(5.0) / 5.0,
        )
        self.assert_logged_close(
            module,
            "exact/residual/input_ratio",
            10.0 / math.sqrt(10.0),
        )

        callback._LayerControllerMonitorCallback__track_activation_diagnostics(
            _LayerActivationTrackingContext(
                pl_module=module,
                module_name="edge",
                activation_values=torch.tensor([0.0, 1.0, 1.0]),
            )
        )
        self.assert_logged_close(module, "edge/activation/zero_fraction", 1.0 / 3.0)
        callback._LayerControllerMonitorCallback__track_dropout_diagnostics(
            _LayerDropoutTrackingContext(
                pl_module=module,
                module_name="edge",
                input_values=torch.tensor([2.0, 0.0]),
                output_values=torch.tensor([0.0, 0.0]),
            )
        )
        self.assert_logged_close(
            module,
            "edge/dropout/dropped_nonzero_fraction",
            1.0,
        )
        callback._LayerControllerMonitorCallback__track_layer_norm_diagnostics(
            _LayerNormTrackingContext(
                pl_module=module,
                module_name="edge",
                input_values=torch.tensor([0.25]),
                output_values=torch.tensor([0.5]),
            )
        )
        self.assert_logged_close(
            module,
            "edge/layer_norm/relative_delta_norm",
            1.0,
        )
        callback._LayerControllerMonitorCallback__track_residual_diagnostics(
            _LayerResidualTrackingContext(
                pl_module=module,
                module_name="edge",
                output_values=torch.tensor([0.5]),
                input_values=torch.tensor([0.25]),
                previous_values=torch.tensor([0.125]),
            )
        )
        self.assert_logged_close(
            module,
            "edge/residual/contribution_ratio",
            0.5,
        )
        self.assert_logged_close(
            module,
            "edge/residual/input_ratio",
            0.5,
        )

    def test_layer_wrappers_preserve_keyword_calls_and_malformed_inputs(self) -> None:
        layer = Layer(
            _layer_config(
                2,
                activation=ActivationOptions.TANH,
                residual=ResidualConnectionOptions.RESIDUAL,
                dropout=0.25,
                norm=LayerNormPositionOptions.BEFORE,
            )
        )
        module = CaptureLightningModule(layer=layer)
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        activation_input = torch.tensor([[-1.0, 0.5]], requires_grad=True)
        activation_output = layer._Layer__maybe_apply_activation(input=activation_input)
        torch.testing.assert_close(activation_output, torch.tanh(activation_input))
        self.assertFalse(
            torch.as_tensor(
                module.logged_value("layer/activation/zero_fraction")
            ).requires_grad
        )

        current = torch.tensor([[3.0, 1.0]], requires_grad=True)
        previous = torch.tensor([[6.0, 8.0]], requires_grad=True)
        residual_output = layer._Layer__maybe_apply_residual_connection(
            input=current,
            prev_input=previous,
        )
        torch.testing.assert_close(residual_output, current + previous)
        self.assert_logged_close(
            module,
            "layer/residual/contribution_ratio",
            10.0 / math.sqrt(162.0),
        )
        self.assert_logged_close(
            module,
            "layer/residual/input_ratio",
            10.0 / math.sqrt(10.0),
        )
        positional_output = layer._Layer__maybe_apply_residual_connection(
            current,
            previous,
        )
        torch.testing.assert_close(positional_output, current + previous)
        mixed_output = layer._Layer__maybe_apply_residual_connection(
            current,
            prev_input=previous,
        )
        torch.testing.assert_close(mixed_output, current + previous)

        dropout_hook = callback._LayerControllerMonitorCallback__make_dropout_hook(
            "malformed",
            module,
        )
        norm_hook = callback._LayerControllerMonitorCallback__make_layer_norm_hook(
            "malformed",
            module,
        )
        before = list(module.logged)
        dropout_hook(torch.nn.Identity(), (torch.ones(2),), object())
        norm_hook(torch.nn.Identity(), (torch.ones(2),), object())
        self.assertEqual(module.logged, before)

        callback.on_fit_end(TrainerStub(), module)

    def test_recurrent_metric_equations_and_media_payloads_are_exact(self) -> None:
        recurrent = _recurrent(max_steps=5)
        module = CaptureLightningModule(recurrent=recurrent)
        module.global_step = 7
        callback = RecurrentLayerMonitorCallback(
            log_every_n_steps=1,
            history_size=4,
            log_per_step_scalars=True,
        )
        callback._delta_history["unit"] = MonitorTensorHistory(4)
        observation = _RecurrentObservation(
            step_deltas=[
                torch.tensor([1.0, 3.0]),
                torch.tensor([2.0, 6.0]),
            ],
            gate_values=[torch.tensor([0.0, 0.5, 0.6, 1.0, 0.01, 0.99, -0.1, 1.1])],
        )

        callback._RecurrentLayerMonitorCallback__emit_observation(
            module,
            "unit",
            recurrent,
            observation,
        )

        self.assert_logged_close(module, "unit/recurrent/actual_steps", 2.0)
        self.assert_logged_close(module, "unit/recurrent/hidden_delta_mean", 3.0)
        self.assert_logged_close(module, "unit/recurrent/hidden_delta_max", 6.0)
        self.assert_logged_close(module, "unit/recurrent/hidden_delta_final", 4.0)
        self.assert_logged_close(module, "unit/recurrent/convergence_ratio", 2.0)
        self.assert_logged_close(module, "unit/recurrent/max_step_fraction", 0.4)
        self.assert_logged_close(
            module,
            "unit/recurrent/step_0/hidden_delta_mean",
            2.0,
        )
        self.assert_logged_close(
            module,
            "unit/recurrent/step_1/hidden_delta_mean",
            4.0,
        )
        gate_values = observation.gate_values[0]
        self.assert_logged_close(
            module,
            "unit/recurrent/gate/open_mean",
            gate_values.mean(),
        )
        self.assert_logged_close(
            module,
            "unit/recurrent/gate/open_fraction",
            0.5,
        )
        self.assert_logged_close(
            module,
            "unit/recurrent/gate/saturation_fraction",
            0.5,
        )
        self.assertEqual(len(callback._delta_history["unit"]), 1)
        torch.testing.assert_close(
            callback._delta_history["unit"].tensors[0],
            torch.tensor([2.0, 4.0]),
        )

        experiment = module.logger.experiment
        self.assertEqual(len(experiment.histograms), 1)
        histogram_tag, histogram_values, histogram_step = experiment.histograms[0]
        self.assertEqual(histogram_tag, "unit/recurrent/histogram/hidden_delta")
        torch.testing.assert_close(
            histogram_values,
            torch.tensor([1.0, 3.0, 2.0, 6.0]),
        )
        self.assertEqual(histogram_step, 7)

        self.assertEqual(len(experiment.images), 1)
        image_tag, image, image_step, dataformats = experiment.images[0]
        self.assertEqual(image_tag, "unit/recurrent/heatmap/hidden_delta_by_step")
        torch.testing.assert_close(
            image,
            torch.tensor([[[0.5], [1.0]]]),
        )
        self.assertEqual(image_step, 7)
        self.assertEqual(dataformats, "CHW")

        low_delta_module = CaptureLightningModule()
        low_delta_callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        low_delta_callback._delta_history["low"] = MonitorTensorHistory(2)
        low_delta_callback._RecurrentLayerMonitorCallback__emit_observation(
            low_delta_module,
            "low",
            torch.nn.Identity(),
            _RecurrentObservation(
                step_deltas=[torch.tensor([0.5]), torch.tensor([1.0])]
            ),
        )
        self.assert_logged_close(
            low_delta_module,
            "low/recurrent/convergence_ratio",
            2.0,
        )
        self.assert_logged_close(
            low_delta_module,
            "low/recurrent/max_step_fraction",
            2.0,
        )

        stale_history = MonitorTensorHistory(2)
        stale_history.append(torch.tensor([1.0, 2.0]))
        stale_callback = RecurrentLayerMonitorCallback()
        stale_callback._delta_history["stale"] = stale_history
        stale_module = CaptureLightningModule()
        stale_context = _RecurrentTrackingContext(
            pl_module=stale_module,
            module_name="stale",
            metric_prefix="stale/recurrent",
            recurrent_layer=torch.nn.Identity(),
            metrics=None,
            device=torch.device("cpu"),
            experiment=stale_module.logger.experiment,
            global_step=0,
        )
        stale_callback._RecurrentLayerMonitorCallback__track_hidden_delta_heatmap(
            stale_context
        )
        self.assertEqual(stale_module.logger.experiment.images, [])

    def test_recurrent_emit_context_and_metric_devices_are_exact(self) -> None:
        class ContextCaptureCallback(RecurrentLayerMonitorCallback):
            def __init__(self) -> None:
                super().__init__()
                self.contexts: list[_RecurrentTrackingContext] = []

            def _RecurrentLayerMonitorCallback__track_recurrent_diagnostics(
                self,
                context: _RecurrentTrackingContext,
            ) -> None:
                self.contexts.append(context)

        recurrent = _recurrent(max_steps=3)
        experiment = object()
        owner = SimpleNamespace(
            device=torch.device("meta"),
            logger=SimpleNamespace(experiment=experiment),
            global_step=17,
        )
        callback = ContextCaptureCallback()
        observation = _RecurrentObservation(step_deltas=[torch.tensor([2.0])])
        callback._RecurrentLayerMonitorCallback__emit_observation(
            owner,
            "captured",
            recurrent,
            observation,
        )
        context = callback.contexts[-1]
        self.assertIs(context.pl_module, owner)
        self.assertEqual(context.module_name, "captured")
        self.assertEqual(context.metric_prefix, "captured/recurrent")
        self.assertIs(context.recurrent_layer, recurrent)
        self.assertIsNotNone(context.metrics)
        self.assertEqual(context.device, torch.device("meta"))
        self.assertIs(context.experiment, experiment)
        self.assertEqual(context.global_step, 17)

        default_owner = SimpleNamespace()
        callback._RecurrentLayerMonitorCallback__emit_observation(
            default_owner,
            "default",
            recurrent,
            observation,
        )
        default_context = callback.contexts[-1]
        self.assertEqual(default_context.device, torch.device("cpu"))
        self.assertIsNone(default_context.experiment)
        self.assertEqual(default_context.global_step, 0)

        class TensorCapture:
            def __init__(self) -> None:
                self.logged: list[tuple[str, torch.Tensor]] = []

            def log(self, tag: str, value: torch.Tensor) -> None:
                self.logged.append((tag, value))

        tensor_capture = TensorCapture()
        metrics = _RecurrentDiagnostics.calculate(observation)
        self.assertIsNotNone(metrics)
        device_context = _RecurrentTrackingContext(
            pl_module=tensor_capture,
            module_name="device",
            metric_prefix="device/recurrent",
            recurrent_layer=torch.nn.Identity(),
            metrics=metrics,
            device=torch.device("meta"),
            experiment=None,
            global_step=0,
        )
        callback._RecurrentLayerMonitorCallback__track_actual_steps(device_context)
        callback._RecurrentLayerMonitorCallback__track_maximum_step_fraction(
            device_context
        )
        self.assertEqual(
            [value.device.type for _, value in tensor_capture.logged],
            ["meta", "meta"],
        )

    def test_recurrent_recording_and_wrappers_preserve_values_and_keywords(
        self,
    ) -> None:
        gate_config = GateConfig(
            gate_dim=2,
            option=LayerGateOptions.MULTIPLIER,
            activation=ActivationOptions.TANH,
            model_config=_linear_stack_config(2),
        )
        recurrent = _recurrent(max_steps=1, gate_config=gate_config)
        gate_layer = recurrent.recurrent_gate.model[0]
        with torch.no_grad():
            gate_layer.model.weight_params.copy_(torch.eye(2))
            gate_layer.model.bias_params.zero_()

        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        keyword_output = recurrent.forward(
            state=LayerState(hidden=torch.tensor([[0.25, -0.5]]))
        )
        expected_candidate = torch.tensor([[0.25, -0.5]])
        torch.testing.assert_close(
            keyword_output.hidden,
            torch.tanh(expected_candidate) * expected_candidate,
        )

        observation = _RecurrentObservation()
        callback._observations[id(recurrent)] = observation
        recurrent_state = _RecurrentState(
            hidden=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            loss=None,
            context_state=LayerState(hidden=torch.zeros(2, 2)),
        )
        previous_hidden = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
        controller_output = recurrent._RecurrentLayer__run_controllers(
            recurrent_state=recurrent_state,
            previous_hidden=previous_hidden,
        )
        expected_hidden = torch.tanh(recurrent_state.hidden) * recurrent_state.hidden
        torch.testing.assert_close(controller_output.hidden, expected_hidden)
        expected_delta = (expected_hidden - previous_hidden).norm(dim=-1)
        torch.testing.assert_close(observation.step_deltas[-1], expected_delta)
        torch.testing.assert_close(
            observation.gate_values[-1],
            torch.tanh(recurrent_state.hidden).reshape(-1),
        )
        callback.on_fit_end(TrainerStub(), module)

        rectangular_recurrent = _recurrent(max_steps=1)
        rectangular_callback = RecurrentLayerMonitorCallback()
        rectangular_observation = _RecurrentObservation()
        previous = torch.tensor(
            [[0.0, 0.0], [1.0, 2.0], [3.0, 5.0]],
        )
        output = torch.tensor(
            [[1.0, 2.0], [4.0, 6.0], [8.0, 12.0]],
        )
        rectangular_callback._RecurrentLayerMonitorCallback__record_recurrent_step(
            rectangular_recurrent,
            rectangular_observation,
            previous,
            output,
        )
        torch.testing.assert_close(
            rectangular_observation.step_deltas[0],
            (output - previous).norm(dim=-1),
        )

        skipped_recurrent = _recurrent(max_steps=1, gate_config=gate_config)
        skipped_module = CaptureLightningModule(recurrent=skipped_recurrent)
        skipped_module.global_step = 1
        skipped_callback = RecurrentLayerMonitorCallback(log_every_n_steps=2)
        skipped_callback.on_fit_start(TrainerStub(), skipped_module)
        skipped_callback._observations[id(skipped_recurrent)] = _RecurrentObservation()
        skipped_callback._latest_gate_logits[id(skipped_recurrent)] = torch.ones(1)
        skipped_recurrent(LayerState(hidden=torch.ones(1, 2)))
        self.assertEqual(skipped_module.logged, [])
        self.assertNotIn(id(skipped_recurrent), skipped_callback._observations)
        self.assertEqual(skipped_callback._latest_gate_logits, {})
        self.assertFalse(
            skipped_callback._RecurrentLayerMonitorCallback__should_sample(
                skipped_module
            )
        )
        skipped_module.global_step = 2
        self.assertTrue(
            skipped_callback._RecurrentLayerMonitorCallback__should_sample(
                skipped_module
            )
        )
        skipped_callback.on_fit_end(TrainerStub(), skipped_module)

        multi_step_recurrent = _recurrent(max_steps=3)
        multi_step_module = CaptureLightningModule(recurrent=multi_step_recurrent)
        multi_step_callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        multi_step_callback.on_fit_start(TrainerStub(), multi_step_module)
        multi_step_recurrent(LayerState(hidden=torch.ones(1, 2)))
        self.assert_logged_close(
            multi_step_module,
            "recurrent/recurrent/max_step_fraction",
            1.0,
        )
        multi_step_callback.on_fit_end(TrainerStub(), multi_step_module)

        mixed_recurrent = _recurrent(max_steps=1)
        mixed_module = CaptureLightningModule(recurrent=mixed_recurrent)
        mixed_callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        mixed_callback.on_fit_start(TrainerStub(), mixed_module)
        mixed_observation = _RecurrentObservation()
        mixed_callback._observations[id(mixed_recurrent)] = mixed_observation
        mixed_state = _RecurrentState(
            hidden=torch.tensor([[2.0, 3.0]]),
            loss=None,
            context_state=LayerState(hidden=torch.zeros(1, 2)),
        )
        mixed_previous = torch.tensor([[1.0, 1.0]])
        mixed_recurrent._RecurrentLayer__run_controllers(
            mixed_state,
            previous_hidden=mixed_previous,
        )
        self.assertEqual(len(mixed_observation.step_deltas), 1)
        mixed_callback.on_fit_end(TrainerStub(), mixed_module)


if __name__ == "__main__":
    unittest.main()
