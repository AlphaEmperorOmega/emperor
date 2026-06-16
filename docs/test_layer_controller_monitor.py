from emperor.base.layer.residual import ResidualConnectionOptions
import unittest

import torch

from docs._monitor_test_helpers import (
    CaptureLightningModule,
    NoExperimentLightningModule,
    TrainerStub,
    same_bound_method,
)
from emperor.base.layer import (
    Layer,
    LayerConfig,
    LayerStack,
    LayerStackConfig,
    LayerState,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.layer.monitor import LayerControllerMonitorCallback
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig


class TestLayerControllerMonitorCallback(unittest.TestCase):
    def linear_stack_config(self, dim: int = 4) -> LayerStackConfig:
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

    def layer(
        self,
        with_gate: bool = True,
        gate_option: LayerGateOptions | None = None,
        activation: ActivationOptions = ActivationOptions.TANH,
    ) -> Layer:
        return Layer(
            LayerConfig(
                input_dim=4,
                output_dim=4,
                activation=activation,
                residual_connection_option=ResidualConnectionOptions.RESIDUAL,
                dropout_probability=0.25,
                layer_norm_position=LayerNormPositionOptions.BEFORE,
                gate_config=(
                    GateConfig(
                        model_config=self.linear_stack_config(),
                        option=gate_option or LayerGateOptions.MULTIPLIER,
                        activation=ActivationOptions.SIGMOID,
                    )
                    if with_gate
                    else None
                ),
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    bias_flag=True,
                ),
            )
        )

    def state(self):
        return LayerState(hidden=torch.randn(3, 4))

    def test_rejects_non_positive_cadence(self):
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    LayerControllerMonitorCallback(log_every_n_steps=bad)

    def test_discovers_only_layer_modules(self):
        module = CaptureLightningModule(layer=self.layer(), other=torch.nn.Linear(4, 4))
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)

        names = [name for name, _ in callback._layer_modules]
        self.assertIn("layer", names)
        self.assertNotIn("other", names)
        callback.on_fit_end(TrainerStub(), module)

    def test_respects_global_step_cadence(self):
        layer = self.layer()
        module = CaptureLightningModule(layer=layer)
        callback = LayerControllerMonitorCallback(log_every_n_steps=2)
        callback.on_fit_start(TrainerStub(), module)

        module.global_step = 1
        layer(self.state())
        self.assertEqual(module.logged, [])

        module.global_step = 2
        layer(self.state())
        self.assertIn("layer/gate/output_mean", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_logs_expected_finite_scalar_tags(self):
        layer = self.layer()
        module = CaptureLightningModule(layer=layer)
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        layer(self.state())

        expected_tags = {
            "layer/gate/output_mean",
            "layer/gate/output_var",
            "layer/gate/positive_fraction",
            "layer/gate/saturation_fraction",
            "layer/gate/effective_mean",
            "layer/gate/effective_var",
            "layer/gate/effective_positive_fraction",
            "layer/gate/effective_saturation_fraction",
            "layer/residual/contribution_ratio",
            "layer/dropout/zero_fraction",
            "layer/layer_norm/output_mean",
            "layer/layer_norm/output_var",
            "layer/layer_norm/relative_delta_norm",
            "layer/activation/zero_fraction",
            "layer/activation/saturation_fraction",
        }
        self.assertTrue(expected_tags.issubset(set(module.logged_tags)))
        for tag in expected_tags:
            self.assertTrue(torch.isfinite(torch.as_tensor(module.logged_value(tag))).all(), tag)
        callback.on_fit_end(TrainerStub(), module)

    def test_skips_missing_layer_gate_metrics(self):
        layer = self.layer(with_gate=False)
        module = CaptureLightningModule(layer=layer)
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        layer(self.state())

        self.assertEqual(callback._hooked_gate_model_ids, set())
        self.assertFalse(any(tag.startswith("layer/gate/") for tag in module.logged_tags))
        callback.on_fit_end(TrainerStub(), module)

    def test_skips_disabled_activation_metrics(self):
        layer = self.layer(activation=ActivationOptions.DISABLED)
        original_activation = layer._Layer__maybe_apply_activation
        module = CaptureLightningModule(layer=layer)
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        self.assertTrue(
            same_bound_method(layer._Layer__maybe_apply_activation, original_activation)
        )
        layer(self.state())

        self.assertFalse(
            any(tag.startswith("layer/activation/") for tag in module.logged_tags)
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_logs_effective_gate_values_with_selected_gate_option(self):
        layer = self.layer(gate_option=LayerGateOptions.MULTIPLIER)
        gate_layer = layer.gate_model.model[0]
        with torch.no_grad():
            gate_layer.model.weight_params.zero_()
            gate_layer.model.bias_params.zero_()
        module = CaptureLightningModule(layer=layer)
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        layer(self.state())

        torch.testing.assert_close(
            torch.as_tensor(module.logged_value("layer/gate/output_mean")),
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            torch.as_tensor(module.logged_value("layer/gate/effective_mean")),
            torch.tensor(0.5),
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_runs_without_visual_experiment(self):
        layer = self.layer()
        module = NoExperimentLightningModule(layer=layer)
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        layer(self.state())

        self.assertIn("layer/gate/output_mean", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_shared_gate_module_registers_one_hook(self):
        stack = LayerStack(
            LayerStackConfig(
                input_dim=4,
                hidden_dim=4,
                output_dim=4,
                num_layers=3,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                shared_gate_config=GateConfig(
                    model_config=self.linear_stack_config(4),
                    option=LayerGateOptions.MULTIPLIER,
                    activation=ActivationOptions.SIGMOID,
                ),
                layer_config=LayerConfig(
                    input_dim=4,
                    output_dim=4,
                    activation=ActivationOptions.DISABLED,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=4,
                        output_dim=4,
                        bias_flag=True,
                    ),
                ),
            )
        )
        module = CaptureLightningModule(stack=stack)
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        stack(self.state())

        self.assertEqual(len(callback._hooks), 1)
        self.assertEqual(
            module.logged_tags.count("stack.layers.0/gate/output_mean"),
            len(stack),
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_restores_hooks_wrappers_and_clears_state_on_fit_end(self):
        layer = self.layer()
        original_activation = layer._Layer__maybe_apply_activation
        original_residual = layer._Layer__maybe_apply_residual_connection
        module = CaptureLightningModule(layer=layer)
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        self.assertIsNot(layer._Layer__maybe_apply_activation, original_activation)
        self.assertIsNot(
            layer._Layer__maybe_apply_residual_connection,
            original_residual,
        )
        self.assertGreater(len(callback._hooks), 0)

        callback.on_fit_end(TrainerStub(), module)

        self.assertTrue(
            same_bound_method(
                layer._Layer__maybe_apply_activation,
                original_activation,
            )
        )
        self.assertTrue(
            same_bound_method(
                layer._Layer__maybe_apply_residual_connection,
                original_residual,
            )
        )
        self.assertEqual(callback._layer_modules, [])
        self.assertEqual(callback._wrapped_methods, [])
        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._hooked_gate_model_ids, set())


if __name__ == "__main__":
    unittest.main()
