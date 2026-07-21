import unittest
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from emperor.halting import HaltingConfig
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerControllerMonitorCallback,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
    RecurrentLayerMonitorCallback,
    ResidualConfig,
    ResidualConnection,
    ResidualConnectionOptions,
)
from emperor.layers._composition.gate import LayerGate
from emperor.layers._monitoring.callbacks._hooks import _extract_hidden_tensor
from emperor.layers._monitoring.diagnostics import (
    _LayerNormTrackingContext,
    _RecurrentDiagnostics,
    _RecurrentObservation,
)
from emperor.layers._support import LayerModuleBase
from emperor.layers._validation.gate import LayerGateValidator
from emperor.linears import LinearLayerConfig
from emperor.memory import MemoryPositionOptions
from support.layers import (
    base_layer_config,
    configure_weighted_memory,
    linear_stack_config,
    recurrent_config,
    set_layer_identity,
    weighted_memory_config,
)
from support.monitor import CaptureLightningModule, TrainerStub, same_bound_method


class PositionedMemory(nn.Module):
    def __init__(self, position: MemoryPositionOptions) -> None:
        super().__init__()
        self.memory_position_option = position

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 3.0


class MemoryDispatchHarness(LayerModuleBase):
    def __init__(self, position: MemoryPositionOptions) -> None:
        super().__init__()
        self.memory_model = PositionedMemory(position)


class LayerBehavioralContractTests(unittest.TestCase):
    def test_strict_checkpoint_round_trip_preserves_optimizer_continuation(
        self,
    ) -> None:
        def stack_factory() -> LayerStack:
            return LayerStack(
                LayerStackConfig(
                    input_dim=2,
                    hidden_dim=2,
                    output_dim=2,
                    num_layers=2,
                    last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                    apply_output_pipeline_flag=True,
                    shared_gate_config=None,
                    shared_halting_config=None,
                    shared_memory_config=None,
                    layer_config=LayerConfig(
                        input_dim=2,
                        output_dim=2,
                        activation=ActivationOptions.TANH,
                        residual_config=ResidualConfig(
                            option=(ResidualConnectionOptions.WEIGHTED_BLEND)
                        ),
                        dropout_probability=0.0,
                        layer_norm_position=LayerNormPositionOptions.AFTER,
                        gate_config=GateConfig(
                            gate_dim=2,
                            option=LayerGateOptions.MULTIPLIER,
                            activation=ActivationOptions.SIGMOID,
                            model_config=linear_stack_config(2, bias_flag=True),
                        ),
                        halting_config=None,
                        memory_config=None,
                        layer_model_config=LinearLayerConfig(
                            input_dim=2,
                            output_dim=2,
                            bias_flag=True,
                        ),
                    ),
                )
            )

        def recurrent_factory() -> RecurrentLayer:
            return RecurrentLayer(
                RecurrentLayerConfig(
                    input_dim=2,
                    output_dim=2,
                    max_steps=2,
                    recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
                    block_config=LayerConfig(
                        input_dim=2,
                        output_dim=2,
                        activation=ActivationOptions.TANH,
                        residual_config=None,
                        dropout_probability=0.0,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        gate_config=None,
                        halting_config=None,
                        memory_config=None,
                        layer_model_config=LinearLayerConfig(
                            input_dim=2,
                            output_dim=2,
                            bias_flag=True,
                        ),
                    ),
                    gate_config=GateConfig(
                        gate_dim=2,
                        option=LayerGateOptions.ADDITION,
                        activation=ActivationOptions.TANH,
                        model_config=linear_stack_config(2, bias_flag=True),
                    ),
                    residual_config=ResidualConfig(
                        option=(ResidualConnectionOptions.WEIGHTED_BLEND)
                    ),
                    halting_config=None,
                    memory_config=None,
                )
            )

        first_batch = torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
        )
        continuation_batch = torch.tensor(
            [[-1.5, 0.75], [2.5, -0.25]],
        )

        for name, factory in (
            ("layer_stack", stack_factory),
            ("recurrent_layer", recurrent_factory),
        ):
            with self.subTest(name=name):
                torch.manual_seed(101)
                source = factory()
                source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)

                def train_step(model, optimizer, hidden):
                    optimizer.zero_grad()
                    output = model(LayerState(hidden=hidden.clone())).hidden
                    loss = output.square().mean()
                    loss.backward()
                    gradients = [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                    self.assertTrue(gradients)
                    self.assertTrue(
                        all(torch.isfinite(gradient).all() for gradient in gradients)
                    )
                    self.assertTrue(any(gradient.ne(0).any() for gradient in gradients))
                    optimizer.step()
                    return output.detach(), loss.detach()

                train_step(source, source_optimizer, first_batch)
                model_checkpoint = deepcopy(source.state_dict())
                optimizer_checkpoint = deepcopy(source_optimizer.state_dict())

                torch.manual_seed(999)
                restored = factory()
                restored_optimizer = torch.optim.Adam(
                    restored.parameters(),
                    lr=0.01,
                )
                incompatible = restored.load_state_dict(
                    model_checkpoint,
                    strict=True,
                )
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                restored_optimizer.load_state_dict(optimizer_checkpoint)

                for key, value in source.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value)

                source_output, source_loss = train_step(
                    source,
                    source_optimizer,
                    continuation_batch,
                )
                restored_output, restored_loss = train_step(
                    restored,
                    restored_optimizer,
                    continuation_batch,
                )

                torch.testing.assert_close(restored_output, source_output)
                torch.testing.assert_close(restored_loss, source_loss)
                for source_parameter, restored_parameter in zip(
                    source.parameters(),
                    restored.parameters(),
                    strict=True,
                ):
                    torch.testing.assert_close(
                        restored_parameter,
                        source_parameter,
                    )

    def test_feature_last_layers_preserve_non_contiguous_float64_inputs(
        self,
    ) -> None:
        def non_contiguous_input() -> torch.Tensor:
            base = torch.tensor(
                [
                    [[1.0, 9.0, -2.0, 8.0], [0.5, 7.0, 3.0, 6.0]],
                    [[-4.0, 5.0, 2.5, 4.0], [6.0, 3.0, -1.5, 2.0]],
                ],
                dtype=torch.float64,
            )
            hidden = base[..., ::2].requires_grad_(True)
            self.assertFalse(hidden.is_contiguous())
            return hidden

        layer = Layer(base_layer_config(2)).double()
        set_layer_identity(layer)
        recurrent = RecurrentLayer(recurrent_config()).double()

        for name, model in (("layer", layer), ("recurrent", recurrent)):
            with self.subTest(name=name):
                hidden = non_contiguous_input()
                output = model(LayerState(hidden=hidden)).hidden

                self.assertEqual(output.shape, (2, 2, 2))
                self.assertEqual(output.dtype, torch.float64)
                self.assertEqual(output.device, hidden.device)
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output, hidden)
                output.sum().backward()
                self.assertIsNotNone(hidden.grad)
                torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))

    def test_every_activation_executes_exact_math_through_real_layer(self) -> None:
        expected_functions = {
            ActivationOptions.RELU: F.relu,
            ActivationOptions.GELU: F.gelu,
            ActivationOptions.SIGMOID: torch.sigmoid,
            ActivationOptions.TANH: torch.tanh,
            ActivationOptions.LEAKY_RELU: F.leaky_relu,
            ActivationOptions.ELU: F.elu,
            ActivationOptions.SELU: F.selu,
            ActivationOptions.SOFTPLUS: F.softplus,
            ActivationOptions.SOFTSIGN: F.softsign,
            ActivationOptions.SILU: F.silu,
            ActivationOptions.MISH: F.mish,
        }
        inputs = torch.tensor(
            [[-2.0, 0.5], [1.5, -0.25]],
            dtype=torch.float64,
            requires_grad=True,
        )

        for option, expected_function in expected_functions.items():
            with self.subTest(option=option):
                layer = Layer(base_layer_config(2, activation=option)).double()
                set_layer_identity(layer)
                sample = inputs.detach().clone().requires_grad_(True)

                output = layer(LayerState(hidden=sample)).hidden
                expected = expected_function(sample)

                self.assertEqual(output.dtype, torch.float64)
                self.assertEqual(output.device, sample.device)
                self.assertEqual(output.shape, sample.shape)
                torch.testing.assert_close(output, expected)
                self.assertTrue(torch.isfinite(output).all())
                output.sum().backward()
                self.assertIsNotNone(sample.grad)
                self.assertTrue(torch.isfinite(sample.grad).all())
                self.assertTrue(sample.grad.ne(0).any())

        disabled = Layer(
            base_layer_config(2, activation=ActivationOptions.DISABLED)
        ).double()
        set_layer_identity(disabled)
        disabled_input = inputs.detach().clone()
        disabled_output = disabled(LayerState(hidden=disabled_input)).hidden
        torch.testing.assert_close(disabled_output, disabled_input)

    def test_per_layer_memory_applies_exact_weighted_equation_and_gradients(
        self,
    ) -> None:
        layer = Layer(
            base_layer_config(
                2,
                memory_config=weighted_memory_config(
                    2,
                    MemoryPositionOptions.AFTER_AFFINE,
                ),
            )
        )
        set_layer_identity(layer)
        configure_weighted_memory(layer.memory_model)
        hidden = torch.tensor([[1.0, -2.0], [0.5, 3.0]], requires_grad=True)

        output = layer(LayerState(hidden=hidden)).hidden

        torch.testing.assert_close(output, hidden * 1.75)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(hidden.grad.ne(0).all())
        gradients = [
            parameter.grad
            for parameter in layer.memory_model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(
            any(
                gradient is not None
                and torch.isfinite(gradient).all()
                and gradient.ne(0).any()
                for gradient in gradients
            )
        )

    def test_shared_memory_reuses_one_real_module_and_composes_each_layer(
        self,
    ) -> None:
        dim = 2
        config = LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            shared_gate_config=None,
            shared_halting_config=None,
            shared_memory_config=weighted_memory_config(dim),
            layer_config=base_layer_config(dim),
        )
        stack = LayerStack(config)
        for layer in stack:
            set_layer_identity(layer)
        shared_memory = stack[0].memory_model
        configure_weighted_memory(shared_memory)
        hidden = torch.tensor([[1.0, -2.0], [0.5, 3.0]], requires_grad=True)

        output = stack(LayerState(hidden=hidden)).hidden

        self.assertIsNotNone(shared_memory)
        self.assertTrue(all(layer.memory_model is shared_memory for layer in stack))
        torch.testing.assert_close(output, hidden * (1.75**2))
        output.sum().backward()
        memory_gradients = [
            parameter.grad
            for parameter in shared_memory.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(
            any(
                gradient is not None
                and torch.isfinite(gradient).all()
                and gradient.ne(0).any()
                for gradient in memory_gradients
            )
        )

    def test_memory_dispatch_applies_only_at_the_model_position(
        self,
    ) -> None:
        hidden = torch.tensor([[1.0, -2.0]])
        harness = MemoryDispatchHarness(MemoryPositionOptions.AFTER_AFFINE)

        before = harness._maybe_apply_memory_by_position(
            hidden,
            MemoryPositionOptions.BEFORE_AFFINE,
        )
        after = harness._maybe_apply_memory_by_position(
            hidden,
            MemoryPositionOptions.AFTER_AFFINE,
        )

        self.assertIs(before, hidden)
        torch.testing.assert_close(after, hidden * 3.0)

    def test_absent_residual_config_is_identity_and_corrupt_option_is_rejected(
        self,
    ) -> None:
        current = torch.tensor([[1.0, 2.0]])
        previous = torch.tensor([[3.0, 4.0]])
        layer = Layer(base_layer_config())

        result = layer._Layer__maybe_apply_residual_connection(current, previous)

        self.assertIsNone(layer.residual_config)
        self.assertIsNone(layer.residual_connection)
        self.assertIs(result, current)

        with self.assertRaisesRegex(
            TypeError,
            r"^ResidualConfig.option must be a ResidualConnectionOptions value,",
        ):
            ResidualConnection(ResidualConfig(option=object()))

    def test_layer_gate_rejects_corrupt_runtime_option_after_real_model_run(
        self,
    ) -> None:
        gate = LayerGate(
            GateConfig(
                gate_dim=2,
                option=LayerGateOptions.MULTIPLIER,
                activation=ActivationOptions.DISABLED,
                model_config=linear_stack_config(2),
            )
        )
        set_layer_identity(gate.model[0])
        gate.option = object()

        with self.assertRaisesRegex(
            ValueError,
            r"^Unsupported gate option .* for LayerGate\.$",
        ):
            gate(torch.tensor([[1.0, -2.0]]))

    def test_stack_rejects_corrupt_runtime_last_layer_bias_option(self) -> None:
        stack = LayerStack(
            LayerStackConfig(
                input_dim=2,
                hidden_dim=2,
                output_dim=2,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=base_layer_config(2),
            )
        )
        stack.last_layer_bias_option = object()

        with self.assertRaisesRegex(
            ValueError,
            r"^Unsupported last layer bias option .* for LayerStack\.$",
        ):
            stack._LayerStack__resolve_last_layer_bias_override()

    def test_layer_without_halting_runs_the_normal_pipeline(self) -> None:
        layer = Layer(base_layer_config(2))
        set_layer_identity(layer)
        hidden = torch.tensor([[1.0, -2.0]])
        state = LayerState(hidden=hidden, loss=torch.tensor(0.75))

        output = layer(state)

        torch.testing.assert_close(output.hidden, hidden)
        torch.testing.assert_close(output.loss, torch.tensor(0.75))
        self.assertIsNone(output.halting_state)

    def test_layer_gate_validator_rejects_wrong_cfg_and_activation_exactly(
        self,
    ) -> None:
        gate = LayerGate.__new__(LayerGate)
        nn.Module.__init__(gate)
        gate.cfg = object()
        with self.assertRaisesRegex(
            TypeError,
            r"^LayerGate cfg must be a GateConfig, got object\.$",
        ):
            LayerGateValidator.validate(gate)

        invalid_activation = GateConfig(
            gate_dim=2,
            option=LayerGateOptions.MULTIPLIER,
            activation=object(),
            model_config=linear_stack_config(2),
        )
        with self.assertRaisesRegex(
            TypeError,
            r"^GateConfig\.activation must be an ActivationOptions value or None, "
            r"got object\.$",
        ):
            invalid_activation.build()

    def test_gate_stack_missing_layer_config_reports_full_owner_path(self) -> None:
        gate_model_config = linear_stack_config(2)
        gate_model_config.layer_config = None
        config = GateConfig(
            gate_dim=2,
            option=LayerGateOptions.MULTIPLIER,
            activation=ActivationOptions.SIGMOID,
            model_config=gate_model_config,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"^gate_config\.model_config\.layer_config is required when "
            r"gate_config is provided$",
        ):
            config.build()

    def test_stack_shared_controller_validation_reports_exact_memory_errors(
        self,
    ) -> None:
        valid = LayerStackConfig(
            input_dim=2,
            hidden_dim=2,
            output_dim=2,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            shared_gate_config=None,
            shared_halting_config=None,
            shared_memory_config=None,
            layer_config=base_layer_config(2),
        )

        invalid_type = replace(valid)
        invalid_type.shared_gate_config = object()
        with self.assertRaisesRegex(
            TypeError,
            r"^shared_gate_config must be an instance of GateConfig for "
            r"LayerStackConfig, got object$",
        ):
            LayerStack(invalid_type)

        invalid_memory_type = replace(valid)
        invalid_memory_type.shared_memory_config = object()
        with self.assertRaisesRegex(
            TypeError,
            r"^shared_memory_config must be an instance of DynamicMemoryConfig "
            r"for LayerStackConfig, got object$",
        ):
            LayerStack(invalid_memory_type)

        conflicting = replace(valid)
        conflicting.shared_memory_config = weighted_memory_config(2)
        conflicting.layer_config = base_layer_config(
            2,
            memory_config=weighted_memory_config(2),
        )
        with self.assertRaisesRegex(
            ValueError,
            r"^shared_memory_config and layer_config\.memory_config are mutually "
            r"exclusive\.",
        ):
            LayerStack(conflicting)

        mismatched = replace(valid)
        mismatched.hidden_dim = 3
        mismatched.shared_memory_config = weighted_memory_config(2)
        with self.assertRaisesRegex(
            ValueError,
            r"^input_dim, hidden_dim, and output_dim must all be equal when "
            r"shared_memory_config is provided",
        ):
            LayerStack(mismatched)

    def test_recurrent_validator_rejects_abstract_halting_and_invalid_memory(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"^halting_config must be a concrete halting config for "
            r"RecurrentLayerConfig$",
        ):
            RecurrentLayer(recurrent_config(halting_config=HaltingConfig()))

        with self.assertRaisesRegex(
            TypeError,
            r"^memory_config must be an instance of DynamicMemoryConfig for "
            r"RecurrentLayerConfig, got object$",
        ):
            RecurrentLayer(recurrent_config(memory_config=object()))

    def test_layer_and_stack_reject_abstract_halting_configs(self) -> None:
        layer_config = base_layer_config(2)
        layer_config.halting_config = HaltingConfig()
        with self.assertRaisesRegex(
            ValueError,
            r"^halting_config must be a concrete halting config for LayerConfig$",
        ):
            Layer(layer_config)

        stack_config = LayerStackConfig(
            input_dim=2,
            hidden_dim=2,
            output_dim=2,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            shared_gate_config=None,
            shared_halting_config=HaltingConfig(),
            shared_memory_config=None,
            layer_config=base_layer_config(2),
        )
        with self.assertRaisesRegex(
            ValueError,
            r"^shared_halting_config must be a concrete halting config for "
            r"LayerStackConfig$",
        ):
            LayerStack(stack_config)

    def test_monitor_tensor_extraction_and_empty_diagnostics_are_exact(self) -> None:
        tensor = torch.tensor([1.0, 2.0])
        state = LayerState(hidden=tensor)

        self.assertIs(_extract_hidden_tensor(tensor), tensor)
        self.assertIs(_extract_hidden_tensor(state), tensor)
        self.assertIsNone(_extract_hidden_tensor(SimpleNamespace(hidden="bad")))
        self.assertIsNone(_RecurrentDiagnostics.calculate(_RecurrentObservation()))

    def test_recurrent_monitor_empty_observation_logs_only_zero_steps(self) -> None:
        recurrent = RecurrentLayer(recurrent_config())
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(
            log_every_n_steps=1,
            log_per_step_scalars=True,
        )
        callback.on_fit_start(TrainerStub(), module)

        callback._RecurrentLayerMonitorCallback__emit_observation(
            module,
            "recurrent",
            recurrent,
            _RecurrentObservation(),
        )

        self.assertEqual(module.logged_tags, ["recurrent/recurrent/actual_steps"])
        torch.testing.assert_close(
            torch.as_tensor(module.logged_value("recurrent/recurrent/actual_steps")),
            torch.tensor(0.0),
        )
        self.assertEqual(len(callback._delta_history["recurrent"]), 0)
        callback.on_fit_end(TrainerStub(), module)

    def test_recurrent_monitor_malformed_gate_output_is_ignored(
        self,
    ) -> None:
        recurrent = RecurrentLayer(recurrent_config())
        recurrent.recurrent_gate = SimpleNamespace(model=nn.Identity())
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        hook = callback._RecurrentLayerMonitorCallback__make_gate_hook(
            recurrent,
            module,
        )
        hook(nn.Identity(), (), SimpleNamespace(hidden="not-a-tensor"))
        self.assertEqual(callback._latest_gate_logits, {})

    def test_layer_monitor_malformed_hook_inputs_are_noops(
        self,
    ) -> None:
        module = CaptureLightningModule()
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)

        gate_hook = callback._LayerControllerMonitorCallback__make_gate_hook(
            "layer",
            nn.Identity(),
            module,
        )
        dropout_hook = callback._LayerControllerMonitorCallback__make_dropout_hook(
            "layer",
            module,
        )
        norm_hook = callback._LayerControllerMonitorCallback__make_layer_norm_hook(
            "layer",
            module,
        )
        before = list(module.logged)
        gate_hook(nn.Identity(), (), SimpleNamespace(hidden="bad"))
        dropout_hook(nn.Identity(), (), object())
        norm_hook(nn.Identity(), (), object())
        self.assertEqual(module.logged, before)

    def test_layer_monitor_zero_dropout_input_and_shape_changing_norm_are_safe(
        self,
    ) -> None:
        module = CaptureLightningModule()
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        zeros = torch.zeros(2, 3)
        dropout_hook = callback._LayerControllerMonitorCallback__make_dropout_hook(
            "layer",
            module,
        )

        dropout_hook(nn.Identity(), (zeros,), zeros)

        self.assertIn("layer/dropout/zero_fraction", module.logged_tags)
        self.assertNotIn(
            "layer/dropout/dropped_nonzero_fraction",
            module.logged_tags,
        )
        context = _LayerNormTrackingContext(
            pl_module=module,
            module_name="layer",
            input_values=torch.ones(2, 3),
            output_values=torch.ones(2, 4),
        )
        callback._LayerControllerMonitorCallback__track_layer_norm_diagnostics(context)
        self.assertIn("layer/layer_norm/output_mean", module.logged_tags)
        self.assertIn("layer/layer_norm/output_var", module.logged_tags)
        self.assertNotIn(
            "layer/layer_norm/relative_delta_norm",
            module.logged_tags,
        )

    def test_monitor_exception_cleanup_restores_real_layer_methods(self) -> None:
        layer = Layer(
            LayerConfig(
                input_dim=2,
                output_dim=2,
                activation=ActivationOptions.TANH,
                residual_config=ResidualConfig(
                    option=ResidualConnectionOptions.RESIDUAL
                ),
                dropout_probability=0.25,
                layer_norm_position=LayerNormPositionOptions.BEFORE,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=2,
                    output_dim=2,
                    bias_flag=True,
                ),
            )
        )
        module = CaptureLightningModule(layer=layer)
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        original_activation = layer._Layer__maybe_apply_activation
        original_residual = layer._Layer__maybe_apply_residual_connection
        callback.on_fit_start(TrainerStub(), module)

        callback.on_exception(
            TrainerStub(),
            module,
            RuntimeError("deliberate"),
        )

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
        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._wrapped_methods, [])

    def test_recurrent_monitor_exception_cleanup_restores_real_methods(self) -> None:
        recurrent = RecurrentLayer(recurrent_config())
        module = CaptureLightningModule(recurrent=recurrent)
        callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        original_forward = recurrent.forward
        original_controllers = recurrent._RecurrentLayer__run_controllers
        callback.on_fit_start(TrainerStub(), module)

        callback.on_exception(
            TrainerStub(),
            module,
            RuntimeError("deliberate"),
        )

        self.assertTrue(same_bound_method(recurrent.forward, original_forward))
        self.assertTrue(
            same_bound_method(
                recurrent._RecurrentLayer__run_controllers,
                original_controllers,
            )
        )
        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._wrapped_methods, [])
        self.assertEqual(callback._observations, {})
        self.assertEqual(callback._delta_history, {})
        self.assertEqual(callback._latest_gate_logits, {})


if __name__ == "__main__":
    unittest.main()
