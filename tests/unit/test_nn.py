from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Linear, Parameter, ReLU, Sequential

from emperor.config import ConfigBase, optional_field
from emperor.linears import LinearLayerConfig
from emperor.linears._layer import LinearLayer
from emperor.nn import Module


class _ParameterOwner(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = self._init_parameter_bank(
            (2, 3),
            initializer=torch.nn.init.zeros_,
        )


class _StepModule(Module):
    def __init__(
        self,
        auxiliary_loss: float | None,
        *,
        plot_progress: bool = False,
    ) -> None:
        super().__init__(plotProgress=plot_progress)
        self.scale = Parameter(torch.tensor(2.0))
        self.auxiliary_loss = auxiliary_loss
        self.lr = 0.25
        self.last_loss: Tensor | None = None

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor | None]:
        auxiliary_loss = (
            None
            if self.auxiliary_loss is None
            else inputs.new_tensor(self.auxiliary_loss)
        )
        return inputs * self.scale, auxiliary_loss

    def loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        self.last_loss = (predictions - targets).square().mean()
        return self.last_loss


class _TwoInputStepModule(Module):
    def __init__(self) -> None:
        super().__init__(plotProgress=False)
        self.scale = Parameter(torch.tensor(2.0))
        self.last_loss: Tensor | None = None

    def forward(
        self,
        left: Tensor,
        right: Tensor,
    ) -> tuple[Tensor, None]:
        return left * self.scale + right, None

    def loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        self.last_loss = (predictions - targets).square().mean()
        return self.last_loss


class _LazyModule(Module):
    def __init__(self) -> None:
        super().__init__(plotProgress=False)
        self.net = Sequential(torch.nn.LazyLinear(2))


@dataclass
class _ModuleConfig(ConfigBase):
    width: int | None = optional_field("Width override.")
    enabled: bool | None = optional_field("Boolean override.")
    child: ConfigBase | None = optional_field("Nested override.")


@dataclass
class _PlotTrainerState:
    train_batch_idx: int
    num_train_batches: int
    epoch: int
    num_val_batches: int


def _constant_linear_initializer(module: torch.nn.Module) -> None:
    if isinstance(module, Linear):
        with torch.no_grad():
            module.weight.fill_(2.0)
            if module.bias is not None:
                module.bias.fill_(1.0)


class NeuralNetworkFoundationTests(unittest.TestCase):
    def test_module_defaults_and_abstract_loss_contract(self) -> None:
        module = Module()

        self.assertEqual(module.plot_train_per_epoch, 2)
        self.assertEqual(module.plot_valid_per_epoch, 1)
        self.assertTrue(module.plotProgress)
        self.assertTrue(module.board.display)
        with self.assertRaises(NotImplementedError):
            module.loss(torch.tensor(1.0), torch.tensor(1.0))

    def test_forward_requires_net_and_delegates_exact_tensor_behavior(self) -> None:
        module = Module(plotProgress=False)

        with self.assertRaises(AssertionError) as error:
            module(torch.tensor([[1.0, 2.0]]))
        self.assertEqual(str(error.exception), "Neural network is defined")

        module.net = Linear(2, 1)
        with torch.no_grad():
            module.net.weight.copy_(torch.tensor([[3.0, -2.0]]))
            module.net.bias.copy_(torch.tensor([0.5]))

        output = module(torch.tensor([[2.0, 4.0], [-1.0, 3.0]]))

        torch.testing.assert_close(output, torch.tensor([[-1.5], [-8.5]]))

    def test_plot_disabled_is_a_strict_no_op(self) -> None:
        module = Module(plotProgress=False)

        module.plot("loss", torch.tensor(3.0, requires_grad=True), train=True)

        self.assertFalse(hasattr(module.board, "raw_points"))

    def test_plot_requires_attached_trainer_when_enabled(self) -> None:
        module = Module(plotProgress=True)
        module.board.display = False

        with self.assertRaises(RuntimeError) as error:
            module.plot("loss", torch.tensor(1.0), train=True)
        self.assertEqual(
            str(error.exception),
            "Module is not attached to a `Trainer`.",
        )

    def test_plot_rejects_non_boolean_training_mode(self) -> None:
        module = Module(plotProgress=True)
        module.board.display = False
        module._trainer = _PlotTrainerState(
            train_batch_idx=0,
            num_train_batches=1,
            epoch=0,
            num_val_batches=1,
        )

        for invalid_mode, type_name in ((None, "NoneType"), (1, "int")):
            with self.subTest(invalid_mode=invalid_mode):
                with self.assertRaises(TypeError) as error:
                    module.plot(
                        "loss",
                        torch.tensor(1.0),
                        train=invalid_mode,
                    )
                self.assertEqual(
                    str(error.exception),
                    f"train must be bool, got {type_name}",
                )

    def test_plot_records_exact_train_and_validation_cadence(self) -> None:
        module = Module(
            plot_train_per_epoch=2,
            plot_valid_per_epoch=2,
            plotProgress=True,
        )
        module.board.display = False
        module._trainer = _PlotTrainerState(
            train_batch_idx=1,
            num_train_batches=4,
            epoch=2,
            num_val_batches=4,
        )

        module.plot("loss", torch.tensor(3.5, requires_grad=True), train=True)
        self.assertEqual(module.board.data["train_loss"], [])
        self.assertEqual(len(module.board.raw_points["train_loss"]), 1)

        module._trainer.train_batch_idx = 2
        module.plot("loss", torch.tensor(4.5, requires_grad=True), train=True)
        module.plot("loss", torch.tensor(1.25, requires_grad=True), train=False)
        self.assertEqual(module.board.data["val_loss"], [])
        module.plot("loss", torch.tensor(2.75, requires_grad=True), train=False)

        train_point = module.board.data["train_loss"][0]
        validation_point = module.board.data["val_loss"][0]
        self.assertEqual((train_point.x, train_point.y.item()), (0.375, 4.0))
        self.assertEqual((validation_point.x, validation_point.y.item()), (3, 2.0))
        self.assertEqual(module.board.xlabel, "epoch")

    def test_training_step_adds_optional_auxiliary_loss_and_backpropagates(
        self,
    ) -> None:
        without_auxiliary = _StepModule(auxiliary_loss=None)
        with_auxiliary = _StepModule(auxiliary_loss=1.5)
        batch = (torch.tensor([2.0]), torch.tensor([1.0]))

        base_loss = without_auxiliary.training_step(batch)
        total_loss = with_auxiliary.training_step(batch)
        total_loss.backward()

        torch.testing.assert_close(base_loss, torch.tensor(9.0))
        torch.testing.assert_close(total_loss, torch.tensor(10.5))
        torch.testing.assert_close(with_auxiliary.scale.grad, torch.tensor(12.0))

    def test_training_and_validation_steps_plot_exact_loss_channel(self) -> None:
        batch = (torch.tensor([2.0]), torch.tensor([1.0]))
        training_module = _StepModule(auxiliary_loss=1.5, plot_progress=True)
        validation_module = _StepModule(auxiliary_loss=1.5, plot_progress=True)
        for module in (training_module, validation_module):
            module.board.display = False
            module._trainer = _PlotTrainerState(
                train_batch_idx=1,
                num_train_batches=1,
                epoch=2,
                num_val_batches=1,
            )

        training_loss = training_module.training_step(batch)
        validation_module.validation_step(batch)

        self.assertEqual(list(training_module.board.data), ["train_loss"])
        self.assertEqual(list(validation_module.board.data), ["val_loss"])
        torch.testing.assert_close(training_loss, torch.tensor(10.5))
        self.assertEqual(
            training_module.board.data["train_loss"][0].y.item(),
            10.5,
        )
        self.assertEqual(
            validation_module.board.data["val_loss"][0].y.item(),
            10.5,
        )

    def test_step_methods_use_all_inputs_and_final_batch_item_as_target(self) -> None:
        batch = (
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            torch.tensor([10.0]),
        )
        training_module = _TwoInputStepModule()
        validation_module = _TwoInputStepModule()
        test_module = _TwoInputStepModule()

        training_loss = training_module.training_step(batch)
        validation_module.validation_step(batch)
        test_module.test_step(batch)

        torch.testing.assert_close(training_loss, torch.tensor(9.0))
        assert validation_module.last_loss is not None
        assert test_module.last_loss is not None
        torch.testing.assert_close(validation_module.last_loss, torch.tensor(9.0))
        torch.testing.assert_close(test_module.last_loss, torch.tensor(9.0))

    def test_validation_and_test_steps_compute_both_auxiliary_branches(self) -> None:
        batch = (torch.tensor([2.0]), torch.tensor([1.0]))
        validation_with_auxiliary = _StepModule(auxiliary_loss=1.5)
        validation_without_auxiliary = _StepModule(auxiliary_loss=None)
        test_with_auxiliary = _StepModule(auxiliary_loss=1.5)
        test_without_auxiliary = _StepModule(auxiliary_loss=None)

        validation_result = validation_with_auxiliary.validation_step(batch)
        validation_without_auxiliary.validation_step(batch)
        test_result = test_without_auxiliary.test_step(batch)
        test_with_auxiliary.test_step(batch)

        self.assertIsNone(validation_result)
        self.assertIsNone(test_result)
        assert validation_with_auxiliary.last_loss is not None
        assert validation_without_auxiliary.last_loss is not None
        assert test_with_auxiliary.last_loss is not None
        assert test_without_auxiliary.last_loss is not None
        torch.testing.assert_close(
            validation_with_auxiliary.last_loss,
            torch.tensor(10.5),
        )
        torch.testing.assert_close(
            validation_without_auxiliary.last_loss,
            torch.tensor(9.0),
        )
        torch.testing.assert_close(test_with_auxiliary.last_loss, torch.tensor(10.5))
        torch.testing.assert_close(test_without_auxiliary.last_loss, torch.tensor(9.0))

    def test_configure_optimizers_builds_sgd_with_module_learning_rate(self) -> None:
        module = _StepModule(auxiliary_loss=None)

        optimizer = module.configure_optimizers()

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.25)
        self.assertEqual(optimizer.param_groups[0]["params"], [module.scale])

    def test_apply_init_materializes_lazy_net_and_applies_real_initializer(
        self,
    ) -> None:
        module_without_initializer = _LazyModule()
        module_with_initializer = _LazyModule()
        inputs = (torch.tensor([[1.0, 2.0, 3.0]]),)

        module_without_initializer.apply_init(inputs)
        module_with_initializer.apply_init(inputs, _constant_linear_initializer)

        self.assertIsInstance(module_without_initializer.net[0], Linear)
        output = module_with_initializer(*inputs)
        torch.testing.assert_close(output, torch.tensor([[13.0, 13.0]]))

    def test_initialize_parameters_matches_xavier_sequence_and_zeroes_biases(
        self,
    ) -> None:
        module = Module(plotProgress=False)
        direct_parameter = Parameter(torch.full((2, 3), 9.0))
        first = Linear(3, 2, bias=True)
        second = Linear(2, 2, bias=False)
        nested = Sequential(first, ReLU(), Sequential(second))

        torch.manual_seed(314)
        module._initialize_parameters(direct_parameter, nested)

        expected_parameter = torch.empty_like(direct_parameter)
        expected_first_weight = torch.empty_like(first.weight)
        expected_second_weight = torch.empty_like(second.weight)
        torch.manual_seed(314)
        torch.nn.init.xavier_uniform_(expected_parameter)
        torch.nn.init.xavier_uniform_(expected_first_weight)
        torch.nn.init.xavier_uniform_(expected_second_weight)

        torch.testing.assert_close(direct_parameter, expected_parameter)
        torch.testing.assert_close(first.weight, expected_first_weight)
        torch.testing.assert_close(first.bias, torch.zeros_like(first.bias))
        torch.testing.assert_close(second.weight, expected_second_weight)

    def test_override_config_preserves_base_and_applies_only_explicit_values(
        self,
    ) -> None:
        child = _ModuleConfig(width=4)
        base = _ModuleConfig(width=2, enabled=True, child=child)
        overrides = _ModuleConfig(
            width=None, enabled=False, child=_ModuleConfig(width=9)
        )
        module = Module(plotProgress=False)

        unchanged = module._override_config(base)
        resolved = module._override_config(base, overrides)
        copied_without_overrides = module._override_config(base, object())

        self.assertIs(unchanged, base)
        self.assertIsNot(resolved, base)
        self.assertEqual(resolved.width, 2)
        self.assertFalse(resolved.enabled)
        self.assertEqual(resolved.child, _ModuleConfig(width=9))
        self.assertIs(base.child, child)
        self.assertIsNot(copied_without_overrides, base)
        self.assertEqual(copied_without_overrides, base)
        self.assertIsNot(copied_without_overrides.child, base.child)

    def test_resolve_config_overrides_filters_unknown_keys_and_uses_defaults(
        self,
    ) -> None:
        config = _ModuleConfig(width=3, enabled=True)
        module = Module(plotProgress=False)

        overrides = module._resolve_config_overrides(
            config,
            width=8,
            enabled=None,
            unknown="ignored",
        )

        self.assertIsInstance(overrides, _ModuleConfig)
        self.assertEqual(overrides.width, 8)
        self.assertIsNone(overrides.enabled)
        self.assertIsNone(overrides.child)

    def test_build_from_config_handles_disabled_and_real_build_paths(self) -> None:
        module = Module(plotProgress=False)
        config = LinearLayerConfig(input_dim=2, output_dim=4, bias_flag=True)

        disabled = module._build_from_config(None, input_dim=3)
        built = module._build_from_config(
            config,
            input_dim=3,
            output_dim=2,
            bias_flag=False,
            unknown="ignored",
        )

        self.assertIsNone(disabled)
        self.assertIsInstance(built, LinearLayer)
        assert built is not None
        self.assertEqual(
            (built.input_dim, built.output_dim, built.bias_flag),
            (3, 2, False),
        )
        self.assertEqual(tuple(built.weight_params.shape), (3, 2))
        self.assertIsNone(built.bias_params)

    def test_resolve_main_config_prefers_explicit_sub_config_override(self) -> None:
        module = Module(plotProgress=False)
        main = _ModuleConfig(width=3)
        sub = _ModuleConfig(width=5)
        override = _ModuleConfig(width=7)

        self.assertIs(module._resolve_main_config(sub, main), main)

        sub.override_config = override
        self.assertIs(module._resolve_main_config(sub, main), override)

    def test_parameter_bank_preserves_owner_registration_and_gradients(self) -> None:
        owner = _ParameterOwner()

        self.assertEqual(list(owner.state_dict()), ["weights"])
        self.assertEqual(
            [name for name, _ in owner.named_parameters()],
            ["weights"],
        )
        torch.testing.assert_close(owner.weights, torch.zeros(2, 3))

        owner.weights.sum().backward()

        self.assertIsNotNone(owner.weights.grad)
        assert owner.weights.grad is not None
        self.assertTrue(torch.isfinite(owner.weights.grad).all().item())
        torch.testing.assert_close(owner.weights.grad, torch.ones(2, 3))

    def test_default_parameter_bank_initialization_is_deterministic_xavier(
        self,
    ) -> None:
        owner = Module(plotProgress=False)

        torch.manual_seed(91)
        parameter = owner._init_parameter_bank((2, 3))

        torch.manual_seed(91)
        expected = Parameter(torch.randn(2, 3))
        torch.nn.init.xavier_uniform_(expected)

        self.assertIsInstance(parameter, Parameter)
        self.assertTrue(parameter.requires_grad)
        self.assertEqual(parameter.dtype, torch.float32)
        self.assertEqual(parameter.device.type, "cpu")
        torch.testing.assert_close(parameter, expected)

    def test_parameter_bank_strict_state_round_trip_preserves_values(self) -> None:
        source = _ParameterOwner()
        target = _ParameterOwner()
        with torch.no_grad():
            source.weights.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        incompatible = target.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        torch.testing.assert_close(target.weights, source.weights)

    def test_construct_handles_none_default_and_configured_real_classes(self) -> None:
        module = Module(plotProgress=False)
        config = LinearLayerConfig(input_dim=2, output_dim=1, bias_flag=True)

        self.assertIsNone(module.construct(None))
        self.assertIsInstance(module.construct(ReLU), ReLU)
        configured = module.construct(LinearLayer, config)
        self.assertIsInstance(configured, LinearLayer)
        assert isinstance(configured, LinearLayer)
        self.assertIs(configured.cfg, config)


if __name__ == "__main__":
    unittest.main()
