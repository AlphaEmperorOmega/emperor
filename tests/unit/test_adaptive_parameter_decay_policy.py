from __future__ import annotations

import copy
import unittest

import torch
from torch import Tensor, nn

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    BankExpansionFactorOptions,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDepthOptions,
    GeneratorDynamicBiasConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    TanhGatedDynamicBiasConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._biases.base import (
    DynamicBiasAbstract,
)
from emperor.augmentations.adaptive_parameters._decay import _DecayPolicy
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.augmentations.adaptive_parameters._weights.base import (
    DynamicWeightAbstract,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    RowLayout,
)
from emperor.linears import LinearLayerConfig
from emperor.nn import Module


def linear_stack_config(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=True,
            ),
        ),
    )


def single_weight_config(
    schedule: WeightDecayScheduleOptions,
    *,
    rate: float,
    warmup_batches: int,
) -> SingleModelDynamicWeightConfig:
    return SingleModelDynamicWeightConfig(
        input_dim=2,
        output_dim=2,
        generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
        decay_schedule=schedule,
        decay_rate=rate,
        decay_warmup_batches=warmup_batches,
        model_config=linear_stack_config(2, 2),
        normalization_option=WeightNormalizationOptions.DISABLED,
        normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
    )


def weight_configs() -> tuple:
    common = {
        "input_dim": 2,
        "output_dim": 3,
        "generator_depth": DynamicDepthOptions.DEPTH_OF_ONE,
        "decay_schedule": WeightDecayScheduleOptions.MULTIPLICATIVE,
        "decay_rate": 0.25,
        "decay_warmup_batches": 1,
        "model_config": linear_stack_config(2, 3),
    }
    square_common = {
        **common,
        "output_dim": 2,
        "model_config": linear_stack_config(2, 2),
    }
    return (
        SingleModelDynamicWeightConfig(
            **square_common,
            normalization_option=WeightNormalizationOptions.DISABLED,
            normalization_position_option=(WeightNormalizationPositionOptions.DISABLED),
        ),
        DualModelDynamicWeightConfig(
            **common,
            normalization_option=WeightNormalizationOptions.DISABLED,
            normalization_position_option=(WeightNormalizationPositionOptions.DISABLED),
        ),
        LowRankDynamicWeightConfig(
            **common,
            normalization_option=WeightNormalizationOptions.DISABLED,
        ),
        HypernetworkDynamicWeightConfig(
            **common,
            normalization_option=WeightNormalizationOptions.DISABLED,
        ),
        LayeredWeightedBankDynamicWeightConfig(
            **common,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
        ),
        SoftWeightedBankDynamicWeightConfig(
            **common,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
        ),
    )


def bias_config(
    config_type: type[DynamicBiasConfig],
    schedule: WeightDecayScheduleOptions,
    *,
    rate: float,
    warmup_batches: int = 0,
) -> DynamicBiasConfig:
    common = {
        "input_dim": 2,
        "output_dim": 3,
        "decay_schedule": schedule,
        "decay_rate": rate,
        "decay_warmup_batches": warmup_batches,
        "model_config": linear_stack_config(2, 3),
    }
    if config_type is WeightedBankDynamicBiasConfig:
        return config_type(
            **common,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
        )
    return config_type(**common)


def all_bias_config_types() -> tuple[type[DynamicBiasConfig], ...]:
    return (
        AdditiveDynamicBiasConfig,
        AffineTransformDynamicBiasConfig,
        MultiplicativeDynamicBiasConfig,
        SigmoidGatedDynamicBiasConfig,
        TanhGatedDynamicBiasConfig,
        GeneratorDynamicBiasConfig,
        WeightedBankDynamicBiasConfig,
    )


class ConstantGenerator(nn.Module):
    def __init__(self, output: Tensor):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, input_value: Tensor | LayerState) -> Tensor | LayerState:
        input_tensor = (
            input_value.hidden if isinstance(input_value, LayerState) else input_value
        )
        if input_tensor.size(0) != self.output.size(0):
            raise ValueError(
                f"Expected batch size {self.output.size(0)}, "
                f"received {input_tensor.size(0)}."
            )
        if isinstance(input_value, LayerState):
            input_value.hidden = self.output
            return input_value
        return self.output


class FailingGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, _input_value) -> Tensor:
        self.calls += 1
        raise RuntimeError("generator failed")


class AdaptiveParameterDecayPolicyTests(unittest.TestCase):
    def test_decay_policy_uses_composition_and_public_apply_method(self) -> None:
        self.assertTupleEqual(DynamicWeightAbstract.__bases__, (Module,))
        self.assertTupleEqual(DynamicBiasAbstract.__bases__, (Module,))
        self.assertTupleEqual(_DecayPolicy.__bases__, (object,))
        self.assertTrue(callable(_DecayPolicy.apply))
        for legacy_attribute in (
            "decay_schedule_option",
            "decay_rate",
            "decay_warmup_batches",
        ):
            self.assertNotIn(legacy_attribute, vars(DynamicWeightAbstract))
            self.assertNotIn(legacy_attribute, vars(DynamicBiasAbstract))
        self.assertFalse(hasattr(_DecayPolicy, "_compute_decay_factor_by_schedule"))
        self.assertTrue(
            callable(
                vars(_DecayPolicy)["_DecayPolicy__compute_decay_factor_by_schedule"]
            )
        )

    def test_all_leaves_keep_root_decay_buffer_topology(self) -> None:
        models = [config.build() for config in weight_configs()]
        models.extend(
            bias_config(
                config_type,
                WeightDecayScheduleOptions.MULTIPLICATIVE,
                rate=0.25,
                warmup_batches=1,
            ).build()
            for config_type in all_bias_config_types()
        )

        for model in models:
            with self.subTest(model_type=type(model).__name__):
                self.assertTupleEqual(
                    tuple(model._buffers),
                    ("decay_step", "warmup_step"),
                )
                self.assertTupleEqual(tuple(model.decay_step.shape), (1,))
                self.assertTupleEqual(tuple(model.warmup_step.shape), (1,))
                self.assertEqual(model.decay_step.dtype, torch.float32)
                self.assertEqual(model.warmup_step.dtype, torch.float32)
                self.assertEqual(model.decay_step.item(), 0.0)
                self.assertEqual(model.warmup_step.item(), 0.0)
                self.assertIn("decay_step", model.state_dict())
                self.assertIn("warmup_step", model.state_dict())
                self.assertNotIn("decay_step", dict(model.named_parameters()))
                self.assertNotIn("warmup_step", dict(model.named_parameters()))
                self.assertIsInstance(model._decay_policy, _DecayPolicy)
                self.assertNotIsInstance(model._decay_policy, nn.Module)
                self.assertFalse(hasattr(model._decay_policy, "cfg"))
                for legacy_attribute in (
                    "decay_schedule_option",
                    "decay_rate",
                    "decay_warmup_batches",
                ):
                    self.assertFalse(hasattr(model, legacy_attribute))
                self.assertFalse(
                    any(
                        module_name
                        for module_name, _module in model.named_modules()
                        if "decay" in module_name
                    )
                )

                model.double()
                self.assertEqual(model.decay_step.dtype, torch.float64)
                self.assertEqual(model.warmup_step.dtype, torch.float64)

        nested = AdaptiveLinearLayer(
            AdaptiveLinearLayerConfig(
                input_dim=2,
                output_dim=3,
                bias_flag=True,
                adaptive_augmentation_config=(
                    AdaptiveParameterAugmentationConfig(
                        bias_config=bias_config(
                            AdditiveDynamicBiasConfig,
                            WeightDecayScheduleOptions.MULTIPLICATIVE,
                            rate=0.25,
                            warmup_batches=1,
                        ),
                        grouping_scope=(AdaptiveParameterGroupingScopeOptions.DISABLED),
                    )
                ),
            )
        )
        nested_decay_keys = tuple(
            key
            for key in nested.state_dict()
            if key.endswith(("decay_step", "warmup_step"))
        )
        self.assertTupleEqual(
            nested_decay_keys,
            (
                "adaptive_behaviour.bias_model.decay_step",
                "adaptive_behaviour.bias_model.warmup_step",
            ),
        )

    def test_weight_and_additive_bias_share_exact_forward_decay_traces(self) -> None:
        schedules = (
            WeightDecayScheduleOptions.EXPONENTIAL,
            WeightDecayScheduleOptions.LINEAR,
            WeightDecayScheduleOptions.MULTIPLICATIVE,
        )
        dtypes = (torch.float32, torch.float64)
        rate = 0.25

        for schedule in schedules:
            for dtype in dtypes:
                with self.subTest(schedule=schedule, dtype=dtype):
                    weight = single_weight_config(
                        schedule,
                        rate=rate,
                        warmup_batches=1,
                    ).build()
                    bias = bias_config(
                        AdditiveDynamicBiasConfig,
                        schedule,
                        rate=rate,
                        warmup_batches=1,
                    ).build()
                    weight.model = ConstantGenerator(
                        torch.zeros(2, 1, 2),
                    )
                    bias.model = ConstantGenerator(torch.zeros(2, 3))
                    weight.to(dtype=dtype)
                    bias.to(dtype=dtype)

                    weight_params = torch.tensor(
                        [[1.0, -2.0], [3.0, -4.0]],
                        dtype=dtype,
                    )
                    bias_params = torch.tensor(
                        [1.0, -2.0, 3.0],
                        dtype=dtype,
                    )
                    inputs = torch.zeros(2, 2, dtype=dtype)
                    training_factors = (
                        1.0,
                        1.0,
                        self._factor(schedule, rate, step=1),
                    )

                    random_state = torch.random.get_rng_state()
                    for factor in training_factors:
                        weight_output = weight(weight_params, inputs)
                        bias_output = bias(bias_params, inputs)
                        torch.testing.assert_close(
                            weight_output,
                            weight_params.mul(factor).expand(2, -1, -1),
                        )
                        torch.testing.assert_close(
                            bias_output,
                            bias_params.mul(factor).expand(2, -1),
                        )
                        self.assertTrue(torch.equal(weight.decay_step, bias.decay_step))
                        self.assertTrue(
                            torch.equal(weight.warmup_step, bias.warmup_step)
                        )
                    self.assertTrue(
                        torch.equal(torch.random.get_rng_state(), random_state)
                    )

                    weight.eval()
                    bias.eval()
                    frozen_decay_step = weight.decay_step.clone()
                    evaluation_factor = self._factor(schedule, rate, step=2)
                    differentiable_weight = weight_params.clone().requires_grad_()
                    differentiable_bias = bias_params.clone().requires_grad_()
                    evaluation_weight = weight(differentiable_weight, inputs)
                    evaluation_bias = bias(differentiable_bias, inputs)
                    (evaluation_weight.sum() + evaluation_bias.sum()).backward()

                    torch.testing.assert_close(
                        evaluation_weight,
                        weight_params.mul(evaluation_factor).expand(2, -1, -1),
                    )
                    torch.testing.assert_close(
                        evaluation_bias,
                        bias_params.mul(evaluation_factor).expand(2, -1),
                    )
                    torch.testing.assert_close(
                        differentiable_weight.grad,
                        torch.full_like(
                            differentiable_weight,
                            2.0 * evaluation_factor,
                        ),
                    )
                    torch.testing.assert_close(
                        differentiable_bias.grad,
                        torch.full_like(
                            differentiable_bias,
                            2.0 * evaluation_factor,
                        ),
                    )
                    self.assertTrue(torch.equal(weight.decay_step, frozen_decay_step))
                    self.assertTrue(torch.equal(bias.decay_step, frozen_decay_step))

    def test_decay_config_overrides_resolve_before_policy_initialization(self) -> None:
        weight = single_weight_config(
            WeightDecayScheduleOptions.EXPONENTIAL,
            rate=0.1,
            warmup_batches=4,
        ).build(
            SingleModelDynamicWeightConfig(
                decay_schedule=WeightDecayScheduleOptions.LINEAR,
                decay_rate=0.2,
                decay_warmup_batches=2,
            )
        )
        bias = bias_config(
            AdditiveDynamicBiasConfig,
            WeightDecayScheduleOptions.EXPONENTIAL,
            rate=0.1,
            warmup_batches=4,
        ).build(
            AdditiveDynamicBiasConfig(
                decay_schedule=WeightDecayScheduleOptions.LINEAR,
                decay_rate=0.2,
                decay_warmup_batches=2,
            )
        )

        for model in (weight, bias):
            with self.subTest(model_type=type(model).__name__):
                self.assertEqual(
                    model._decay_policy.decay_schedule_option,
                    WeightDecayScheduleOptions.LINEAR,
                )
                self.assertEqual(model._decay_policy.decay_rate, 0.2)
                self.assertEqual(model._decay_policy.decay_warmup_batches, 2)
                self.assertEqual(model.decay_step.item(), 0.0)
                self.assertEqual(model.warmup_step.item(), 0.0)

    def test_policy_snapshots_config_and_owns_decay_attributes(self) -> None:
        source_configs = (
            single_weight_config(
                WeightDecayScheduleOptions.EXPONENTIAL,
                rate=0.1,
                warmup_batches=4,
            ),
            bias_config(
                AdditiveDynamicBiasConfig,
                WeightDecayScheduleOptions.EXPONENTIAL,
                rate=0.1,
                warmup_batches=4,
            ),
        )

        for source_config in source_configs:
            with self.subTest(config_type=type(source_config).__name__):
                model = source_config.build()

                source_config.decay_schedule = WeightDecayScheduleOptions.DISABLED
                source_config.decay_rate = 0.9
                source_config.decay_warmup_batches = 8

                self.assertEqual(
                    model._decay_policy.decay_schedule_option,
                    WeightDecayScheduleOptions.EXPONENTIAL,
                )
                self.assertEqual(model._decay_policy.decay_rate, 0.1)
                self.assertEqual(model._decay_policy.decay_warmup_batches, 4)

                model._decay_policy.decay_schedule_option = (
                    WeightDecayScheduleOptions.LINEAR
                )
                model._decay_policy.decay_rate = 0.2
                model._decay_policy.decay_warmup_batches = 2

                self.assertEqual(
                    model._decay_policy.decay_schedule_option,
                    WeightDecayScheduleOptions.LINEAR,
                )
                self.assertEqual(model._decay_policy.decay_rate, 0.2)
                self.assertEqual(model._decay_policy.decay_warmup_batches, 2)

    def test_non_additive_bias_leaves_keep_active_decay_as_a_no_op(self) -> None:
        non_additive_config_types = all_bias_config_types()[1:]
        inputs = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        bias_params = torch.tensor([2.0, -4.0, 1.0])

        for config_type in non_additive_config_types:
            with self.subTest(config_type=config_type.__name__):
                active = bias_config(
                    config_type,
                    WeightDecayScheduleOptions.MULTIPLICATIVE,
                    rate=0.5,
                ).build()
                disabled = bias_config(
                    config_type,
                    WeightDecayScheduleOptions.DISABLED,
                    rate=0.0,
                ).build()
                incompatible = disabled.load_state_dict(
                    copy.deepcopy(active.state_dict()),
                    strict=True,
                )
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])

                for _ in range(2):
                    torch.testing.assert_close(
                        active(bias_params, inputs),
                        disabled(bias_params, inputs),
                    )
                self.assertEqual(active.decay_step.item(), 0.0)
                self.assertEqual(active.warmup_step.item(), 0.0)

    def test_generator_failure_and_invalid_schedule_order_are_preserved(self) -> None:
        weight = single_weight_config(
            WeightDecayScheduleOptions.MULTIPLICATIVE,
            rate=0.25,
            warmup_batches=0,
        ).build()
        failing_weight_generator = FailingGenerator()
        weight.model = failing_weight_generator
        weight._decay_policy.decay_schedule_option = "invalid"

        with self.assertRaisesRegex(RuntimeError, "^generator failed$"):
            weight(torch.ones(2, 2), torch.ones(1, 2))
        self.assertEqual(failing_weight_generator.calls, 1)
        self.assertEqual(weight.decay_step.item(), 0.0)

        bias = bias_config(
            AdditiveDynamicBiasConfig,
            WeightDecayScheduleOptions.MULTIPLICATIVE,
            rate=0.25,
        ).build()
        failing_bias_generator = FailingGenerator()
        bias.model = failing_bias_generator
        bias._decay_policy.decay_schedule_option = "invalid"

        with self.assertRaisesRegex(
            ValueError,
            r"^Unsupported decay_schedule value: 'invalid'\.$",
        ):
            bias(torch.ones(3), torch.ones(1, 2))
        self.assertEqual(failing_bias_generator.calls, 0)
        self.assertEqual(bias.decay_step.item(), 0.0)

        bias._decay_policy.decay_schedule_option = (
            WeightDecayScheduleOptions.MULTIPLICATIVE
        )
        with self.assertRaisesRegex(RuntimeError, "^generator failed$"):
            bias(torch.ones(3), torch.ones(1, 2))
        self.assertEqual(failing_bias_generator.calls, 1)
        self.assertEqual(bias.decay_step.item(), 1.0)

    def test_active_additive_bias_model_and_adam_state_continue_after_restore(
        self,
    ) -> None:
        config = bias_config(
            AdditiveDynamicBiasConfig,
            WeightDecayScheduleOptions.MULTIPLICATIVE,
            rate=0.25,
            warmup_batches=1,
        )
        torch.manual_seed(37)
        source = config.build().double()
        source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
        bias_params = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
        first_input = torch.tensor(
            [[1.0, 2.0], [-0.5, 3.0]],
            dtype=torch.float64,
        )

        self._training_step(source, source_optimizer, bias_params, first_input)
        model_state = copy.deepcopy(source.state_dict())
        optimizer_state = copy.deepcopy(source_optimizer.state_dict())

        torch.manual_seed(101)
        restored = config.build().double()
        incompatible = restored.load_state_dict(model_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
        restored_optimizer.load_state_dict(optimizer_state)
        self.assertTupleEqual(
            tuple(source.state_dict()),
            tuple(restored.state_dict()),
        )
        self.assertTupleEqual(
            tuple(dict(source.named_parameters())),
            tuple(dict(restored.named_parameters())),
        )

        continuation_input = torch.tensor(
            [[0.25, -2.0], [3.0, 1.5]],
            dtype=torch.float64,
        )
        source_output = self._training_step(
            source,
            source_optimizer,
            bias_params,
            continuation_input,
        )
        restored_output = self._training_step(
            restored,
            restored_optimizer,
            bias_params,
            continuation_input,
        )
        torch.testing.assert_close(restored_output, source_output)
        for name, source_value in source.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[name],
                source_value,
            )
        self.assertEqual(
            source_optimizer.state_dict()["param_groups"],
            restored_optimizer.state_dict()["param_groups"],
        )
        for source_values, restored_values in zip(
            source_optimizer.state_dict()["state"].values(),
            restored_optimizer.state_dict()["state"].values(),
            strict=True,
        ):
            self.assertEqual(source_values.keys(), restored_values.keys())
            for key, source_value in source_values.items():
                restored_value = restored_values[key]
                if torch.is_tensor(source_value):
                    torch.testing.assert_close(restored_value, source_value)
                else:
                    self.assertEqual(restored_value, source_value)

    def test_grouped_forward_advances_decay_once_per_top_level_call(self) -> None:
        linear = AdaptiveLinearLayer(
            AdaptiveLinearLayerConfig(
                input_dim=2,
                output_dim=3,
                bias_flag=True,
                adaptive_augmentation_config=(
                    AdaptiveParameterAugmentationConfig(
                        bias_config=bias_config(
                            AdditiveDynamicBiasConfig,
                            WeightDecayScheduleOptions.MULTIPLICATIVE,
                            rate=0.25,
                        ),
                        grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
                        group_count=2,
                    )
                ),
            )
        )
        inputs = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 4.0], [2.0, -0.5]])
        row_layout = RowLayout.rows(
            4,
            context_sharing_restricted=False,
        )

        linear(inputs, row_layout=row_layout)
        self.assertEqual(
            linear.adaptive_behaviour.bias_model.decay_step.item(),
            1.0,
        )
        linear(inputs, row_layout=row_layout)
        self.assertEqual(
            linear.adaptive_behaviour.bias_model.decay_step.item(),
            2.0,
        )

    @staticmethod
    def _factor(
        schedule: WeightDecayScheduleOptions,
        rate: float,
        *,
        step: int,
    ) -> float:
        if schedule == WeightDecayScheduleOptions.EXPONENTIAL:
            return float(torch.exp(torch.tensor(-rate * step)))
        if schedule == WeightDecayScheduleOptions.LINEAR:
            return max(1.0 - rate * step, 0.0)
        if schedule == WeightDecayScheduleOptions.MULTIPLICATIVE:
            return (1.0 - rate) ** step
        raise AssertionError(f"Unsupported schedule in test: {schedule!r}.")

    @staticmethod
    def _training_step(
        model,
        optimizer,
        bias_params: Tensor,
        inputs: Tensor,
    ) -> Tensor:
        optimizer.zero_grad()
        output = model(bias_params, inputs)
        output.square().mean().backward()
        optimizer.step()
        return output.detach()


if __name__ == "__main__":
    unittest.main()
