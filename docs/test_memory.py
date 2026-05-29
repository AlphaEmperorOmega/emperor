import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import Sequential

from emperor.base.layer import Layer, LayerConfig, LayerStackConfig, LayerState
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from emperor.memory.core import (
    AttentionDynamicMemory,
    DynamicMemoryAbstract,
    ElementWiseWeightedDynamicMemory,
    GatedResidualDynamicMemory,
    WeightedDynamicMemory,
)
from emperor.memory.options import MemoryPositionOptions


MEMORY_CASES = [
    (GatedResidualDynamicMemoryConfig, GatedResidualDynamicMemory),
    (WeightedDynamicMemoryConfig, WeightedDynamicMemory),
    (ElementWiseWeightedDynamicMemoryConfig, ElementWiseWeightedDynamicMemory),
    (AttentionDynamicMemoryConfig, AttentionDynamicMemory),
]


def make_layer_stack_config(
    input_dim: int = 4,
    hidden_dim: int = 8,
    output_dim: int = 4,
    num_layers: int = 1,
    bias_flag: bool = True,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            residual_flag=False,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            shared_halting_flag=False,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
            ),
        ),
    )


def make_memory_config(
    config_cls: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig,
    input_dim: int = 4,
    output_dim: int = 6,
    memory_position_option: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE,
    test_time_training_learning_rate: float | None = None,
    test_time_training_num_inner_steps: int | None = None,
    model_config: LayerStackConfig | None = None,
    num_memory_slots: int | None = 2,
) -> DynamicMemoryConfig:
    if model_config is None:
        model_config = make_layer_stack_config(
            input_dim=input_dim,
            hidden_dim=max(input_dim, output_dim),
            output_dim=output_dim,
        )
    kwargs = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "memory_position_option": memory_position_option,
        "test_time_training_learning_rate": test_time_training_learning_rate,
        "test_time_training_num_inner_steps": test_time_training_num_inner_steps,
        "model_config": model_config,
    }
    if config_cls is AttentionDynamicMemoryConfig:
        kwargs["num_memory_slots"] = num_memory_slots
    return config_cls(**kwargs)


def make_layer_config(
    input_dim: int = 4,
    output_dim: int = 4,
    activation: ActivationOptions = ActivationOptions.DISABLED,
    residual_flag: bool = False,
    dropout_probability: float = 0.0,
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
    gate_config: LayerStackConfig | None = None,
    halting_config=None,
    memory_config: DynamicMemoryConfig | None = None,
    shared_halting_flag: bool = False,
    layer_model_config: LinearLayerConfig | None = None,
) -> LayerConfig:
    if memory_config is None:
        memory_config = make_memory_config(
            input_dim=input_dim,
            output_dim=output_dim,
        )
    if layer_model_config is None:
        layer_model_config = LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=True,
        )
    return LayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
        residual_flag=residual_flag,
        dropout_probability=dropout_probability,
        layer_norm_position=layer_norm_position,
        gate_config=gate_config,
        halting_config=halting_config,
        memory_config=memory_config,
        shared_halting_flag=shared_halting_flag,
        layer_model_config=layer_model_config,
    )


def memory_dim(
    position: MemoryPositionOptions,
    input_dim: int,
    output_dim: int,
) -> int:
    if position == MemoryPositionOptions.BEFORE_AFFINE:
        return input_dim
    return output_dim


def assert_layer_stack_shape(
    test_case: unittest.TestCase,
    model: Layer | Sequential,
    input_dim: int,
    output_dim: int,
) -> None:
    if isinstance(model, Layer):
        test_case.assertEqual(model.input_dim, input_dim)
        test_case.assertEqual(model.output_dim, output_dim)
        test_case.assertEqual(model.model.weight_params.shape, (input_dim, output_dim))
        return

    test_case.assertIsInstance(model, Sequential)
    test_case.assertEqual(model[0].input_dim, input_dim)
    test_case.assertEqual(model[-1].output_dim, output_dim)


class IdentityModule(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class ScaleModule(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.factor


class AddConstantModule(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.value


class ConstantLastDimModule(nn.Module):
    def __init__(self, values: list[float] | torch.Tensor):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.values.to(device=inputs.device, dtype=inputs.dtype)
        view_shape = (1,) * (inputs.ndim - 1) + (values.numel(),)
        return values.reshape(view_shape).expand(*inputs.shape[:-1], values.numel())


class RepeatSlotsModule(nn.Module):
    def __init__(self, multipliers: list[float]):
        super().__init__()
        self.multipliers = multipliers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        slots = [inputs * multiplier for multiplier in self.multipliers]
        return torch.cat(slots, dim=-1)


@dataclass
class InvalidLayerStackConfig(LayerStackConfig):
    def build(self, overrides: LayerStackConfig | None = None) -> nn.Module:
        return nn.Identity()


class TestMemoryHandlers(unittest.TestCase):
    def preset(
        self,
        config_cls: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig,
        input_dim: int = 4,
        output_dim: int = 6,
        memory_position_option: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE,
        test_time_training_learning_rate: float | None = None,
        test_time_training_num_inner_steps: int | None = None,
        model_config: LayerStackConfig | None = None,
        num_memory_slots: int | None = 2,
    ) -> DynamicMemoryConfig:
        return make_memory_config(
            config_cls=config_cls,
            input_dim=input_dim,
            output_dim=output_dim,
            memory_position_option=memory_position_option,
            test_time_training_learning_rate=test_time_training_learning_rate,
            test_time_training_num_inner_steps=test_time_training_num_inner_steps,
            model_config=model_config,
            num_memory_slots=num_memory_slots,
        )

    def memory_cases(self):
        return MEMORY_CASES

    def test_config_build_returns_registry_owner_for_each_leaf_config(self):
        for config_cls, model_cls in self.memory_cases():
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(config_cls=config_cls)
                model = cfg.build()

                self.assertIsInstance(model, cfg._registry_owner())
                self.assertIsInstance(model, model_cls)
                self.assertIsInstance(model, DynamicMemoryAbstract)

    def test_abstract_config_build_raises(self):
        cfg = DynamicMemoryConfig(
            input_dim=4,
            output_dim=6,
            memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
            model_config=make_layer_stack_config(),
        )

        with self.assertRaises(ValueError):
            cfg.build()

    def test_config_build_applies_overrides(self):
        for config_cls, _ in self.memory_cases():
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=4,
                    output_dim=6,
                    memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
                )
                override_kwargs = {
                    "input_dim": 8,
                    "output_dim": 10,
                    "memory_position_option": MemoryPositionOptions.BEFORE_AFFINE,
                }
                if config_cls is AttentionDynamicMemoryConfig:
                    override_kwargs["num_memory_slots"] = 3
                overrides = config_cls(**override_kwargs)

                model = cfg.build(overrides=overrides)

                self.assertEqual(model.input_dim, 8)
                self.assertEqual(model.output_dim, 10)
                self.assertEqual(
                    model.memory_position_option,
                    MemoryPositionOptions.BEFORE_AFFINE,
                )
                self.assertEqual(model.memory_dim, 8)
                if isinstance(model, AttentionDynamicMemory):
                    self.assertEqual(model.num_memory_slots, 3)

    def test_partial_overrides_keep_unset_base_fields(self):
        for config_cls, _ in self.memory_cases():
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=4,
                    output_dim=6,
                    memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
                )
                overrides = config_cls(input_dim=8)

                model = cfg.build(overrides=overrides)

                self.assertEqual(model.input_dim, 8)
                self.assertEqual(model.output_dim, cfg.output_dim)
                self.assertEqual(
                    model.memory_position_option,
                    cfg.memory_position_option,
                )
                self.assertEqual(model.memory_dim, cfg.output_dim)
                if isinstance(model, AttentionDynamicMemory):
                    self.assertEqual(model.num_memory_slots, cfg.num_memory_slots)

    def test_init_stores_config_attributes_and_child_model_shapes(self):
        input_dim = 4
        output_dim = 6
        for config_cls, model_cls in self.memory_cases():
            for position in MemoryPositionOptions:
                with self.subTest(config_cls=config_cls.__name__, position=position):
                    cfg = self.preset(
                        config_cls=config_cls,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        memory_position_option=position,
                    )
                    model = cfg.build()
                    dim = memory_dim(position, input_dim, output_dim)

                    self.assertIsInstance(model, model_cls)
                    self.assertEqual(model.input_dim, cfg.input_dim)
                    self.assertEqual(model.output_dim, cfg.output_dim)
                    self.assertEqual(
                        model.memory_position_option,
                        cfg.memory_position_option,
                    )
                    self.assertEqual(model.memory_dim, dim)
                    self.assertEqual(model.model_config, cfg.model_config)
                    self.assertFalse(model.test_time_training_flag)
                    self.assertIsNone(model.memory_decoder)

                    if isinstance(model, AttentionDynamicMemory):
                        self.assertEqual(model.num_memory_slots, cfg.num_memory_slots)
                        assert_layer_stack_shape(
                            self,
                            model.memory_model,
                            dim,
                            cfg.num_memory_slots * dim,
                        )
                        assert_layer_stack_shape(self, model.query_model, dim, dim)
                        assert_layer_stack_shape(self, model.key_model, dim, dim)
                        assert_layer_stack_shape(self, model.value_model, dim, dim)
                        assert_layer_stack_shape(self, model.output_model, dim, dim)
                        assert_layer_stack_shape(
                            self,
                            model.memory_gate_model,
                            dim * 2,
                            dim,
                        )
                    else:
                        assert_layer_stack_shape(self, model.memory_model, dim, dim)
                        if isinstance(model, GatedResidualDynamicMemory):
                            assert_layer_stack_shape(
                                self,
                                model.memory_gate_model,
                                dim * 2,
                                dim,
                            )
                        elif isinstance(model, WeightedDynamicMemory):
                            assert_layer_stack_shape(
                                self,
                                model.memory_weight_model,
                                dim * 2,
                                2,
                            )
                        else:
                            assert_layer_stack_shape(
                                self,
                                model.memory_weight_model,
                                dim * 2,
                                dim,
                            )

    def test_init_with_ttt_builds_decoder_models(self):
        input_dim = 4
        output_dim = 6
        for config_cls, _ in self.memory_cases():
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    test_time_training_learning_rate=0.01,
                    test_time_training_num_inner_steps=2,
                )
                model = cfg.build()
                dim = output_dim

                self.assertTrue(model.test_time_training_flag)
                self.assertEqual(model.test_time_training_learning_rate, 0.01)
                self.assertEqual(model.test_time_training_num_inner_steps, 2)
                self.assertIsNotNone(model.memory_decoder)
                if isinstance(model, AttentionDynamicMemory):
                    assert_layer_stack_shape(
                        self,
                        model.memory_decoder,
                        cfg.num_memory_slots * dim,
                        dim,
                    )
                else:
                    assert_layer_stack_shape(self, model.memory_decoder, dim, dim)

    def test_forward_shape_and_type_for_all_variants_and_positions(self):
        input_dim = 4
        output_dim = 6
        for config_cls, _ in self.memory_cases():
            for position in MemoryPositionOptions:
                with self.subTest(config_cls=config_cls.__name__, position=position):
                    dim = memory_dim(position, input_dim, output_dim)
                    cfg = self.preset(
                        config_cls=config_cls,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        memory_position_option=position,
                    )
                    model = cfg.build()
                    model.eval()
                    inputs = torch.randn(2, 3, dim)

                    output = model(inputs)

                    self.assertIsInstance(output, torch.Tensor)
                    self.assertEqual(output.shape, inputs.shape)

    def test_deterministic_output_in_eval_mode(self):
        input_dim = 4
        output_dim = 6
        for config_cls, _ in self.memory_cases():
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                model = cfg.build()
                model.eval()
                inputs = torch.randn(2, output_dim)

                output_1 = model(inputs)
                output_2 = model(inputs)

                torch.testing.assert_close(output_1, output_2)

    def test_attention_num_memory_slots_validation(self):
        for num_memory_slots in [1, 3]:
            with self.subTest(num_memory_slots=num_memory_slots):
                cfg = self.preset(
                    config_cls=AttentionDynamicMemoryConfig,
                    num_memory_slots=num_memory_slots,
                )
                model = cfg.build()

                self.assertEqual(model.num_memory_slots, num_memory_slots)

        for num_memory_slots in [None, 0, -2]:
            with self.subTest(num_memory_slots=num_memory_slots):
                cfg = self.preset(
                    config_cls=AttentionDynamicMemoryConfig,
                    num_memory_slots=num_memory_slots,
                )
                with self.assertRaises(ValueError):
                    cfg.build()

        cfg = self.preset(
            config_cls=AttentionDynamicMemoryConfig,
            num_memory_slots="two",
        )
        with self.assertRaises(TypeError):
            cfg.build()

    def test_non_attention_configs_do_not_accept_memory_slots(self):
        for config_cls, _ in self.memory_cases()[:-1]:
            with self.subTest(config_cls=config_cls.__name__):
                with self.assertRaises(TypeError):
                    config_cls(
                        input_dim=4,
                        output_dim=6,
                        memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
                        model_config=make_layer_stack_config(),
                        num_memory_slots=2,
                    )

    def test_init_raises_on_invalid_config_values(self):
        invalid_cases = [
            (
                "missing_model_config",
                GatedResidualDynamicMemoryConfig(
                    input_dim=4,
                    output_dim=6,
                    memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
                ),
                ValueError,
            ),
            (
                "zero_input_dim",
                self.preset(input_dim=0),
                ValueError,
            ),
            (
                "negative_output_dim",
                self.preset(output_dim=-1),
                ValueError,
            ),
            (
                "invalid_memory_position_type",
                self.preset(memory_position_option="after"),
                TypeError,
            ),
            (
                "invalid_model_config_type",
                self.preset(model_config="invalid"),
                TypeError,
            ),
        ]

        for name, cfg, error_type in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaises(error_type):
                    cfg.build()

    def test_init_raises_when_generator_builds_wrong_model_type(self):
        invalid_model_config = InvalidLayerStackConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=make_layer_stack_config().layer_config,
        )
        cfg = self.preset(model_config=invalid_model_config)

        with self.assertRaises(TypeError):
            cfg.build()

    def test_output_matches_gated_residual_closed_form(self):
        dim = 4
        model = self.preset(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        ).build()
        model.memory_model = ScaleModule(2.0)
        model.memory_gate_model = ConstantLastDimModule([0.0] * dim)
        inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

        output = model(inputs)

        expected = inputs + 0.5 * (inputs * 2.0)
        torch.testing.assert_close(output, expected)

    def test_output_matches_weighted_closed_form(self):
        dim = 4
        model = self.preset(
            config_cls=WeightedDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        ).build()
        model.memory_model = ScaleModule(3.0)
        model.memory_weight_model = ConstantLastDimModule([0.0, 0.0])
        inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

        output = model(inputs)

        expected = (inputs + inputs * 3.0) * 0.5
        torch.testing.assert_close(output, expected)

    def test_output_matches_element_wise_weighted_closed_form(self):
        dim = 4
        model = self.preset(
            config_cls=ElementWiseWeightedDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        ).build()
        model.memory_model = ScaleModule(3.0)
        model.memory_weight_model = ConstantLastDimModule([0.0] * dim)
        inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

        output = model(inputs)

        expected = (inputs + inputs * 3.0) * 0.5
        torch.testing.assert_close(output, expected)

    def test_output_matches_attention_closed_form(self):
        dim = 3
        model = self.preset(
            config_cls=AttentionDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
            num_memory_slots=2,
        ).build()
        model.memory_model = RepeatSlotsModule([1.0, 2.0])
        model.query_model = IdentityModule()
        model.key_model = IdentityModule()
        model.value_model = IdentityModule()
        model.output_model = IdentityModule()
        model.memory_gate_model = ConstantLastDimModule([0.0] * dim)
        inputs = torch.tensor([[1.0, 2.0, -1.0]])

        output = model(inputs)

        memory_bank = torch.stack([inputs, inputs * 2.0], dim=-2)
        query = inputs.unsqueeze(-2)
        attention_scores = torch.matmul(query, memory_bank.transpose(-2, -1))
        attention_scores = attention_scores / (dim**0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_memory = torch.matmul(attention_weights, memory_bank).squeeze(-2)
        expected = inputs + 0.5 * attended_memory
        torch.testing.assert_close(output, expected)

    def test_gradients_flow_to_parameters_and_inputs(self):
        input_dim = 4
        output_dim = 6
        for config_cls, _ in self.memory_cases():
            for position in MemoryPositionOptions:
                with self.subTest(config_cls=config_cls.__name__, position=position):
                    dim = memory_dim(position, input_dim, output_dim)
                    cfg = self.preset(
                        config_cls=config_cls,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        memory_position_option=position,
                    )
                    model = cfg.build()
                    inputs = torch.randn(2, dim, requires_grad=True)

                    output = model(inputs)
                    output.sum().backward()
                    gradients = [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    nonzero_gradients = [
                        gradient
                        for gradient in gradients
                        if gradient is not None and torch.any(gradient.abs() > 0)
                    ]

                    self.assertTrue(len(nonzero_gradients) > 0)
                    self.assertIsNotNone(inputs.grad)
                    self.assertTrue(torch.any(inputs.grad.abs() > 0))

    def test_forward_raises_on_invalid_inputs(self):
        dim = 6
        for config_cls, _ in self.memory_cases():
            with self.subTest(config_cls=config_cls.__name__, error="type"):
                model = self.preset(config_cls=config_cls, output_dim=dim).build()
                with self.assertRaises(TypeError):
                    model([1.0, 2.0])

            with self.subTest(config_cls=config_cls.__name__, error="rank"):
                model = self.preset(config_cls=config_cls, output_dim=dim).build()
                with self.assertRaises(ValueError):
                    model(torch.randn(dim))

            with self.subTest(config_cls=config_cls.__name__, error="shape"):
                model = self.preset(config_cls=config_cls, output_dim=dim).build()
                with self.assertRaises(ValueError):
                    model(torch.randn(2, dim + 1))


class TestLayerMemoryIntegration(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        activation: ActivationOptions = ActivationOptions.DISABLED,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        gate_config: LayerStackConfig | None = None,
        halting_config=None,
        memory_config: DynamicMemoryConfig | None = None,
        shared_halting_flag: bool = False,
        layer_model_config: LinearLayerConfig | None = None,
    ) -> LayerConfig:
        return make_layer_config(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            residual_flag=residual_flag,
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=gate_config,
            halting_config=halting_config,
            memory_config=memory_config,
            shared_halting_flag=shared_halting_flag,
            layer_model_config=layer_model_config,
        )

    def test_layer_config_builds_memory_model_from_leaf_config(self):
        for config_cls, model_cls in MEMORY_CASES:
            with self.subTest(config_cls=config_cls.__name__):
                memory_config = make_memory_config(
                    config_cls=config_cls,
                    input_dim=4,
                    output_dim=4,
                )
                cfg = self.preset(memory_config=memory_config)
                layer = Layer(cfg)

                self.assertIsInstance(
                    layer.memory_model,
                    memory_config._registry_owner(),
                )
                self.assertIsInstance(layer.memory_model, model_cls)

    def test_layer_applies_memory_only_at_configured_position(self):
        dim = 4
        inputs = torch.zeros(2, dim)

        cases = [
            (MemoryPositionOptions.BEFORE_AFFINE, torch.full((2, dim), 2.0)),
            (MemoryPositionOptions.AFTER_AFFINE, torch.full((2, dim), 1.0)),
        ]
        for position, expected in cases:
            with self.subTest(position=position):
                memory_config = make_memory_config(
                    config_cls=GatedResidualDynamicMemoryConfig,
                    input_dim=dim,
                    output_dim=dim,
                    memory_position_option=position,
                )
                cfg = self.preset(
                    input_dim=dim,
                    output_dim=dim,
                    memory_config=memory_config,
                )
                layer = Layer(cfg)
                layer.model = ScaleModule(2.0)
                layer.memory_model = AddConstantModule(1.0)

                output = layer(LayerState(hidden=inputs))

                torch.testing.assert_close(output.hidden, expected)


if __name__ == "__main__":
    unittest.main()
