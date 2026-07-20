import unittest
from dataclasses import dataclass
from unittest import mock

import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from emperor.halting import HaltingHiddenStateModeOptions, StickBreakingConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,
)
from emperor.memory._base import DynamicMemoryAbstract
from emperor.memory._variants.attention import AttentionDynamicMemory
from emperor.memory._variants.element_wise_weighted import (
    ElementWiseWeightedDynamicMemory,
)
from emperor.memory._variants.gated_residual import GatedResidualDynamicMemory
from emperor.memory._variants.weighted import WeightedDynamicMemory

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
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
            ),
        ),
    )


def make_halting_config(input_dim: int = 4) -> StickBreakingConfig:
    return StickBreakingConfig(
        input_dim=input_dim,
        threshold=0.99,
        dropout_probability=0.0,
        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        halting_gate_config=make_layer_stack_config(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=2,
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
    residual_connection_option: ResidualConnectionOptions | None = None,
    dropout_probability: float = 0.0,
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
    gate_config: LayerStackConfig | None = None,
    halting_config=None,
    memory_config: DynamicMemoryConfig | None = None,
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
        residual_config=None
        if residual_connection_option is None
        else ResidualConfig(option=residual_connection_option),
        dropout_probability=dropout_probability,
        layer_norm_position=layer_norm_position,
        gate_config=gate_config,
        halting_config=halting_config,
        memory_config=memory_config,
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
    model: Layer | LayerStack,
    input_dim: int,
    output_dim: int,
) -> None:
    if isinstance(model, Layer):
        test_case.assertEqual(model.input_dim, input_dim)
        test_case.assertEqual(model.output_dim, output_dim)
        test_case.assertEqual(model.model.weight_params.shape, (input_dim, output_dim))
        return

    test_case.assertIsInstance(model, LayerStack)
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


class DynamicMemoryEvaluationHarness(LightningModule):
    def __init__(self, memory: DynamicMemoryAbstract):
        super().__init__()
        self.memory = memory
        self.last_output = None
        self.evaluation_inference_modes = []
        self.evaluation_grad_modes = []
        self.saw_sanity_validation = False

    def _evaluation_step(self, batch: tuple[torch.Tensor]) -> torch.Tensor:
        (inputs,) = batch
        self.evaluation_inference_modes.append(
            torch.is_inference_mode_enabled()
        )
        self.evaluation_grad_modes.append(torch.is_grad_enabled())
        output = self.memory(inputs)
        self.last_output = output
        return output.square().mean()

    def validation_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        if self.trainer.sanity_checking:
            self.saw_sanity_validation = True
        return self._evaluation_step(batch)

    def test_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._evaluation_step(batch)

    def predict_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._evaluation_step(batch)

    def training_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        (inputs,) = batch
        return self.memory(inputs).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


@dataclass
class InvalidLayerStackConfig(LayerStackConfig):
    def build(self, overrides: LayerStackConfig | None = None) -> nn.Module:
        return nn.Identity()


@dataclass
class FrozenLayerStackConfig(LayerStackConfig):
    def build(self, overrides: LayerStackConfig | None = None) -> LayerStack:
        model = super().build(overrides)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model


class TestMemoryHandlers(unittest.TestCase):
    def preset(
        self,
        config_cls: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig,
        input_dim: int = 4,
        output_dim: int = 6,
        memory_position_option: MemoryPositionOptions = (
            MemoryPositionOptions.AFTER_AFFINE
        ),
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

    def test_ttt_config_requires_learning_rate_and_inner_steps_together(self):
        cases = [
            ("missing_inner_steps", 0.01, None),
            ("missing_learning_rate", None, 1),
        ]

        for name, learning_rate, num_inner_steps in cases:
            with self.subTest(name=name):
                cfg = self.preset(
                    test_time_training_learning_rate=learning_rate,
                    test_time_training_num_inner_steps=num_inner_steps,
                )

                with self.assertRaises(ValueError):
                    cfg.build()

    def test_ttt_learning_rate_must_be_positive(self):
        for learning_rate in [0.0, -0.01]:
            with self.subTest(learning_rate=learning_rate):
                cfg = self.preset(
                    test_time_training_learning_rate=learning_rate,
                    test_time_training_num_inner_steps=1,
                )

                with self.assertRaises(ValueError):
                    cfg.build()

    def test_ttt_learning_rate_must_be_finite(self):
        for learning_rate in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(learning_rate=learning_rate):
                cfg = self.preset(
                    test_time_training_learning_rate=learning_rate,
                    test_time_training_num_inner_steps=1,
                )

                with self.assertRaises(ValueError):
                    cfg.build()

    def test_ttt_inner_steps_must_be_positive_int(self):
        invalid_cases = [
            ("zero", 0, ValueError),
            ("negative", -1, ValueError),
            ("float", 1.5, TypeError),
            ("bool", True, TypeError),
        ]

        for name, num_inner_steps, error_type in invalid_cases:
            with self.subTest(name=name):
                cfg = self.preset(
                    test_time_training_learning_rate=0.01,
                    test_time_training_num_inner_steps=num_inner_steps,
                )

                with self.assertRaises(error_type):
                    cfg.build()

    def test_memory_generator_rejects_nested_controllers(self):
        nested_memory_config = make_memory_config(input_dim=4, output_dim=4)
        nested_halting_config = make_halting_config(input_dim=4)
        cases = [
            (
                "layer_gate",
                lambda model_config: setattr(
                    model_config.layer_config,
                    "gate_config",
                    make_layer_stack_config(),
                ),
            ),
            (
                "layer_halting",
                lambda model_config: setattr(
                    model_config.layer_config,
                    "halting_config",
                    nested_halting_config,
                ),
            ),
            (
                "layer_memory",
                lambda model_config: setattr(
                    model_config.layer_config,
                    "memory_config",
                    nested_memory_config,
                ),
            ),
            (
                "shared_memory",
                lambda model_config: setattr(
                    model_config,
                    "shared_memory_config",
                    nested_memory_config,
                ),
            ),
            (
                "shared_halting",
                lambda model_config: setattr(
                    model_config,
                    "shared_halting_config",
                    nested_halting_config,
                ),
            ),
        ]

        for name, configure in cases:
            with self.subTest(name=name):
                model_config = make_layer_stack_config()
                configure(model_config)
                cfg = self.preset(model_config=model_config)

                with self.assertRaises(ValueError):
                    cfg.build()

    def test_ttt_forward_shape_for_supported_generators(self):
        input_dim = 4
        output_dim = 6
        for config_cls, _ in self.memory_cases():
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    test_time_training_learning_rate=0.01,
                    test_time_training_num_inner_steps=1,
                )
                model = cfg.build()
                model.eval()
                inputs = torch.randn(2, output_dim)

                output = model(inputs)

                self.assertEqual(output.shape, inputs.shape)

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
                "bool_input_dim",
                self.preset(input_dim=True),
                TypeError,
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
            (
                "partial_ttt_learning_rate",
                self.preset(test_time_training_learning_rate=0.01),
                ValueError,
            ),
            (
                "partial_ttt_inner_steps",
                self.preset(test_time_training_num_inner_steps=1),
                ValueError,
            ),
            (
                "zero_ttt_learning_rate",
                self.preset(
                    test_time_training_learning_rate=0.0,
                    test_time_training_num_inner_steps=1,
                ),
                ValueError,
            ),
            (
                "negative_ttt_learning_rate",
                self.preset(
                    test_time_training_learning_rate=-0.1,
                    test_time_training_num_inner_steps=1,
                ),
                ValueError,
            ),
            (
                "invalid_ttt_learning_rate_type",
                self.preset(
                    test_time_training_learning_rate="fast",
                    test_time_training_num_inner_steps=1,
                ),
                TypeError,
            ),
            (
                "zero_ttt_inner_steps",
                self.preset(
                    test_time_training_learning_rate=0.01,
                    test_time_training_num_inner_steps=0,
                ),
                ValueError,
            ),
            (
                "invalid_ttt_inner_steps_type",
                self.preset(
                    test_time_training_learning_rate=0.01,
                    test_time_training_num_inner_steps=1.5,
                ),
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

    def test_ttt_rejects_generator_without_trainable_parameters(self):
        frozen_model_config = FrozenLayerStackConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=make_layer_stack_config().layer_config,
        )
        cfg = self.preset(
            model_config=frozen_model_config,
            test_time_training_learning_rate=0.01,
            test_time_training_num_inner_steps=1,
        )

        with self.assertRaises(ValueError):
            cfg.build()

    def test_ttt_adapts_memory_without_mutating_generator_parameters(self):
        dim = 1
        cfg = self.preset(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
            test_time_training_learning_rate=0.1,
            test_time_training_num_inner_steps=1,
            model_config=make_layer_stack_config(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=dim,
                bias_flag=False,
            ),
        )
        model = cfg.build()
        model.eval()
        memory_weight = model.memory_model[0].model.weight_params
        decoder_weight = model.memory_decoder[0].model.weight_params
        with torch.no_grad():
            memory_weight.fill_(0.0)
            decoder_weight.fill_(1.0)
        original_memory_weight = memory_weight.detach().clone()
        original_decoder_weight = decoder_weight.detach().clone()
        inputs = torch.tensor([[2.0]])

        adapted_memory = model._adapt_and_retrieve(
            inputs,
            model.memory_model,
            model.memory_decoder,
        )

        torch.testing.assert_close(adapted_memory, torch.tensor([[1.6]]))
        torch.testing.assert_close(memory_weight, original_memory_weight)
        torch.testing.assert_close(decoder_weight, original_decoder_weight)

    def test_ttt_training_retains_higher_order_meta_gradient(self):
        dim = 1
        cfg = self.preset(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
            test_time_training_learning_rate=0.1,
            test_time_training_num_inner_steps=1,
            model_config=make_layer_stack_config(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=dim,
                bias_flag=False,
            ),
        )
        model = cfg.build()
        model.train()
        memory_weight = model.memory_model[0].model.weight_params
        decoder_weight = model.memory_decoder[0].model.weight_params
        with torch.no_grad():
            memory_weight.fill_(0.0)
            decoder_weight.fill_(1.0)

        adapted_memory = model._adapt_and_retrieve(
            torch.tensor([[2.0]]),
            model.memory_model,
            model.memory_decoder,
        )
        adapted_memory.sum().backward()

        torch.testing.assert_close(adapted_memory, torch.tensor([[1.6]]))
        torch.testing.assert_close(memory_weight.grad, torch.tensor([[0.4]]))

    def test_ttt_succeeds_through_lightning_validate_and_test(self):
        data_loader = DataLoader(
            TensorDataset(
                torch.tensor(
                    [
                        [1.0, -1.0, 0.5, -0.5],
                        [0.25, 0.75, -0.25, -0.75],
                    ]
                )
            ),
            batch_size=2,
        )

        for config_cls, _ in self.memory_cases():
            with self.subTest(config_cls=config_cls.__name__):
                memory = self.preset(
                    config_cls=config_cls,
                    input_dim=4,
                    output_dim=4,
                    test_time_training_learning_rate=0.1,
                    test_time_training_num_inner_steps=1,
                ).build()
                harness = DynamicMemoryEvaluationHarness(memory)
                original_state = {
                    name: value.detach().clone()
                    for name, value in memory.state_dict().items()
                }
                trainer = Trainer(
                    accelerator="cpu",
                    devices=1,
                    max_epochs=1,
                    limit_train_batches=1,
                    limit_val_batches=1,
                    num_sanity_val_steps=1,
                    logger=False,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                )

                with mock.patch.object(
                    memory,
                    "_adapt_and_retrieve",
                    wraps=memory._adapt_and_retrieve,
                ) as adapt_and_retrieve:
                    trainer.fit(
                        harness,
                        train_dataloaders=data_loader,
                        val_dataloaders=data_loader,
                    )
                    trainer.validate(
                        harness,
                        dataloaders=data_loader,
                        verbose=False,
                    )
                    trainer.test(
                        harness,
                        dataloaders=data_loader,
                        verbose=False,
                    )
                    trainer.predict(
                        harness,
                        dataloaders=data_loader,
                    )

                self.assertFalse(any(harness.evaluation_grad_modes))
                self.assertTrue(all(harness.evaluation_inference_modes[-3:]))
                self.assertTrue(harness.saw_sanity_validation)
                self.assertEqual(adapt_and_retrieve.call_count, 6)
                self.assertIsNotNone(harness.last_output)
                self.assertTrue(torch.isfinite(harness.last_output).all().item())
                self.assertFalse(harness.last_output.requires_grad)
                for name, value in memory.state_dict().items():
                    torch.testing.assert_close(value, original_state[name])

    def test_output_matches_gated_residual_closed_form(self):
        dim = 4
        model = self.preset(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        ).build()
        model.memory_model = ScaleModule(2.0)
        gate_logits = torch.tensor([2.0, -1.0, 0.5, -2.0])
        model.memory_gate_model = ConstantLastDimModule(gate_logits)
        inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

        output = model(inputs)

        expected = inputs + torch.sigmoid(gate_logits).reshape(1, dim) * (inputs * 2.0)
        torch.testing.assert_close(output, expected)

    def test_output_matches_weighted_closed_form(self):
        dim = 4
        model = self.preset(
            config_cls=WeightedDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        ).build()
        model.memory_model = ScaleModule(3.0)
        weight_logits = torch.tensor([-1.0, 2.0])
        model.memory_weight_model = ConstantLastDimModule(weight_logits)
        inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

        output = model(inputs)

        weights = torch.softmax(weight_logits, dim=-1)
        expected = weights[0] * inputs + weights[1] * (inputs * 3.0)
        torch.testing.assert_close(output, expected)

    def test_output_matches_element_wise_weighted_closed_form(self):
        dim = 4
        model = self.preset(
            config_cls=ElementWiseWeightedDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        ).build()
        model.memory_model = ScaleModule(3.0)
        weight_logits = torch.tensor([-2.0, -0.5, 1.0, 3.0])
        model.memory_weight_model = ConstantLastDimModule(weight_logits)
        inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

        output = model(inputs)

        feature_weights = torch.sigmoid(weight_logits).reshape(1, dim)
        expected = (1 - feature_weights) * inputs + feature_weights * (inputs * 3.0)
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
        gate_logits = torch.tensor([-2.0, 0.5, 2.0])
        model.memory_gate_model = ConstantLastDimModule(gate_logits)
        inputs = torch.tensor([[1.0, 2.0, -1.0]])

        output = model(inputs)

        memory_bank = torch.stack([inputs, inputs * 2.0], dim=-2)
        query = inputs.unsqueeze(-2)
        attention_scores = torch.matmul(query, memory_bank.transpose(-2, -1))
        attention_scores = attention_scores / (dim**0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_memory = torch.matmul(attention_weights, memory_bank).squeeze(-2)
        expected = inputs + torch.sigmoid(gate_logits).reshape(1, dim) * attended_memory
        torch.testing.assert_close(output, expected)

    def assert_module_has_nonzero_gradient(self, module: nn.Module, name: str) -> None:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(gradients, f"{name} has no trainable parameters")
        self.assertTrue(
            any(
                gradient is not None and torch.any(gradient.abs() > 0)
                for gradient in gradients
            ),
            f"{name} did not receive a nonzero gradient",
        )

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
                    output.pow(2).sum().backward()

                    self.assert_module_has_nonzero_gradient(
                        model.memory_model,
                        "memory_model",
                    )
                    if isinstance(model, GatedResidualDynamicMemory):
                        self.assert_module_has_nonzero_gradient(
                            model.memory_gate_model,
                            "memory_gate_model",
                        )
                    elif isinstance(model, WeightedDynamicMemory):
                        self.assert_module_has_nonzero_gradient(
                            model.memory_weight_model,
                            "memory_weight_model",
                        )
                    elif isinstance(model, ElementWiseWeightedDynamicMemory):
                        self.assert_module_has_nonzero_gradient(
                            model.memory_weight_model,
                            "memory_weight_model",
                        )
                    elif isinstance(model, AttentionDynamicMemory):
                        for child_name in (
                            "query_model",
                            "key_model",
                            "value_model",
                            "output_model",
                            "memory_gate_model",
                        ):
                            self.assert_module_has_nonzero_gradient(
                                getattr(model, child_name),
                                child_name,
                            )
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
        residual_connection_option: ResidualConnectionOptions | None = None,
        dropout_probability: float = 0.0,
        layer_norm_position: LayerNormPositionOptions = (
            LayerNormPositionOptions.DISABLED
        ),
        gate_config: LayerStackConfig | None = None,
        halting_config=None,
        memory_config: DynamicMemoryConfig | None = None,
        layer_model_config: LinearLayerConfig | None = None,
    ) -> LayerConfig:
        return make_layer_config(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            residual_connection_option=residual_connection_option,
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=gate_config,
            halting_config=halting_config,
            memory_config=memory_config,
            layer_model_config=layer_model_config,
        )

    def layer_config_without_memory(self, dim: int) -> LayerConfig:
        return LayerConfig(
            input_dim=dim,
            output_dim=dim,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
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
                layer.memory_model.memory_position_option = position

                output = layer(LayerState(hidden=inputs))

                torch.testing.assert_close(output.hidden, expected)

    def test_layer_reads_memory_position_from_model_before_config(self):
        dim = 4
        inputs = torch.zeros(2, dim)
        cfg = self.layer_config_without_memory(dim)
        layer = Layer(cfg)
        layer.model = ScaleModule(2.0)
        layer.memory_model = AddConstantModule(1.0)
        layer.memory_model.memory_position_option = MemoryPositionOptions.BEFORE_AFFINE

        output = layer(LayerState(hidden=inputs))

        torch.testing.assert_close(output.hidden, torch.full((2, dim), 2.0))

    def test_layer_stack_shared_memory_reuses_one_module(self):
        dim = 4
        memory_config = make_memory_config(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        )
        cfg = LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=3,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            shared_memory_config=memory_config,
            layer_config=self.layer_config_without_memory(dim),
        )

        model = LayerStack(cfg)
        memory_models = [layer.memory_model for layer in model]

        self.assertTrue(all(memory_model is not None for memory_model in memory_models))
        first_memory_model = memory_models[0]
        self.assertTrue(
            all(memory_model is first_memory_model for memory_model in memory_models)
        )
        for layer in model:
            self.assertIsNone(layer.cfg.memory_config)
            self.assertIsNone(layer.memory_config)

    def test_shared_memory_rejects_per_layer_memory_config(self):
        dim = 4
        shared_memory_config = make_memory_config(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        )
        per_layer_memory_config = make_memory_config(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        )
        layer_config = self.layer_config_without_memory(dim)
        layer_config.memory_config = per_layer_memory_config
        cfg = LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=3,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            shared_memory_config=shared_memory_config,
            layer_config=layer_config,
        )

        with self.assertRaises(ValueError):
            LayerStack(cfg)

    def test_shared_memory_rejects_invalid_config_type(self):
        dim = 4
        cfg = LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=3,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            shared_memory_config=object(),
            layer_config=self.layer_config_without_memory(dim),
        )

        with self.assertRaises(TypeError):
            LayerStack(cfg)

    def test_shared_memory_rejects_width_changing_stack(self):
        dim = 4
        memory_config = make_memory_config(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=dim,
            output_dim=dim,
        )
        cfg = LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim * 2,
            output_dim=dim,
            num_layers=3,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            shared_memory_config=memory_config,
            layer_config=self.layer_config_without_memory(dim),
        )

        with self.assertRaises(ValueError):
            LayerStack(cfg)


if __name__ == "__main__":
    unittest.main()
