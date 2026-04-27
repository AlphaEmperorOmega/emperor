import torch
import unittest

from emperor.linears.options import LinearOptions
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.options import (
    DynamicMemoryOptions,
    MemoryPositionOptions,
    MemorySizeOptions,
)
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.core import (
    DynamicMemoryAbstract,
    GatedResidualDynamicMemory,
    WeightedDynamicMemory,
)


class TestMemoryHandlers(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 36,
        output_dim: int = 24,
        bias_flag: bool = True,
        model_type: DynamicMemoryOptions = DynamicMemoryOptions.FUSION,
        memory_size_option: MemorySizeOptions = MemorySizeOptions.MEDIUM,
        memory_position_option: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE,
        stack_num_layers: int = 1,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
    ) -> DynamicMemoryConfig:
        return DynamicMemoryConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=model_type,
            memory_size_option=memory_size_option,
            memory_position_option=memory_position_option,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=stack_residual_flag,
                    dropout_probability=stack_dropout_probability,
                    gate_config=None,
                    halting_config=None,
                    shared_halting_flag=False,
                    layer_model_config=LinearLayerConfig(
                        model_type=LinearOptions.LINEAR,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )

    def _get_memory_dim(
        self,
        position_option: MemoryPositionOptions,
        input_dim: int = 12,
        output_dim: int = 24,
    ) -> int:
        if position_option == MemoryPositionOptions.BEFORE_AFFINE:
            return input_dim
        return output_dim

    def test_fusion_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        for position_option in MemoryPositionOptions:
            for size_option in MemorySizeOptions:
                msg = f"position={position_option}, size={size_option}"
                with self.subTest(msg=msg):
                    dim = self._get_memory_dim(position_option, input_dim, output_dim)
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        memory_size_option=size_option,
                        memory_position_option=position_option,
                    )
                    if size_option == MemorySizeOptions.DISABLED:
                        with self.assertRaises(ValueError):
                            GatedResidualDynamicMemory(cfg)
                    else:
                        model = GatedResidualDynamicMemory(cfg)
                        input_tensor = torch.ones(batch_size, dim)
                        output = model(input_tensor)
                        self.assertEqual(output.shape, (batch_size, dim))
                        self.assertIsInstance(output, torch.Tensor)

    def test_weighted_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        for size_option in MemorySizeOptions:
            for position_option in MemoryPositionOptions:
                msg = f"position={position_option}, size={size_option}"
                with self.subTest(msg=msg):
                    dim = self._get_memory_dim(position_option, input_dim, output_dim)
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        model_type=DynamicMemoryOptions.WEIGHTED,
                        memory_size_option=size_option,
                        memory_position_option=position_option,
                    )
                    if size_option == MemorySizeOptions.DISABLED:
                        with self.assertRaises(ValueError):
                            WeightedDynamicMemory(cfg)
                    else:
                        model = WeightedDynamicMemory(cfg)
                        input_tensor = torch.randn(batch_size, dim)
                        output = model(input_tensor)
                        self.assertEqual(output.shape, (batch_size, dim))
                        self.assertIsInstance(output, torch.Tensor)

    def test_fusion_concatenation_and_compression_correctness(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        for position_option in MemoryPositionOptions:
            with self.subTest(position=position_option):
                dim = self._get_memory_dim(position_option, input_dim, output_dim)
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    memory_position_option=position_option,
                )
                model = GatedResidualDynamicMemory(cfg)
                model.eval()
                input_tensor = torch.randn(batch_size, dim)
                memory = model.memory_model(input_tensor)
                combined = torch.cat([input_tensor, memory], dim=-1)
                expected = model.compression_model(combined)
                output = model(input_tensor)
                self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_weighted_blending_mathematical_correctness(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        for position_option in MemoryPositionOptions:
            with self.subTest(position=position_option):
                dim = self._get_memory_dim(position_option, input_dim, output_dim)
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=DynamicMemoryOptions.WEIGHTED,
                    memory_position_option=position_option,
                )
                model = WeightedDynamicMemory(cfg)
                model.eval()
                input_tensor = torch.randn(batch_size, dim)
                memory = model.memory_model(input_tensor)
                combined = torch.cat([input_tensor, memory], dim=-1)
                weight_logits = model.memory_weight_model(combined)
                weights = torch.softmax(weight_logits, dim=-1).unsqueeze(-1)
                stacked = torch.cat(
                    [input_tensor.unsqueeze(-2), memory.unsqueeze(-2)], dim=-2
                )
                expected = (stacked * weights).sum(dim=-2)
                output = model(input_tensor)
                self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_weighted_softmax_weights_form_valid_distribution(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        for position_option in MemoryPositionOptions:
            with self.subTest(position=position_option):
                dim = self._get_memory_dim(position_option, input_dim, output_dim)
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=DynamicMemoryOptions.WEIGHTED,
                    memory_position_option=position_option,
                )
                model = WeightedDynamicMemory(cfg)
                model.eval()
                input_tensor = torch.randn(batch_size, dim)
                memory = model.memory_model(input_tensor)
                combined = torch.cat([input_tensor, memory], dim=-1)
                weight_logits = model.memory_weight_model(combined)
                weights = torch.softmax(weight_logits, dim=-1)
                self.assertTrue(
                    torch.allclose(weights.sum(dim=-1), torch.ones(batch_size))
                )
                self.assertTrue((weights >= 0).all())

    def test_fusion_output_varies_with_input(self):
        input_dim = 12
        output_dim = 24
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = GatedResidualDynamicMemory(cfg)
        model.eval()
        input_a = torch.randn(1, output_dim)
        input_b = torch.randn(1, output_dim)
        output_a = model(input_a)
        output_b = model(input_b)
        self.assertFalse(torch.allclose(output_a, output_b))

    def test_weighted_output_varies_with_input(self):
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicMemoryOptions.WEIGHTED,
        )
        model = WeightedDynamicMemory(cfg)
        model.eval()
        input_a = torch.randn(1, output_dim)
        input_b = torch.randn(1, output_dim)
        output_a = model(input_a)
        output_b = model(input_b)
        self.assertFalse(torch.allclose(output_a, output_b))

    def test_disabled_memory_size_raises(self):
        for model_cls, model_type in [
            (GatedResidualDynamicMemory, DynamicMemoryOptions.FUSION),
            (WeightedDynamicMemory, DynamicMemoryOptions.WEIGHTED),
        ]:
            with self.subTest(model_type=model_type):
                cfg = self.preset(
                    model_type=model_type,
                    memory_size_option=MemorySizeOptions.DISABLED,
                )
                with self.assertRaises(ValueError):
                    model_cls(cfg)

    def test_fusion_gradients_flow(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        for position_option in MemoryPositionOptions:
            for size_option in MemorySizeOptions:
                if size_option == MemorySizeOptions.DISABLED:
                    continue
                with self.subTest(position=position_option, size=size_option):
                    dim = self._get_memory_dim(position_option, input_dim, output_dim)
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        memory_size_option=size_option,
                        memory_position_option=position_option,
                    )
                    model = GatedResidualDynamicMemory(cfg)
                    input_tensor = torch.randn(
                        batch_size, dim, requires_grad=True
                    )
                    output = model(input_tensor)
                    output.sum().backward()
                    grads = [
                        p.grad for p in model.parameters() if p.requires_grad
                    ]
                    non_none_grads = [
                        g for g in grads if g is not None and g.abs().sum() > 0
                    ]
                    self.assertTrue(len(non_none_grads) > 0)
                    self.assertIsNotNone(input_tensor.grad)

    def test_weighted_gradients_flow(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        for position_option in MemoryPositionOptions:
            for size_option in MemorySizeOptions:
                if size_option == MemorySizeOptions.DISABLED:
                    continue
                with self.subTest(position=position_option, size=size_option):
                    dim = self._get_memory_dim(position_option, input_dim, output_dim)
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        model_type=DynamicMemoryOptions.WEIGHTED,
                        memory_size_option=size_option,
                        memory_position_option=position_option,
                    )
                    model = WeightedDynamicMemory(cfg)
                    input_tensor = torch.randn(
                        batch_size, dim, requires_grad=True
                    )
                    output = model(input_tensor)
                    output.sum().backward()
                    grads = [
                        p.grad for p in model.parameters() if p.requires_grad
                    ]
                    non_none_grads = [
                        g for g in grads if g is not None and g.abs().sum() > 0
                    ]
                    self.assertTrue(len(non_none_grads) > 0)
                    self.assertIsNotNone(input_tensor.grad)

    def test_build_creates_model_for_each_option(self):
        input_dim = 12
        output_dim = 24
        for memory_option in DynamicMemoryOptions:
            if memory_option == DynamicMemoryOptions.DISABLED:
                with self.subTest(msg="option=DISABLED raises ValueError"):
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        model_type=memory_option,
                    )
                    with self.assertRaises(ValueError):
                        cfg.build()
                continue
            for position_option in MemoryPositionOptions:
                for size_option in MemorySizeOptions:
                    if size_option == MemorySizeOptions.DISABLED:
                        continue
                    msg = f"memory={memory_option}, position={position_option}, size={size_option}"
                    with self.subTest(msg=msg):
                        cfg = self.preset(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            model_type=memory_option,
                            memory_size_option=size_option,
                            memory_position_option=position_option,
                        )
                        model = cfg.build()
                        self.assertIsInstance(model, DynamicMemoryAbstract)

    def test_build_creates_correct_model_type(self):
        expected_types = {
            DynamicMemoryOptions.FUSION: GatedResidualDynamicMemory,
            DynamicMemoryOptions.WEIGHTED: WeightedDynamicMemory,
        }
        for memory_option, expected_type in expected_types.items():
            with self.subTest(memory_option=memory_option):
                cfg = self.preset(model_type=memory_option)
                model = cfg.build()
                self.assertIsInstance(model, expected_type)
