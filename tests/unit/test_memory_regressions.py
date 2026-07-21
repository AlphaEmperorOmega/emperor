import copy
import unittest

import torch

from emperor.memory import MemoryPositionOptions
from emperor.memory._base import DynamicMemoryAbstract
from emperor.memory._variants.attention import AttentionDynamicMemory
from emperor.memory._variants.element_wise_weighted import (
    ElementWiseWeightedDynamicMemory,
)
from emperor.memory._variants.gated_residual import GatedResidualDynamicMemory
from emperor.memory._variants.weighted import WeightedDynamicMemory
from unit.test_memory import (
    MEMORY_CASES,
    make_layer_stack_config,
    make_memory_config,
    only_layer,
    set_affine_parameters,
)


def active_generators(memory: DynamicMemoryAbstract) -> dict[str, torch.nn.Module]:
    generators = {"memory_model": memory.memory_model}
    if isinstance(memory, GatedResidualDynamicMemory):
        generators["memory_gate_model"] = memory.memory_gate_model
    elif isinstance(memory, WeightedDynamicMemory):
        generators["memory_weight_model"] = memory.memory_weight_model
    elif isinstance(memory, ElementWiseWeightedDynamicMemory):
        generators["memory_weight_model"] = memory.memory_weight_model
    elif isinstance(memory, AttentionDynamicMemory):
        generators.update(
            {
                "query_model": memory.query_model,
                "key_model": memory.key_model,
                "value_model": memory.value_model,
                "output_model": memory.output_model,
                "memory_gate_model": memory.memory_gate_model,
            }
        )
    return generators


def optimization_step(
    model: DynamicMemoryAbstract,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    output.square().mean().backward()
    optimizer.step()
    return output.detach().clone()


class MemoryRegressionTests(unittest.TestCase):
    def test_all_variants_preserve_float64_noncontiguous_inputs_and_samples(
        self,
    ) -> None:
        base = torch.tensor(
            [
                [
                    [1.0, -2.0, 0.5, 3.0],
                    [0.25, 1.5, -1.0, 2.0],
                    [2.0, 0.0, -0.5, 1.0],
                ],
                [
                    [-1.0, 0.5, 2.0, -3.0],
                    [1.25, -0.75, 0.0, 0.5],
                    [3.0, -1.0, 1.5, -0.25],
                ],
            ],
            dtype=torch.float64,
        )
        inputs = base.transpose(0, 1)
        self.assertFalse(inputs.is_contiguous())

        for config_cls, _ in MEMORY_CASES:
            with self.subTest(config=config_cls.__name__):
                torch.manual_seed(31)
                model = make_memory_config(
                    config_cls=config_cls,
                    input_dim=4,
                    output_dim=4,
                ).build()
                model.double().eval()

                together = model(inputs)
                separately = torch.cat(
                    [model(sample.unsqueeze(0)) for sample in inputs],
                    dim=0,
                )

                self.assertEqual(together.shape, inputs.shape)
                self.assertEqual(together.dtype, torch.float64)
                self.assertEqual(together.device.type, "cpu")
                self.assertTrue(torch.isfinite(together).all())
                torch.testing.assert_close(together, separately)

    def test_real_optimizer_updates_every_active_generator(self) -> None:
        inputs = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]])
        for config_cls, _ in MEMORY_CASES:
            for position in MemoryPositionOptions:
                with self.subTest(
                    config=config_cls.__name__,
                    position=position,
                ):
                    torch.manual_seed(37)
                    model = make_memory_config(
                        config_cls=config_cls,
                        input_dim=4,
                        output_dim=4,
                        memory_position_option=position,
                    ).build()
                    generators = active_generators(model)
                    before = {
                        child_name: {
                            name: parameter.detach().clone()
                            for name, parameter in child.named_parameters()
                        }
                        for child_name, child in generators.items()
                    }
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

                    optimization_step(model, optimizer, inputs)

                    for child_name, child in generators.items():
                        gradients = [
                            parameter.grad
                            for parameter in child.parameters()
                            if parameter.requires_grad
                        ]
                        self.assertTrue(gradients, child_name)
                        self.assertTrue(
                            all(
                                gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients
                            ),
                            child_name,
                        )
                        self.assertTrue(
                            any(
                                not torch.equal(
                                    parameter.detach(),
                                    before[child_name][name],
                                )
                                for name, parameter in child.named_parameters()
                            ),
                            child_name,
                        )

    def test_strict_model_and_optimizer_checkpoint_continuation(self) -> None:
        inputs = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]])
        for config_cls, _ in MEMORY_CASES:
            with self.subTest(config=config_cls.__name__):
                torch.manual_seed(41)
                source = make_memory_config(
                    config_cls=config_cls,
                    input_dim=4,
                    output_dim=4,
                ).build()
                source_optimizer = torch.optim.SGD(
                    source.parameters(),
                    lr=0.03,
                    momentum=0.9,
                )
                optimization_step(source, source_optimizer, inputs)
                model_state = {
                    name: value.detach().clone()
                    for name, value in source.state_dict().items()
                }
                optimizer_state = copy.deepcopy(source_optimizer.state_dict())

                torch.manual_seed(99)
                restored = make_memory_config(
                    config_cls=config_cls,
                    input_dim=4,
                    output_dim=4,
                ).build()
                restored_optimizer = torch.optim.SGD(
                    restored.parameters(),
                    lr=0.03,
                    momentum=0.9,
                )
                result = restored.load_state_dict(model_state, strict=True)
                restored_optimizer.load_state_dict(optimizer_state)

                self.assertEqual(result.missing_keys, [])
                self.assertEqual(result.unexpected_keys, [])
                torch.testing.assert_close(source(inputs), restored(inputs))
                source_output = optimization_step(
                    source,
                    source_optimizer,
                    inputs,
                )
                restored_output = optimization_step(
                    restored,
                    restored_optimizer,
                    inputs,
                )
                torch.testing.assert_close(source_output, restored_output)
                for name, value in source.state_dict().items():
                    torch.testing.assert_close(
                        value,
                        restored.state_dict()[name],
                    )

    def test_ttt_optimizer_reaches_memory_decoder_and_meta_parameters(self) -> None:
        model = make_memory_config(
            input_dim=1,
            output_dim=1,
            test_time_training_learning_rate=0.1,
            test_time_training_num_inner_steps=1,
            model_config=make_layer_stack_config(
                input_dim=1,
                hidden_dim=1,
                output_dim=1,
                bias_flag=False,
            ),
        ).build()
        self.assertIsInstance(model, GatedResidualDynamicMemory)
        with torch.no_grad():
            model.memory_model[0].model.weight_params.zero_()
            model.memory_decoder[0].model.weight_params.fill_(1.0)
        gate_layer = only_layer(model.memory_gate_model)
        set_affine_parameters(
            model.memory_gate_model,
            torch.zeros_like(gate_layer.model.weight_params),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        memory_before = model.memory_model[0].model.weight_params.detach().clone()
        decoder_before = model.memory_decoder[0].model.weight_params.detach().clone()

        output = optimization_step(
            model,
            optimizer,
            torch.tensor([[2.0]]),
        )

        torch.testing.assert_close(output, torch.tensor([[2.8]]))
        memory_gradient = model.memory_model[0].model.weight_params.grad
        decoder_gradient = model.memory_decoder[0].model.weight_params.grad
        self.assertIsNotNone(memory_gradient)
        self.assertIsNotNone(decoder_gradient)
        self.assertTrue(torch.isfinite(memory_gradient).all())
        self.assertTrue(torch.isfinite(decoder_gradient).all())
        self.assertTrue(torch.any(memory_gradient != 0))
        self.assertTrue(torch.any(decoder_gradient != 0))
        self.assertFalse(
            torch.equal(
                model.memory_model[0].model.weight_params.detach(),
                memory_before,
            )
        )
        self.assertFalse(
            torch.equal(
                model.memory_decoder[0].model.weight_params.detach(),
                decoder_before,
            )
        )


if __name__ == "__main__":
    unittest.main()
