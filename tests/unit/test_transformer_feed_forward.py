import unittest

import torch

from emperor.experts import RoutingInitializationMode
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    MirroredLayerStack,
    RecurrentLayer,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import FeedForward, FeedForwardConfig
from unit.test_experts import MixtureOfExpertsPresetMixin


class TestFeedForward(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        hidden_dim: int = 20,
        num_layers: int = 2,
        dropout_probability: float = 0.0,
        bias_flag: bool = True,
        activation: ActivationOptions = ActivationOptions.RELU,
        layer_norm_position: LayerNormPositionOptions = (
            LayerNormPositionOptions.DISABLED
        ),
        stack_config: "LayerStackConfig | None" = None,
    ) -> FeedForwardConfig:
        if stack_config is None:
            stack_config = LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=activation,
                    layer_norm_position=layer_norm_position,
                    residual_config=None,
                    dropout_probability=dropout_probability,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=bias_flag,
                    ),
                ),
            )

        return FeedForwardConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            stack_config=stack_config,
        )

    def test_init_with_valid_inputs(self):
        num_layers_list = [2, 4, 6]
        for num_layers in num_layers_list:
            message = f"Testing configuration with num_layers={num_layers}"
            with self.subTest(msg=message):
                cfg = self.preset(num_layers=num_layers)
                m = FeedForward(cfg)
                self.assertIsInstance(m, cfg._registry_owner())
                self.assertEqual(m.stack_config.num_layers, num_layers)

    def test_forward(self):
        num_layers_list = [2, 4, 6]
        flag_options = [True, False]
        batch_size = 8
        sequence_length = 4

        for num_layers in num_layers_list:
            for matrix_input_flag in flag_options:
                message = (
                    f"Testing FeedForward configuration with "
                    f"num_layers={num_layers}, "
                    f"matrix_input_flag={matrix_input_flag}"
                )
                with self.subTest(msg=message):
                    cfg = self.preset(num_layers=num_layers)
                    m = FeedForward(cfg)
                    if matrix_input_flag:
                        input = torch.randn(batch_size, cfg.input_dim)
                        expected_output = (batch_size, cfg.output_dim)
                    else:
                        input = torch.randn(
                            batch_size, sequence_length, cfg.input_dim
                        )
                        expected_output = (
                            batch_size,
                            sequence_length,
                            cfg.output_dim,
                        )

                    output, loss = m(input)

                    if isinstance(output, tuple):
                        output, _ = output

                    self.assertEqual(output.shape, expected_output)

    def test_one_layer_depth_builds_two_wrapped_layers(self):
        model = FeedForward(self.preset(num_layers=1))

        self.assertIsInstance(model.model, MirroredLayerStack)
        self.assertEqual(len(model.model), 2)
        self.assertTrue(all(isinstance(layer, Layer) for layer in model.model))
        self.assertEqual(
            [(layer.input_dim, layer.output_dim) for layer in model.model],
            [(10, 20), (20, 10)],
        )

    def test_three_layer_depth_builds_independent_six_layer_arms(self):
        model = FeedForward(self.preset(num_layers=3))

        self.assertEqual(len(model.model.expansion_layers), 3)
        self.assertEqual(len(model.model.contraction_layers), 3)
        self.assertEqual(
            [(layer.input_dim, layer.output_dim) for layer in model.model],
            [
                (10, 20),
                (20, 20),
                (20, 20),
                (20, 20),
                (20, 20),
                (20, 10),
            ],
        )
        expansion_parameters = {
            id(parameter)
            for layer in model.model.expansion_layers
            for parameter in layer.parameters()
        }
        contraction_parameters = {
            id(parameter)
            for layer in model.model.contraction_layers
            for parameter in layer.parameters()
        }
        self.assertTrue(expansion_parameters)
        self.assertTrue(contraction_parameters)
        self.assertTrue(expansion_parameters.isdisjoint(contraction_parameters))

    def test_only_final_contraction_uses_output_pipeline_policy(self):
        model = FeedForward(
            self.preset(
                num_layers=3,
                dropout_probability=0.25,
                activation=ActivationOptions.RELU,
            )
        )

        for layer in model.model[:-1]:
            self.assertEqual(layer.activation_function, ActivationOptions.RELU)
            self.assertIsNotNone(layer.dropout_module)
        self.assertEqual(
            model.model[-1].activation_function,
            ActivationOptions.DISABLED,
        )
        self.assertIsNone(model.model[-1].dropout_module)

    def test_mirrored_stack_preserves_dtype_shape_and_gradients(self):
        model = FeedForward(
            self.preset(input_dim=6, output_dim=4, hidden_dim=12, num_layers=3)
        ).double()
        input_batch = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)

        output, loss = model(input_batch)
        (output.square().mean() + loss).backward()

        self.assertEqual(output.shape, (2, 5, 4))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(loss.shape, ())
        self.assertIsNotNone(input_batch.grad)
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )

    def test_recurrent_feed_forward_reuses_a_mirrored_layer_stack(self):
        block = self.preset(
            input_dim=8,
            output_dim=8,
            hidden_dim=16,
            num_layers=2,
        ).stack_config
        recurrent_config = RecurrentLayerConfig(
            input_dim=8,
            output_dim=8,
            max_steps=2,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=block,
            gate_config=None,
            residual_config=None,
            halting_config=None,
            memory_config=None,
        )
        model = FeedForward(
            FeedForwardConfig(
                input_dim=8,
                output_dim=8,
                stack_config=recurrent_config,
            )
        )

        self.assertIsInstance(model.model, RecurrentLayer)
        self.assertIsInstance(model.model.block_model, MirroredLayerStack)
        self.assertEqual(len(model.model.block_model), 4)
        output, loss = model(torch.randn(2, 3, 8))
        self.assertEqual(output.shape, (2, 3, 8))
        self.assertEqual(loss.shape, ())


class TestFeedForwardWithMixtureOfExperts(
    MixtureOfExpertsPresetMixin, unittest.TestCase
):
    def feed_forward_preset(
        self,
        input_dim: int = 8,
        output_dim: int = 8,
        num_layers: int = 2,
        experts_top_k: int = 2,
        experts_num_experts: int = 4,
        experts_routing_initialization_mode: RoutingInitializationMode = (
            RoutingInitializationMode.LAYER
        ),
    ) -> FeedForwardConfig:
        mixture_of_experts_model_config = self.model_preset(
            input_dim=input_dim,
            output_dim=output_dim,
            experts_stack_num_layers=num_layers,
            experts_top_k=experts_top_k,
            experts_num_experts=experts_num_experts,
            experts_routing_initialization_mode=experts_routing_initialization_mode,
            experts_compute_expert_mixture_flag=True,
        )
        return FeedForwardConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            stack_config=mixture_of_experts_model_config,
        )

    def test_init(self):
        routing_modes = [
            RoutingInitializationMode.LAYER,
        ]
        num_layers_list = [2, 4]
        for routing_mode in routing_modes:
            for num_layers in num_layers_list:
                message = (
                    f"Testing MoE FeedForward init with "
                    f"routing_mode={routing_mode}, num_layers={num_layers}"
                )
                with self.subTest(msg=message):
                    cfg = self.feed_forward_preset(
                        num_layers=num_layers,
                        experts_routing_initialization_mode=routing_mode,
                    )
                    m = FeedForward(cfg)
                    self.assertIsInstance(m, cfg._registry_owner())
                    self.assertEqual(
                        m.stack_config.stack_config.num_layers, num_layers
                    )

    def test_forward(self):
        routing_modes = [
            RoutingInitializationMode.LAYER,
        ]
        num_layers_list = [2, 4]
        flag_options = [True, False]
        batch_size = 8
        sequence_length = 4

        for routing_mode in routing_modes:
            for num_layers in num_layers_list:
                for matrix_input_flag in flag_options:
                    message = (
                        f"Testing MoE FeedForward forward with "
                        f"routing_mode={routing_mode}, "
                        f"num_layers={num_layers}, "
                        f"matrix_input_flag={matrix_input_flag}"
                    )
                    with self.subTest(msg=message):
                        cfg = self.feed_forward_preset(
                            num_layers=num_layers,
                            experts_routing_initialization_mode=routing_mode,
                        )
                        m = FeedForward(cfg)
                        if matrix_input_flag:
                            input = torch.randn(batch_size, cfg.input_dim)
                            expected_output = (batch_size, cfg.output_dim)
                        else:
                            input = torch.randn(
                                batch_size, sequence_length, cfg.input_dim
                            )
                            expected_output = (
                                batch_size,
                                sequence_length,
                                cfg.output_dim,
                            )

                        output, loss = m(input)

                        if isinstance(output, tuple):
                            output, _ = output

                        self.assertEqual(output.shape, expected_output)

    def test_expert_feed_forward_uses_mirrored_expert_stack_and_loss(self):
        model = FeedForward(self.feed_forward_preset(num_layers=3))

        self.assertIsInstance(model.model.expert_stack, MirroredLayerStack)
        self.assertEqual(len(model.model.expert_stack), 6)
        input_batch = torch.randn(2, 4, 8, requires_grad=True)
        output, loss = model(input_batch)
        (output.square().mean() + loss).backward()
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(input_batch.grad)
