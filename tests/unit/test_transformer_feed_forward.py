from emperor.base.layer.residual import ResidualConnectionOptions
import torch
import unittest

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.experts.core.options import RoutingInitializationMode
from emperor.transformer.feed_forward.core.config import FeedForwardConfig
from emperor.transformer.feed_forward.core.layers import FeedForward

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
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
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
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
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
        experts_routing_initialization_mode: RoutingInitializationMode = RoutingInitializationMode.LAYER,
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
