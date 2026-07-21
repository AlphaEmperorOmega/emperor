import unittest
from dataclasses import replace

import torch

from models.transformer.expert_linear.config_builder import (
    TransformerExpertLinearConfigBuilder,
)
from models.transformer.expert_linear_adaptive.config_builder import (
    TransformerExpertLinearAdaptiveConfigBuilder,
)
from models.transformer.linear.config_builder import TransformerLinearConfigBuilder
from models.transformer.linear_adaptive.config_builder import (
    TransformerLinearAdaptiveConfigBuilder,
)


class TestAttentionRuntimeLayout(unittest.TestCase):
    def builder_cases(self):
        return (
            TransformerLinearConfigBuilder,
            TransformerLinearAdaptiveConfigBuilder,
            TransformerExpertLinearConfigBuilder,
            TransformerExpertLinearAdaptiveConfigBuilder,
        )

    def preset(self, builder_type):
        options = dict(
            batch_size=4,
            model_dim=16,
            source_sequence_length=8,
            target_sequence_length=8,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=32,
            dropout_probability=0.0,
        )
        if "Expert" in builder_type.__name__:
            options.update(expert_num_experts=4, expert_top_k=2)
        return builder_type(**options).build()

    def attention_configs(self, builder_type):
        experiment_config = self.preset(builder_type).experiment_config
        encoder_layer = experiment_config.encoder_config.layer_config.layer_model_config
        decoder_layer = experiment_config.decoder_config.layer_config.layer_model_config
        return (
            encoder_layer.attention_config,
            decoder_layer.self_attention_config,
            decoder_layer.cross_attention_config,
        )

    def test_batch_first_dynamic_batch_and_sequence_lengths_for_all_backends(self):
        for builder_type in self.builder_cases():
            encoder_config, decoder_config, cross_config = self.attention_configs(
                builder_type
            )
            modules = (
                ("encoder", encoder_config.build(), False),
                ("decoder", decoder_config.build(), False),
                ("cross", cross_config.build(), True),
            )
            for name, module, cross_attention in modules:
                for batch_size in (1, 4, 3):
                    for target_length in (1, 5, 8):
                        source_length = 8 if cross_attention else target_length
                        with self.subTest(
                            backend=builder_type.__name__,
                            attention=name,
                            batch_size=batch_size,
                            target_length=target_length,
                        ):
                            query = torch.randn(
                                batch_size,
                                target_length,
                                16,
                                requires_grad=True,
                            )
                            if cross_attention:
                                key = torch.randn(
                                    batch_size,
                                    source_length,
                                    16,
                                    requires_grad=True,
                                )
                            else:
                                key = query
                            padding_mask = torch.zeros(
                                batch_size,
                                source_length,
                                dtype=torch.bool,
                            )
                            padding_mask[:, -1] = source_length > 1
                            attention_mask = torch.zeros(
                                target_length,
                                source_length,
                                dtype=torch.bool,
                            )
                            if name == "decoder":
                                attention_mask = torch.triu(
                                    torch.ones_like(attention_mask),
                                    diagonal=1,
                                )
                            output, _, auxiliary_loss = module(
                                query,
                                key,
                                key,
                                k_padding_mask=padding_mask,
                                attention_mask=attention_mask,
                            )
                            self.assertEqual(
                                output.shape,
                                (batch_size, target_length, 16),
                            )
                            self.assertTrue(
                                auxiliary_loss is None or auxiliary_loss.ndim == 0
                            )

                self.assertEqual(module.batch_size, 4)
                self.assertEqual(module.target_sequence_length, 8)
                self.assertEqual(module.source_sequence_length, 8)

    def test_sequence_first_and_unbatched_layouts_for_all_backends(self):
        for builder_type in self.builder_cases():
            encoder_config, _, cross_config = self.attention_configs(builder_type)
            sequence_first_self = replace(
                encoder_config,
                batch_first_flag=False,
            ).build()
            sequence_first_cross = replace(
                cross_config,
                batch_first_flag=False,
            ).build()
            with self.subTest(backend=builder_type.__name__, layout="sequence-first"):
                self_input = torch.randn(5, 3, 16)
                self_output, _, _ = sequence_first_self(
                    self_input,
                    self_input,
                    self_input,
                    k_padding_mask=torch.zeros(3, 5, dtype=torch.bool),
                )
                query = torch.randn(4, 3, 16)
                encoder_output = torch.randn(7, 3, 16)
                cross_output, _, _ = sequence_first_cross(
                    query,
                    encoder_output,
                    encoder_output,
                    k_padding_mask=torch.zeros(3, 7, dtype=torch.bool),
                    attention_mask=torch.zeros(4, 7, dtype=torch.bool),
                )
                self.assertEqual(self_output.shape, (5, 3, 16))
                self.assertEqual(cross_output.shape, (4, 3, 16))

            with self.subTest(backend=builder_type.__name__, layout="unbatched"):
                self_input = torch.randn(5, 16)
                self_output, _, _ = sequence_first_self(
                    self_input,
                    self_input,
                    self_input,
                    k_padding_mask=torch.zeros(5, dtype=torch.bool),
                )
                query = torch.randn(4, 16)
                encoder_output = torch.randn(7, 16)
                cross_output, _, _ = sequence_first_cross(
                    query,
                    encoder_output,
                    encoder_output,
                    k_padding_mask=torch.zeros(7, dtype=torch.bool),
                    attention_mask=torch.zeros(4, 7, dtype=torch.bool),
                )
                self.assertEqual(self_output.shape, (5, 16))
                self.assertEqual(cross_output.shape, (4, 16))

    def test_runtime_dimensions_above_configured_maxima_are_rejected(self):
        for builder_type in self.builder_cases():
            encoder_config, _, cross_config = self.attention_configs(builder_type)
            with self.subTest(backend=builder_type.__name__, dimension="batch"):
                model = encoder_config.build()
                value = torch.randn(5, 4, 16)
                with self.assertRaises(ValueError):
                    model(value, value, value)
            with self.subTest(backend=builder_type.__name__, dimension="target"):
                model = encoder_config.build()
                value = torch.randn(1, 9, 16)
                with self.assertRaises(ValueError):
                    model(value, value, value)
            with self.subTest(backend=builder_type.__name__, dimension="source"):
                model = cross_config.build()
                query = torch.randn(1, 4, 16)
                encoder_output = torch.randn(1, 9, 16)
                with self.assertRaises(ValueError):
                    model(query, encoder_output, encoder_output)

    def test_gradients_flow_with_runtime_layouts(self):
        for builder_type in self.builder_cases():
            _, decoder_config, cross_config = self.attention_configs(builder_type)
            for name, config, cross_attention in (
                ("self", decoder_config, False),
                ("cross", cross_config, True),
            ):
                with self.subTest(backend=builder_type.__name__, attention=name):
                    model = config.build()
                    query = torch.randn(2, 3, 16, requires_grad=True)
                    key = (
                        torch.randn(2, 5, 16, requires_grad=True)
                        if cross_attention
                        else query
                    )
                    output, _, auxiliary_loss = model(query, key, key)
                    loss = output.square().mean()
                    if auxiliary_loss is not None:
                        loss = loss + auxiliary_loss
                    loss.backward()
                    self.assertIsNotNone(query.grad)
                    if cross_attention:
                        self.assertIsNotNone(key.grad)
                    gradients = [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    self.assertTrue(any(gradient is not None for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
