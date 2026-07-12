import unittest

import torch
from emperor.base.layer import LayerState
from emperor.transformer import TransformerDecoderLayerState
from models.transformer.expert_linear.config_builder import (
    TransformerExpertLinearConfigBuilder,
)
from models.transformer.linear.config_builder import TransformerLinearConfigBuilder


class TestTransformerDecoderBlock(unittest.TestCase):
    def preset(self, builder_type=TransformerLinearConfigBuilder, **options):
        builder_options = dict(
            batch_size=3,
            model_dim=16,
            source_sequence_length=7,
            target_sequence_length=6,
            encoder_num_layers=1,
            decoder_num_layers=2,
            attn_num_heads=2,
            feed_forward_hidden_dim=32,
            dropout_probability=0.0,
        )
        if builder_type is TransformerExpertLinearConfigBuilder:
            builder_options.update(
                expert_num_experts=4,
                expert_top_k=1,
                expert_normalize_probabilities_flag=False,
                expert_switch_loss_weight=0.1,
            )
        builder_options.update(options)
        model_config = builder_type(**builder_options).build()
        return model_config.experiment_config.decoder_config.build()

    def state(self):
        target_mask = torch.triu(
            torch.ones(4, 4, dtype=torch.bool),
            diagonal=1,
        )
        cross_mask = torch.zeros(4, 7, dtype=torch.bool)
        target_padding = torch.zeros(3, 4, dtype=torch.bool)
        encoder_padding = torch.zeros(3, 7, dtype=torch.bool)
        encoder_output = torch.randn(3, 7, 16)
        controller_state = {"sentinel": True}
        state = TransformerDecoderLayerState(
            hidden=torch.randn(3, 4, 16),
            loss=torch.tensor(1.0),
            target_key_padding_mask=target_padding,
            target_attention_mask=target_mask,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding,
            cross_attention_mask=cross_mask,
            controller_state=controller_state,
        )
        context = {
            "target_key_padding_mask": target_padding,
            "target_attention_mask": target_mask,
            "encoder_output": encoder_output,
            "encoder_padding_mask": encoder_padding,
            "cross_attention_mask": cross_mask,
            "controller_state": controller_state,
        }
        return state, context

    def controller_cases(self):
        return (
            ("plain", {}),
            ("gated", {"decoder_stack_gate_flag": True}),
            ("halted", {"decoder_stack_halting_flag": True}),
            ("memory", {"decoder_memory_flag": True}),
            ("recurrent", {"decoder_recurrent_flag": True}),
            (
                "recurrent-controlled",
                {
                    "decoder_recurrent_flag": True,
                    "decoder_recurrent_gate_flag": True,
                    "decoder_recurrent_halting_flag": True,
                },
            ),
        )

    def test_decoder_context_survives_all_controller_paths(self):
        for name, options in self.controller_cases():
            with self.subTest(controller=name):
                decoder = self.preset(**options)
                state, context = self.state()

                output = decoder(state)

                self.assertIs(output, state)
                self.assertEqual(output.hidden.shape, (3, 4, 16))
                self.assertTrue(torch.isfinite(output.hidden).all())
                self.assertIsNotNone(output.loss)
                self.assertTrue(torch.isfinite(output.loss))
                for field, value in context.items():
                    self.assertIs(getattr(output, field), value)

    def test_decoder_block_rejects_a_context_free_layer_state(self):
        decoder = self.preset(decoder_num_layers=1)
        with self.assertRaises(TypeError):
            decoder(LayerState(hidden=torch.randn(3, 4, 16)))

    def test_expert_auxiliary_losses_accumulate_on_decoder_state(self):
        decoder = self.preset(TransformerExpertLinearConfigBuilder)
        state, _ = self.state()

        output = decoder(state)

        self.assertGreater(output.loss.detach().item(), 1.0)
        output.loss.backward()
        gradients = [
            parameter.grad
            for parameter in decoder.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
