import unittest

import torch

from tests.support.hierarchical_decoding import decoder_config
from tests.support.hierarchical_embedding import embedding_config


def language_model_config():
    from emperor.decoding.hierarchical import HierarchicalLanguageModelConfig
    from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig

    return HierarchicalLanguageModelConfig(
        sequence_length=4,
        embedding_config=embedding_config(output_dim=8, max_bytes=8),
        backbone_config=decoder_config(
            conditioning_dim=8
        ).decoder_config.decoder_stack_config,
        decoding_config=decoder_config(conditioning_dim=8),
        position_config=TextLearnedPositionalEmbeddingConfig(
            num_embeddings=4, embedding_dim=8
        ),
        dropout_probability=0.0,
    )


class HierarchicalLanguageModelTests(unittest.TestCase):
    def test_invalid_bos_masks_do_not_silently_discard_context(self):
        from dataclasses import replace

        from emperor.decoding.hierarchical import (
            HierarchicalLanguageModelBatch,
            HierarchicalTextCodec,
        )

        model = language_model_config().build()
        batch = HierarchicalLanguageModelBatch.collate(
            list(HierarchicalTextCodec(8).training_windows("a", 4))
        )
        with self.assertRaisesRegex(TypeError, "bos_mask"):
            model(replace(batch, bos_mask=[[True, False]]))
        with self.assertRaisesRegex(ValueError, "BOS context"):
            model(replace(batch, context_texts=(("discard", "a"),)))
        with self.assertRaisesRegex(ValueError, "context_texts"):
            model(replace(batch, context_texts=(("",),)))


    def test_generation_rolls_context_and_preserves_partial_prompts(self):
        model = language_model_config().build().train()
        model.encoder.byte_position.eval()
        with torch.no_grad():
            for parameter in model.decoder.output_projection.parameters():
                parameter.zero_()
            model.decoder.output_projection.layers[0].model.bias_params[97] = 10
        generated = model.generate_text("", max_new_tokens=6, max_new_bytes=100)
        self.assertEqual(generated.text, "a" * 48)
        self.assertEqual(generated.stop_reason, "token_budget")
        partial = model.generate_text("café", max_new_tokens=2, max_new_bytes=5)
        self.assertEqual(partial.text, "café" + "a" * 5)
        self.assertEqual(partial.stop_reason, "byte_budget")
        self.assertTrue(model.training)
        self.assertFalse(model.encoder.byte_position.training)
        with torch.no_grad():
            model.decoder.output_projection.layers[0].model.bias_params[257] = 30
        eos = model.generate_text("complete ", max_new_bytes=8)
        self.assertEqual(eos.text, "complete ")
        self.assertEqual(eos.stop_reason, "end_of_document")
        explicit = model.generate_text("word", prompt_is_complete=True)
        self.assertEqual(explicit.text, "word")
        self.assertEqual(explicit.stop_reason, "end_of_document")

    def test_bos_shifting_causality_and_padding_independence(self):
        from emperor.decoding.hierarchical import (
            HierarchicalLanguageModelBatch,
            HierarchicalTextCodec,
        )

        torch.manual_seed(3)
        model = language_model_config().build().eval()
        codec = HierarchicalTextCodec(8)
        first = list(codec.training_windows("mat cat ", 4))[0]
        other = list(codec.training_windows("dog ", 4))[0]
        single = model(HierarchicalLanguageModelBatch.collate([first]))
        combined = model(HierarchicalLanguageModelBatch.collate([first, other]))
        torch.testing.assert_close(single.logits, combined.logits[: len(single.logits)])
        changed = list(codec.training_windows("mat pig ", 4))[0]
        future = model(HierarchicalLanguageModelBatch.collate([changed]))
        torch.testing.assert_close(single.logits[:6], future.logits[:6])
        batch = HierarchicalLanguageModelBatch.collate([first, other])
        loss = (
            torch.nn.functional.cross_entropy(combined.logits, batch.labels)
            + combined.loss
        )
        loss.backward()
        self.assertGreater(model.beginning_of_document.grad.abs().sum(), 0)
