import math
import tempfile
import unittest

import torch
from lightning import Trainer
from torch.utils.data import DataLoader

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

    def test_metrics_accumulate_symbol_and_byte_counts_across_batches(self):
        from emperor.config import ModelConfig
        from emperor.decoding.hierarchical import (
            HierarchicalByteDecoderOutput,
            HierarchicalLanguageModelBatch,
            HierarchicalTextCodec,
        )
        from emperor.experiments.language_model import LanguageModelExperiment

        class UniformModel(LanguageModelExperiment):
            def __init__(self):
                super().__init__(ModelConfig(learning_rate=0.01, output_dim=258))
                self.scores = torch.nn.Parameter(torch.zeros(258))

            def forward(self, batch):
                counts = torch.where(
                    batch.attention_mask, batch.byte_lengths + 1, 0
                ).flatten()
                offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
                return HierarchicalByteDecoderOutput(
                    self.scores.expand(int(offsets[-1]), -1),
                    offsets,
                    self.scores.new_tensor(0.25),
                )

        codec = HierarchicalTextCodec(8)
        windows = [*codec.training_windows("", 4), *codec.training_windows("é", 4)]
        loader = DataLoader(
            windows, batch_size=1, collate_fn=HierarchicalLanguageModelBatch.collate
        )
        with tempfile.TemporaryDirectory() as root:
            trainer = Trainer(
                default_root_dir=root,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            result = trainer.validate(
                UniformModel(), dataloaders=loader, verbose=False
            )[0]
        self.assertAlmostEqual(
            result["validation/symbol_cross_entropy"], math.log(258), places=5
        )
        # Empty doc: EOS; nonempty doc: two UTF-8 bytes, EOW, EOS.
        self.assertAlmostEqual(
            result["validation/bits_per_byte"], 5 * math.log2(258) / 2, places=5
        )
        self.assertAlmostEqual(result["validation/auxiliary_loss"], 0.25, places=5)
        self.assertAlmostEqual(
            result["validation/loss"], math.log(258) + 0.25, places=5
        )
        empty_loader = DataLoader(
            windows[:1], batch_size=1, collate_fn=HierarchicalLanguageModelBatch.collate
        )
        empty = trainer.validate(
            UniformModel(), dataloaders=empty_loader, verbose=False
        )[0]
        self.assertTrue(math.isnan(empty["validation/bits_per_byte"]))
        self.assertAlmostEqual(
            empty["validation/symbol_cross_entropy"], math.log(258), places=5
        )

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
