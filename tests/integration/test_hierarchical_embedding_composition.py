import unittest

import torch

from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.transformer import TransformerDecoderLayerConfig
from tests.support.hierarchical_embedding import embedding_config, encoder_layer_config


class TestHierarchicalEmbeddingComposition(unittest.TestCase):
    def test_same_embedding_feeds_causal_gpt_and_bidirectional_bert_blocks(self):
        torch.manual_seed(17)
        embedding = embedding_config(output_dim=8).build().eval()
        bert = encoder_layer_config().build().eval()
        causal = encoder_layer_config(causal=True)
        gpt = (
            TransformerDecoderLayerConfig(
                embedding_dim=8,
                layer_norm_position=causal.layer_norm_position,
                dropout_probability=0.0,
                residual_config=causal.residual_config,
                self_attention_config=causal.attention_config,
                cross_attention_config=None,
                feed_forward_config=causal.feed_forward_config,
            )
            .build()
            .eval()
        )
        sentence_positions = TextLearnedPositionalEmbeddingConfig(
            num_embeddings=4, embedding_dim=8
        ).build()
        segments = torch.nn.Embedding(2, 8)
        token_ids_for_positions = torch.zeros(1, 4, dtype=torch.long)
        valid = torch.tensor([[True, True, True, False]])

        def consume(tokens):
            state = embedding(tokens, valid)
            positioned = state.hidden + sentence_positions(token_ids_for_positions)
            gpt_hidden, gpt_loss = gpt(
                target_token_embeddings=positioned, key_padding_mask=~valid
            )
            bert_hidden, bert_loss = bert(
                source_token_embeddings=positioned
                + segments(torch.tensor([[0, 0, 1, 1]])),
                source_key_padding_mask=~valid,
            )
            return gpt_hidden, bert_hidden, state.loss + gpt_loss + bert_loss

        original_gpt, original_bert, loss = consume([["Hello", "猫", "world", ""]])
        changed_gpt, changed_bert, _ = consume([["Hello", "猫", "future", ""]])
        self.assertEqual(original_gpt.shape, (1, 4, 8))
        self.assertEqual(original_bert.shape, (1, 4, 8))
        torch.testing.assert_close(original_gpt[:, :2], changed_gpt[:, :2])
        self.assertFalse(torch.allclose(original_bert[:, 0], changed_bert[:, 0]))
        (
            original_gpt[valid].square().mean()
            + original_bert[valid].square().mean()
            + loss
        ).backward()
        for module in (embedding, gpt, bert, sentence_positions, segments):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients)
            )


if __name__ == "__main__":
    unittest.main()
