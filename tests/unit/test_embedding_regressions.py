from __future__ import annotations

import unittest

import torch
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig


class AbsoluteEmbeddingRegressionTests(unittest.TestCase):
    def test_empty_explicit_positions_return_an_empty_embedding_sequence(
        self,
    ) -> None:
        model = TextLearnedPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=2,
            init_size=4,
            padding_idx=0,
            auto_expand_flag=False,
        ).build()
        tokens = torch.empty((2, 0), dtype=torch.long)
        positions = torch.empty((2, 0), dtype=torch.int32)

        output = model(tokens, positions=positions)

        self.assertEqual(output.shape, (2, 0, 2))
        self.assertEqual(output.dtype, model.embedding_model.weight.dtype)
        self.assertEqual(output.device, model.embedding_model.weight.device)
        self.assertEqual(output.numel(), 0)


if __name__ == "__main__":
    unittest.main()
