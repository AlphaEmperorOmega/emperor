from __future__ import annotations

import unittest

import torch
from emperor.embedding.absolute import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)


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

    def test_text_learned_supports_nonzero_padding_and_incremental_batches(
        self,
    ) -> None:
        config = TextLearnedPositionalEmbeddingConfig(
            num_embeddings=3,
            embedding_dim=2,
            init_size=3,
            padding_idx=2,
            auto_expand_flag=False,
        )
        model = config.build()
        expected_weights = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, -1.0],
                [0.0, 0.0],
                [3.0, -3.0],
                [4.0, -4.0],
                [5.0, -5.0],
            ]
        )
        with torch.no_grad():
            model.embedding_model.weight.copy_(expected_weights)
        tokens = torch.tensor(
            [
                [9, 9, 2, 9],
                [9, 2, 9, 9],
            ]
        )

        full_output = model(tokens)
        incremental_tokens = tokens[:, :2].to(torch.int32)
        make_incremental_position = (
            model._TextLearnedPositionalEmbedding__make_incremental_position
        )
        incremental_positions = make_incremental_position(incremental_tokens)
        incremental_output = model(
            incremental_tokens,
            incremental_state={},
        )

        torch.testing.assert_close(
            full_output,
            expected_weights[torch.tensor([[3, 4, 2, 5], [3, 2, 4, 5]])],
            rtol=0,
            atol=0,
        )
        self.assertEqual(incremental_output.shape, (2, 1, 2))
        self.assertEqual(incremental_positions.dtype, torch.long)
        torch.testing.assert_close(
            incremental_positions,
            torch.full((2, 1), 4, dtype=torch.long),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            incremental_output,
            expected_weights[4].reshape(1, 1, 2).expand(2, -1, -1),
            rtol=0,
            atol=0,
        )

        zero_padding_model = TextLearnedPositionalEmbeddingConfig(
            num_embeddings=3,
            embedding_dim=2,
            init_size=3,
            padding_idx=0,
            auto_expand_flag=False,
        ).build()
        zero_padding_weights = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
        with torch.no_grad():
            zero_padding_model.embedding_model.weight.copy_(zero_padding_weights)

        zero_padding_output = zero_padding_model(
            torch.ones(2, 2, dtype=torch.int32),
            incremental_state={},
        )

        torch.testing.assert_close(
            zero_padding_output,
            zero_padding_weights[2].reshape(1, 1, 2).expand(2, -1, -1),
            rtol=0,
            atol=0,
        )

    def test_text_learned_incremental_position_without_padding_is_zero_based(
        self,
    ) -> None:
        model = TextLearnedPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=2,
            init_size=4,
            padding_idx=None,
            auto_expand_flag=False,
        ).build()
        weights = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        )
        with torch.no_grad():
            model.embedding_model.weight.copy_(weights)

        output = model(
            torch.tensor([[9, 8, 7], [6, 5, 4]], dtype=torch.int32),
            incremental_state={},
        )

        self.assertEqual(output.shape, (2, 1, 2))
        torch.testing.assert_close(
            output,
            weights[2].reshape(1, 1, 2).expand(2, -1, -1),
            rtol=0,
            atol=0,
        )

    def test_incremental_text_embeddings_reject_empty_sequences(self) -> None:
        configs = (
            TextLearnedPositionalEmbeddingConfig(
                num_embeddings=4,
                embedding_dim=2,
                init_size=4,
                padding_idx=0,
                auto_expand_flag=False,
            ),
            TextSinusoidalPositionalEmbeddingConfig(
                num_embeddings=4,
                embedding_dim=2,
                init_size=4,
                padding_idx=0,
                auto_expand_flag=False,
            ),
        )

        for config in configs:
            with self.subTest(config=type(config).__name__):
                model = config.build()
                with self.assertRaises(ValueError) as error:
                    model(
                        torch.empty((2, 0), dtype=torch.long),
                        incremental_state={},
                    )
                self.assertEqual(
                    str(error.exception),
                    "incremental positional embedding requires a non-empty "
                    "input sequence.",
                )


if __name__ == "__main__":
    unittest.main()
