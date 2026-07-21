from __future__ import annotations

import unittest

import torch

from emperor.embedding.absolute import (
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.embedding.relative import DynamicPositionalBiasConfig


class AbsoluteEmbeddingRegressionTests(unittest.TestCase):
    def test_text_inputs_and_explicit_positions_are_validated_exactly(
        self,
    ) -> None:
        model = TextLearnedPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=2,
            init_size=4,
            padding_idx=0,
            auto_expand_flag=False,
        ).build()
        tokens = torch.ones(2, 3, dtype=torch.long)

        with self.assertRaises(ValueError) as finite_error:
            model(torch.tensor([[1.0, float("inf")]]))
        self.assertEqual(
            str(finite_error.exception),
            "input_tokens must contain finite values.",
        )

        with self.assertRaises(TypeError) as position_type_error:
            model(tokens, positions=[[1, 2, 3], [1, 2, 3]])
        self.assertEqual(
            str(position_type_error.exception),
            "positions must be a Tensor, got list",
        )

        invalid_positions = (
            (
                torch.ones(1, 3, dtype=torch.long),
                "positions must have shape (2, 3), got (1, 3)",
            ),
            (
                torch.ones(2, 3),
                "positions must use torch.int32 or torch.int64, got torch.float32",
            ),
            (
                torch.tensor([[1, 2, 5], [1, 2, 3]]),
                "positions must be in [0, 5), got range [1, 5]",
            ),
        )
        for positions, message in invalid_positions:
            with self.subTest(message=message):
                with self.assertRaises((TypeError, ValueError)) as error:
                    model(tokens, positions=positions)
                self.assertEqual(str(error.exception), message)

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

    def test_image_embeddings_reject_sequence_and_feature_mismatches(
        self,
    ) -> None:
        for config_type in (
            ImageLearnedPositionalEmbeddingConfig,
            ImageSinusoidalPositionalEmbeddingConfig,
        ):
            with self.subTest(config_type=config_type.__name__):
                model = config_type(
                    num_embeddings=3,
                    embedding_dim=2,
                    init_size=3,
                    padding_idx=(
                        None
                        if config_type is ImageSinusoidalPositionalEmbeddingConfig
                        else 0
                    ),
                    auto_expand_flag=False,
                    class_token_flag=False,
                ).build()

                with self.assertRaises(ValueError) as sequence_error:
                    model(torch.zeros(2, 4, 2))
                self.assertEqual(
                    str(sequence_error.exception),
                    "patch_embeddings sequence dimension must be 3, got 4",
                )

                with self.assertRaises(ValueError) as feature_error:
                    model(torch.zeros(2, 3, 4))
                self.assertEqual(
                    str(feature_error.exception),
                    "patch_embeddings final dimension must be 2, got 4",
                )

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
        weights = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
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

    def test_text_sinusoidal_none_padding_preserves_position_zero(self) -> None:
        model = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=2,
            init_size=4,
            padding_idx=None,
            auto_expand_flag=False,
        ).build()

        output = model(torch.tensor([[0, 7, 8]]))

        self.assertIsNone(model.padding_idx)
        torch.testing.assert_close(
            output,
            torch.tensor(
                [
                    [
                        [0.0, 1.0],
                        [torch.sin(torch.tensor(1.0)), torch.cos(torch.tensor(1.0))],
                        [torch.sin(torch.tensor(2.0)), torch.cos(torch.tensor(2.0))],
                    ]
                ]
            ),
        )

    def test_image_sinusoidal_without_class_token_matches_patch_count(
        self,
    ) -> None:
        model = ImageSinusoidalPositionalEmbeddingConfig(
            num_embeddings=3,
            embedding_dim=2,
            init_size=3,
            padding_idx=None,
            auto_expand_flag=False,
            class_token_flag=False,
        ).build()
        patches = torch.zeros(2, 3, 2)

        output = model(patches)

        self.assertEqual(model.weights.shape, (3, 2))
        self.assertEqual(output.shape, (2, 3, 2))
        torch.testing.assert_close(output[0], model.weights)
        torch.testing.assert_close(output[1], model.weights)

    def test_incremental_auto_expansion_uses_timestep_and_preserves_dtype(
        self,
    ) -> None:
        model = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=4,
            init_size=2,
            padding_idx=0,
            auto_expand_flag=True,
        ).build()
        model.double()

        output = model(
            torch.ones(3, 1, dtype=torch.long),
            incremental_state={},
            timestep=torch.tensor([5]),
        )

        self.assertEqual(output.shape, (3, 1, 4))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(model.weights.dtype, torch.float64)
        self.assertEqual(model.weights.size(0), 7)
        torch.testing.assert_close(
            output,
            model.weights[6].reshape(1, 1, -1).expand(3, -1, -1),
            rtol=0,
            atol=0,
        )

    def test_incremental_sinusoidal_accepts_zero_scalar_timestep(self) -> None:
        model = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=2,
            init_size=4,
            padding_idx=0,
            auto_expand_flag=False,
        ).build()

        output = model(
            torch.ones(2, 1, dtype=torch.long),
            incremental_state={},
            timestep=torch.tensor(0, dtype=torch.int32),
        )

        torch.testing.assert_close(
            output,
            model.weights[1].reshape(1, 1, 2).expand(2, -1, -1),
            rtol=0,
            atol=0,
        )

    def test_incremental_sinusoidal_without_timestep_uses_last_input_step(
        self,
    ) -> None:
        model = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=2,
            init_size=4,
            padding_idx=0,
            auto_expand_flag=False,
        ).build()

        output = model(
            torch.ones(2, 3, dtype=torch.long),
            incremental_state={},
        )

        expected_row = torch.tensor(
            [torch.sin(torch.tensor(3.0)), torch.cos(torch.tensor(3.0))]
        )
        torch.testing.assert_close(
            output,
            expected_row.reshape(1, 1, 2).expand(2, -1, -1),
            rtol=0,
            atol=0,
        )

    def test_incremental_sinusoidal_rejects_negative_timestep(self) -> None:
        model = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=2,
            init_size=4,
            padding_idx=0,
            auto_expand_flag=True,
        ).build()

        with self.assertRaises(ValueError) as error:
            model(
                torch.ones(2, 1, dtype=torch.long),
                incremental_state={},
                timestep=torch.tensor([-1]),
            )

        self.assertEqual(
            str(error.exception),
            "timestep must be non-negative, received -1",
        )

    def test_incremental_sinusoidal_rejects_malformed_timesteps(self) -> None:
        model = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=2,
            init_size=4,
            padding_idx=0,
            auto_expand_flag=True,
        ).build()
        invalid_cases = (
            (
                [1],
                TypeError,
                "timestep must be a Tensor, got list",
            ),
            (
                torch.tensor([[1]]),
                ValueError,
                "timestep must be a scalar or one-dimensional tensor, got shape (1, 1)",
            ),
            (
                torch.tensor([], dtype=torch.long),
                ValueError,
                "timestep must contain exactly one value, got 0",
            ),
            (
                torch.tensor([1, 2]),
                ValueError,
                "timestep must contain exactly one value, got 2",
            ),
            (
                torch.tensor([1.5]),
                TypeError,
                "timestep must use torch.int32 or torch.int64, got torch.float32",
            ),
            (
                torch.tensor([True]),
                TypeError,
                "timestep must use torch.int32 or torch.int64, got torch.bool",
            ),
        )

        for timestep, error_type, message in invalid_cases:
            with self.subTest(timestep=timestep):
                with self.assertRaises(error_type) as error:
                    model(
                        torch.ones(2, 1, dtype=torch.long),
                        incremental_state={},
                        timestep=timestep,
                    )
                self.assertEqual(str(error.exception), message)


class RelativeEmbeddingRegressionTests(unittest.TestCase):
    def test_relative_forward_validates_modes_and_query_dimensions(self) -> None:
        model = DynamicPositionalBiasConfig(
            text_processing_flag=True,
            num_heads=2,
            num_embeddings=5,
            embedding_dim=4,
            init_size=5,
            padding_idx=None,
            auto_expand_flag=False,
            max_positions=2,
        ).build()
        valid_query = torch.ones(1, 2, 3, 2)
        invalid_cases = (
            (
                valid_query,
                True,
                False,
                TypeError,
                "sequence_length must be int, got bool",
            ),
            (
                valid_query,
                3,
                None,
                TypeError,
                "last must be bool, got NoneType",
            ),
            (
                valid_query,
                3,
                1,
                TypeError,
                "last must be bool, got int",
            ),
            (
                torch.ones(1, 1, 3, 2),
                3,
                False,
                ValueError,
                "query head dimension must contain 2 heads, got 1",
            ),
            (
                torch.ones(1, 2, 3, 3),
                3,
                False,
                ValueError,
                "query final dimension must be 2, got 3",
            ),
            (
                torch.ones(1, 2, 0, 2),
                3,
                False,
                ValueError,
                "query target sequence dimension must be greater than 0, got 0",
            ),
            (
                valid_query,
                3,
                True,
                ValueError,
                "last=True requires a target sequence length of 1, got 3",
            ),
        )
        for query, sequence_length, last, error_type, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as error:
                    model(query, sequence_length, last=last)
                self.assertEqual(str(error.exception), message)

    def test_relative_distance_clamping_uses_both_endpoint_bins(self) -> None:
        model = DynamicPositionalBiasConfig(
            text_processing_flag=True,
            num_heads=1,
            num_embeddings=5,
            embedding_dim=1,
            init_size=5,
            padding_idx=None,
            auto_expand_flag=False,
            max_positions=2,
        ).build()
        with torch.no_grad():
            model.relative_positional_embeddings.copy_(
                torch.tensor([[[10.0, 20.0, 30.0, 40.0, 50.0]]])
            )

        negative_output = model(
            torch.ones(1, 1, 1, 1),
            sequence_length=5,
            last=True,
        )
        positive_output = model(
            torch.ones(1, 1, 1, 1),
            sequence_length=5,
        )

        torch.testing.assert_close(
            negative_output,
            torch.tensor([[[[10.0, 10.0, 10.0, 20.0, 30.0]]]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            positive_output,
            torch.tensor([[[[30.0, 40.0, 50.0, 50.0, 50.0]]]]),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
