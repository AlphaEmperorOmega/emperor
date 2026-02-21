import torch

from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.embedding.absolute.options.learned_embedding import (
        LearnedPositionalEmbedding,
    )
    from Emperor.embedding.absolute.options.sinusoidal_embedding import (
        SinusoidalPositionalEmbedding,
    )


class PositionalEmbeddingValidator:
    def ensure_propper_input_type(self, input: Tensor) -> None:
        if not torch.all(input == input.floor()):
            raise ValueError(
                f"Invalid input: Received input of type {input.dtype} ,expected tensor to be full of ints."
            )

    def ensure_propper_input_shape(self, input: Tensor) -> None:
        if input.dim() != 2:
            raise ValueError(
                f"Invalid input shape: got {input.shape}, expected the input to be a matrix of ints."
            )


class LearnedPositionalEmbeddingValidator(PositionalEmbeddingValidator):
    def __init__(
        self,
        model: "LearnedPositionalEmbedding",
    ):
        self.model = model

    def ensure_padding_index_exists_for_positions(
        self, positions: Tensor | None
    ) -> None:
        is_position = positions is not None
        is_padding_idx = self.model.padding_idx is not None
        if is_position or is_padding_idx:
            ValueError(
                "If positions is pre-computed then padding_idx should not be set."
            )


class SinusoidalPositionalEmbeddingValidator(PositionalEmbeddingValidator):
    def __init__(
        self,
        model: "SinusoidalPositionalEmbedding",
    ):
        self.model = model
        self.__ensure_values_are_not_none()
        self.__ensure_correct_input_types()

    def __ensure_values_are_not_none(self) -> None:
        required_attributes = [
            "embedding_dim",
            "num_embeddings",
            "init_size",
            "padding_idx",
            "auto_expand_flag",
        ]
        for attr_name in required_attributes:
            if getattr(self.model, attr_name) is None:
                raise ValueError(f"Configuration Error: '{attr_name}' is None.")

    def __ensure_correct_input_types(self) -> None:
        required_types = {
            "embedding_dim": int,
            "num_embeddings": int,
            "init_size": int,
            "padding_idx": int,
            "auto_expand_flag": bool,
        }

        for attr_name, expected_type in required_types.items():
            if not isinstance(getattr(self.model, attr_name), expected_type):
                raise TypeError(
                    f"Type Error: '{attr_name}' should be {expected_type.__name__}, but got {type(getattr(self.model, attr_name)).__name__}."
                )
