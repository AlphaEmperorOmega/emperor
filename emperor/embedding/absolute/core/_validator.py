import torch

from torch import Tensor
from emperor.base.validator import ValidatorBase
from emperor.embedding.absolute.core.config import (
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbeddingConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.embedding.absolute.core.config import (
        AbsolutePositionalEmbeddingConfig,
    )


class AbsolutePositionalEmbeddingValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config", "padding_idx"}

    @staticmethod
    def validate_config(cfg: "AbsolutePositionalEmbeddingConfig") -> None:
        AbsolutePositionalEmbeddingValidator.validate_required_fields(cfg)
        AbsolutePositionalEmbeddingValidator.validate_field_types(cfg)
        AbsolutePositionalEmbeddingValidator.validate_dimensions(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            init_size=cfg.init_size,
        )
        if cfg.padding_idx is not None and cfg.padding_idx < 0:
            raise ValueError(
                f"padding_idx must be >= 0 when provided, received {cfg.padding_idx}"
            )

    @staticmethod
    def validate_image_config(
        cfg: "ImageLearnedPositionalEmbeddingConfig | ImageSinusoidalPositionalEmbeddingConfig",
    ) -> None:
        AbsolutePositionalEmbeddingValidator.validate_config(cfg)
        if cfg.class_token_flag is None:
            raise ValueError(
                f"class_token_flag is required for {type(cfg).__name__}, "
                "received None"
            )
        if not isinstance(cfg.class_token_flag, bool):
            raise TypeError(
                f"class_token_flag must be bool for {type(cfg).__name__}, "
                f"got {type(cfg.class_token_flag).__name__}"
            )

    @staticmethod
    def validate_text_tokens(input_tokens: Tensor) -> None:
        if not isinstance(input_tokens, Tensor):
            raise TypeError(
                f"input_tokens must be a Tensor, got {type(input_tokens).__name__}"
            )
        if input_tokens.dim() != 2:
            raise ValueError(
                f"input_tokens must be a 2D tensor, got shape "
                f"{tuple(input_tokens.shape)}"
            )
        if not torch.all(input_tokens == input_tokens.floor()):
            raise ValueError(
                "input_tokens must contain integer-valued positions/tokens."
            )

    @staticmethod
    def validate_patch_embeddings(patch_embeddings: Tensor) -> None:
        if not isinstance(patch_embeddings, Tensor):
            raise TypeError(
                f"patch_embeddings must be a Tensor, got "
                f"{type(patch_embeddings).__name__}"
            )
        if patch_embeddings.dim() != 3:
            raise ValueError(
                f"patch_embeddings must be a 3D tensor, got shape "
                f"{tuple(patch_embeddings.shape)}"
            )
