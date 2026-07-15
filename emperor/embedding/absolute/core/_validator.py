from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.base.validator import ValidatorBase
from emperor.embedding.absolute.core.config import (
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbeddingConfig,
)

if TYPE_CHECKING:
    from emperor.embedding.absolute.core.layers import AbsolutePositionalEmbeddingBase


class AbsolutePositionalEmbeddingValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config", "padding_idx"}

    @classmethod
    def validate(cls, model: "AbsolutePositionalEmbeddingBase") -> None:
        cfg = model.cfg
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_dimensions(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            init_size=cfg.init_size,
        )
        cls._validate_padding_idx(cfg.padding_idx)
        if isinstance(
            cfg,
            (
                ImageLearnedPositionalEmbeddingConfig,
                ImageSinusoidalPositionalEmbeddingConfig,
            ),
        ):
            cls._validate_class_token_flag(cfg)

    @staticmethod
    def _validate_padding_idx(padding_idx: int | None) -> None:
        if padding_idx is not None and padding_idx < 0:
            raise ValueError(
                f"padding_idx must be >= 0 when provided, received {padding_idx}"
            )

    @staticmethod
    def _validate_class_token_flag(
        cfg: (
            ImageLearnedPositionalEmbeddingConfig
            | ImageSinusoidalPositionalEmbeddingConfig
        ),
    ) -> None:
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
