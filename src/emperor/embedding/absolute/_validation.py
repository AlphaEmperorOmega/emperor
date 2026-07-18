from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.embedding.absolute._base import AbsolutePositionalEmbeddingBase
    from emperor.embedding.absolute._config import (
        AbsolutePositionalEmbeddingConfig,
    )


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
        cls._validate_padding_idx(cfg)

    @staticmethod
    def _validate_padding_idx(
        cfg: "AbsolutePositionalEmbeddingConfig",
    ) -> None:
        padding_idx = cfg.padding_idx
        if padding_idx is not None and (
            not isinstance(padding_idx, int) or isinstance(padding_idx, bool)
        ):
            raise TypeError(
                f"padding_idx must be int for {type(cfg).__name__}, "
                f"got {type(padding_idx).__name__}"
            )
        if padding_idx is not None and padding_idx < 0:
            raise ValueError(
                f"padding_idx must be >= 0 when provided, received {padding_idx}"
            )

    @staticmethod
    def validate_padding_idx_bounds(
        *,
        padding_idx: int | None,
        num_embeddings: int,
        model_name: str,
    ) -> None:
        if padding_idx is not None and padding_idx >= num_embeddings:
            raise ValueError(
                f"padding_idx must be in [0, {num_embeddings}) for "
                f"{model_name}, received {padding_idx}"
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
        if not torch.all(torch.isfinite(input_tokens)):
            raise ValueError("input_tokens must contain finite values.")
        if not torch.all(input_tokens == input_tokens.floor()):
            raise ValueError(
                "input_tokens must contain integer-valued positions/tokens."
            )

    @staticmethod
    def validate_positions(
        positions: Tensor,
        *,
        expected_shape: tuple[int, ...],
        num_embeddings: int,
    ) -> None:
        if not isinstance(positions, Tensor):
            raise TypeError(
                f"positions must be a Tensor, got {type(positions).__name__}"
            )
        if tuple(positions.shape) != expected_shape:
            raise ValueError(
                f"positions must have shape {expected_shape}, "
                f"got {tuple(positions.shape)}"
            )
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"positions must use torch.int32 or torch.int64, got {positions.dtype}"
            )
        minimum = int(positions.min().item())
        maximum = int(positions.max().item())
        if minimum < 0 or maximum >= num_embeddings:
            raise ValueError(
                f"positions must be in [0, {num_embeddings}), "
                f"got range [{minimum}, {maximum}]"
            )

    @staticmethod
    def validate_patch_embeddings(
        patch_embeddings: Tensor,
        *,
        num_embeddings: int | None = None,
        embedding_dim: int | None = None,
    ) -> None:
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
        if num_embeddings is not None and patch_embeddings.size(1) != num_embeddings:
            raise ValueError(
                "patch_embeddings sequence dimension must be "
                f"{num_embeddings}, got {patch_embeddings.size(1)}"
            )
        if embedding_dim is not None and patch_embeddings.size(2) != embedding_dim:
            raise ValueError(
                f"patch_embeddings final dimension must be {embedding_dim}, "
                f"got {patch_embeddings.size(2)}"
            )
