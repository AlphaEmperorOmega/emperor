from torch import Tensor
from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.embedding.relative.core.config import RelativePositionalEmbeddingConfig


class RelativePositionalEmbeddingValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config", "padding_idx"}

    @staticmethod
    def validate_config(cfg: "RelativePositionalEmbeddingConfig") -> None:
        RelativePositionalEmbeddingValidator.validate_required_fields(cfg)
        RelativePositionalEmbeddingValidator.validate_field_types(cfg)
        RelativePositionalEmbeddingValidator.validate_dimensions(
            num_heads=cfg.num_heads,
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            init_size=cfg.init_size,
            max_positions=cfg.max_positions,
        )
        if cfg.padding_idx is not None and cfg.padding_idx < 0:
            raise ValueError(
                f"padding_idx must be >= 0 when provided, received {cfg.padding_idx}"
            )
        if cfg.embedding_dim % cfg.num_heads != 0:
            raise ValueError(
                f"embedding_dim must be divisible by num_heads, received "
                f"embedding_dim={cfg.embedding_dim}, num_heads={cfg.num_heads}"
            )

    @staticmethod
    def validate_forward_inputs(query: Tensor, sequence_length: int) -> None:
        if not isinstance(query, Tensor):
            raise TypeError(f"query must be a Tensor, got {type(query).__name__}")
        if query.dim() != 4:
            raise ValueError(
                f"query must be a 4D tensor (batch, heads, sequence, head_dim), "
                f"got shape {tuple(query.shape)}"
            )
        if sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be greater than 0, received {sequence_length}"
            )
