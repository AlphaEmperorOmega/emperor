from typing import TYPE_CHECKING

from torch import Tensor

from emperor.base.validator import ValidatorBase

if TYPE_CHECKING:
    from emperor.embedding.relative.core.layers import DynamicPositionalBias


class RelativePositionalEmbeddingValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config", "padding_idx"}

    @classmethod
    def validate(cls, model: "DynamicPositionalBias") -> None:
        cfg = model.cfg
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_dimensions(
            num_heads=cfg.num_heads,
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            init_size=cfg.init_size,
            max_positions=cfg.max_positions,
        )
        cls._validate_padding_idx(cfg.padding_idx)
        cls._validate_head_dimensions(cfg.embedding_dim, cfg.num_heads)

    @staticmethod
    def _validate_padding_idx(padding_idx: int | None) -> None:
        if padding_idx is not None and padding_idx < 0:
            raise ValueError(
                f"padding_idx must be >= 0 when provided, received {padding_idx}"
            )

    @staticmethod
    def _validate_head_dimensions(embedding_dim: int, num_heads: int) -> None:
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim must be divisible by num_heads, received "
                f"embedding_dim={embedding_dim}, num_heads={num_heads}"
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
