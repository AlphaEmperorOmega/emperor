from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.embedding.relative._bias import DynamicPositionalBias
    from emperor.embedding.relative._config import DynamicPositionalBiasConfig


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
        cls._validate_padding_idx(cfg)
        cls._validate_head_dimensions(cfg.embedding_dim, cfg.num_heads)

    @staticmethod
    def _validate_padding_idx(
        cfg: "DynamicPositionalBiasConfig",
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
    def _validate_head_dimensions(embedding_dim: int, num_heads: int) -> None:
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim must be divisible by num_heads, received "
                f"embedding_dim={embedding_dim}, num_heads={num_heads}"
            )

    @staticmethod
    def validate_forward_inputs(
        query: Tensor,
        sequence_length: int,
        *,
        last: bool = False,
        num_heads: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        if not isinstance(query, Tensor):
            raise TypeError(f"query must be a Tensor, got {type(query).__name__}")
        if query.dim() != 4:
            raise ValueError(
                f"query must be a 4D tensor (batch, heads, sequence, head_dim), "
                f"got shape {tuple(query.shape)}"
            )
        if type(sequence_length) is not int:
            raise TypeError(
                f"sequence_length must be int, got {type(sequence_length).__name__}"
            )
        if type(last) is not bool:
            raise TypeError(f"last must be bool, got {type(last).__name__}")
        if sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be greater than 0, received {sequence_length}"
            )
        if num_heads is not None and query.size(1) != num_heads:
            raise ValueError(
                f"query head dimension must contain {num_heads} heads, "
                f"got {query.size(1)}"
            )
        if head_dim is not None and query.size(3) != head_dim:
            raise ValueError(
                f"query final dimension must be {head_dim}, got {query.size(3)}"
            )
        target_sequence_length = query.size(2)
        if target_sequence_length <= 0:
            raise ValueError(
                "query target sequence dimension must be greater than 0, "
                f"got {target_sequence_length}"
            )
        if last and target_sequence_length != 1:
            raise ValueError(
                "last=True requires a target sequence length of 1, "
                f"got {target_sequence_length}"
            )
