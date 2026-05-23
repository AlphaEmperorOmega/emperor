import torch

from torch import Tensor
from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.layers import MultiHeadAttentionAbstract


class AttentionValidatorBase:
    @staticmethod
    def validate_head_divisibility(model: "MultiHeadAttentionAbstract") -> None:
        if model.embedding_dim % model.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({model.embedding_dim}) must be perfectly divisible "
                f"by num_heads ({model.num_heads})."
            )
        if (
            model.query_key_projection_dim
            and model.query_key_projection_dim % model.num_heads != 0
        ):
            raise ValueError(
                f"query_key_projection_dim ({model.query_key_projection_dim}) must be "
                f"perfectly divisible by num_heads ({model.num_heads})."
            )
        if (
            model.value_projection_dim
            and model.value_projection_dim % model.num_heads != 0
        ):
            raise ValueError(
                f"value_projection_dim ({model.value_projection_dim}) must be "
                f"perfectly divisible by num_heads ({model.num_heads})."
            )

    @staticmethod
    def validate_attention_weights_returned_for_self_attention_only(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        if model.return_attention_weights_flag:
            raise RuntimeError(
                "attention_weights can be returned only when self attention is "
                "computed; ensure the query, key, and value tensors are the same tensor."
            )

    @staticmethod
    def validate_input_shapes(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> None:
        if query.dim() not in (2, 3):
            raise RuntimeError(
                f"Query should be an unbatched 2D or batched 3D tensor but received "
                f"a {query.dim()}-D tensor."
            )
        batched = query.dim() == 3
        expected_dims = 3 if batched else 2
        if key.dim() != expected_dims or value.dim() != expected_dims:
            raise RuntimeError(
                f"For a {'batched (3-D)' if batched else 'unbatched (2-D)'} query, "
                f"expected key and value to be {expected_dims}-D but found "
                f"{key.dim()}-D and {value.dim()}-D tensors respectively."
            )
        if key_padding_mask is not None:
            expected_mask_dim = 2 if batched else 1
            if key_padding_mask.dim() != expected_mask_dim:
                raise RuntimeError(
                    f"For a {'batched (3-D)' if batched else 'unbatched (2-D)'} query, "
                    f"expected key_padding_mask to be None or {expected_mask_dim}-D but "
                    f"found a {key_padding_mask.dim()}-D tensor instead."
                )
        if attention_mask is not None and attention_mask.dim() not in (2, 3):
            raise RuntimeError(
                f"Expected attention_mask to be None, 2-D, or 3-D but found a "
                f"{attention_mask.dim()}-D tensor instead."
            )

    @staticmethod
    def validate_key_value_projection_shapes(key: Tensor, value: Tensor) -> None:
        if key.shape[:-1] != value.shape[:-1]:
            raise RuntimeError(
                f"key shape {tuple(key.shape)} does not match value shape "
                f"{tuple(value.shape)} on the sequence and batch dimensions."
            )

    @staticmethod
    def validate_static_projection_shapes(
        model: "MultiHeadAttentionAbstract",
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ) -> None:
        AttentionValidatorBase._validate_static_projection_shape(
            model, static_keys, "static_keys"
        )
        AttentionValidatorBase._validate_static_projection_shape(
            model, static_values, "static_values"
        )

    @staticmethod
    def _validate_static_projection_shape(
        model: "MultiHeadAttentionAbstract",
        static_tensor: Tensor | None,
        tensor_name: str,
    ) -> None:
        if static_tensor is None:
            return
        expected_first_dim = model.batch_size * model.num_heads
        if static_tensor.size(0) != expected_first_dim:
            raise ValueError(
                f"expecting {tensor_name}.size(0) of {expected_first_dim}, but got "
                f"{static_tensor.size(0)}."
            )
        if static_tensor.size(2) != model.head_dim:
            raise ValueError(
                f"expecting {tensor_name}.size(2) of {model.head_dim}, but got "
                f"{static_tensor.size(2)}."
            )

    @staticmethod
    def validate_mask_is_float_or_bool(mask: Tensor, mask_name: str) -> None:
        is_float = torch.is_floating_point(mask)
        is_bool = mask.dtype == torch.bool
        if not is_float and not is_bool:
            raise RuntimeError(
                f"Only bool and floating types of {mask_name} are supported."
            )

    @staticmethod
    def validate_mask_dtype_matches(
        mask: Tensor,
        mask_name: str,
        other_type,
        other_name: str,
        check_other: bool = True,
    ) -> None:
        if check_other and other_type is not None and mask.dtype != other_type:
            raise RuntimeError(
                f"Support for mismatched {mask_name} and {other_name} is deprecated. "
                "Use the same type for both instead."
            )

    @staticmethod
    def validate_attention_mask_for_required_causal_mask(
        attention_mask: Tensor | None,
        causal_attention_mask_flag: bool,
    ) -> None:
        if causal_attention_mask_flag and attention_mask is None:
            raise RuntimeError(
                "Need an attention_mask when the causal_attention_mask_flag is set. "
                "Use the Transformer module method generate_square_subsequent_mask to "
                "create this mask."
            )


class MultiHeadAttentionValidator(AttentionValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {"relative_positional_embedding_config"}

    @staticmethod
    def validate(model: "MultiHeadAttentionAbstract") -> None:
        MultiHeadAttentionValidator.validate_required_fields(model.cfg)
        MultiHeadAttentionValidator.validate_field_types(model.cfg)
        MultiHeadAttentionValidator.validate_dimensions(
            batch_size=model.batch_size,
            num_heads=model.num_heads,
            embedding_dim=model.embedding_dim,
        )
        AttentionValidatorBase.validate_head_divisibility(model)

    @staticmethod
    def validate_forward_inputs(
        model: "MultiHeadAttentionAbstract",
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> None:
        AttentionValidatorBase.validate_input_shapes(
            query, key, value, key_padding_mask, attention_mask
        )
