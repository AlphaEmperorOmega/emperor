import torch

from torch.types import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.handlers.maks import Mask
    from Emperor.attention.utils.handlers.projector import IndependentProjector
    from Emperor.attention.utils.handlers.projector import SelfAttentionProjector


class MaskValidator:
    def __init__(self, model: "Mask"):
        self.model = model

    def ensure_mask_is_float_or_bool(
        self,
        mask: Tensor,
        mask_name: str,
    ) -> None:
        is_float_float = torch.is_floating_point(mask)
        is_mask_bool = mask.dtype == torch.bool
        is_mask_float_or_bool = not is_mask_bool and not is_float_float
        if is_mask_float_or_bool:
            raise RuntimeError(
                f"Only bool and floating types of {mask_name} are supported"
            )

    def ensure_mask_is_correct_dtype(
        self,
        mask: Tensor,
        mask_name: str,
        other_type: "DType | None",
        other_name: str,
        check_other: bool = True,
    ):
        mask_dtype = mask.dtype
        should_check_other_dtype = check_other and other_type is not None
        if should_check_other_dtype:
            does_dtype_match = mask_dtype == other_type
            if not does_dtype_match:
                raise RuntimeError(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )

    def ensure_attention_mask_for_required_causal_mask(
        self,
        attention_mask: Tensor | None = None,
        causal_attention_mask_flag: bool = False,
    ) -> None:
        if causal_attention_mask_flag and attention_mask is None:
            raise RuntimeError(
                "Need `attention_mask` if specifying the `causal_attention_mask_flag` hint."
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )


class SelfAttentionProjectorValidator:
    def __init__(self, model: "SelfAttentionProjector"):
        self.model = model

    def ensure_qkv_are_equal_for_self_attention(
        self, key: Tensor, query: Tensor, value: Tensor
    ):
        are_qkv_same = key is value and query is key
        if not are_qkv_same:
            raise RuntimeError(
                "Self attention can only be computed when `query`, `key`, and `value` are the same tensor."
            )


class IndependentProjectorValidator:
    def __init__(self, model: "IndependentProjector"):
        self.model = model

    def ensure_attention_weights_returned_for_self_attention_only(self):
        if self.model.return_attention_weights_flag:
            raise RuntimeError(
                "`attention_weights` can be returned only when self attention is computed, ensure that `is_self_attention_projector_flag` is set to `False` and the `query`, `key` and `value` tensors are the same tensor."
            )

    def ensure_propper_kv_shapes_for_independent_projector(
        self, key: Tensor, value: Tensor
    ) -> None:
        k_sequence_length, k_batch_size, _ = key.shape
        v_sequence_length, v_batch_size, _ = value.shape
        is_kv_sequence_length_same = k_sequence_length == v_sequence_length
        is_kv_batch_size_same = k_batch_size == v_batch_size
        if not (is_kv_sequence_length_same and is_kv_batch_size_same):
            raise RuntimeError(
                f"key shape {key.shape} does not match value shape {value.shape}"
            )
