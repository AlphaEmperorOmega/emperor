import torch

from torch.types import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.handlers.maks import Mask


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
