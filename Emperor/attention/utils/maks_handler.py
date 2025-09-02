import torch
import torch.nn.functional as F

from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.attention import MultiHeadAttentionConfig


class Mask:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "Validator",
    ):
        self.cfg = cfg
        self.validator = validator
        self.target_dtype = self.cfg.target_dtype
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag

    def process_attention_masks(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        key_padding_mask = self.__canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attention_mask),
            other_name="attention_mask",
            target_type=self.target_dtype,
        )
        attention_mask = self.__validate_attention_mask(
            key_padding_mask,
            attention_mask,
        )

        return key_padding_mask, attention_mask

    def __validate_attention_mask(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        if (
            self.causal_attention_mask_flag
            and key_padding_mask is None
            and not self.return_attention_weights_flag
        ):
            return

        if key_padding_mask is not None:
            self.causal_attention_mask_flag = False

        attention_mask = self.__canonical_mask(
            mask=attention_mask,
            mask_name="attention_mask",
            other_type=None,
            other_name="",
            target_type=self.target_dtype,
            check_other=False,
        )

        return attention_mask

    def __canonical_mask(
        self,
        mask: Tensor | None,
        mask_name: str,
        other_type: "DType | None",
        other_name: str,
        target_type: "DType",
        check_other: bool = True,
    ) -> Tensor | None:
        if mask is None:
            return mask

        self.__ensure_mask_is_float_or_bool(mask, mask_name)
        self.__ensure_mask_is_correct_dtype(
            mask, mask_name, other_type, other_name, check_other
        )

        if not torch.is_floating_point(mask):
            mask_placeholder = torch.zeros_like(mask, dtype=target_type)
            mask = mask_placeholder.masked_fill_(mask, float("-inf"))
        return mask

    def __ensure_mask_is_float_or_bool(
        self,
        mask: Tensor,
        mask_name: str,
    ) -> None:
        is_float_float = torch.is_floating_point(mask)
        is_mask_bool = mask.dtype == torch.bool
        is_mask_float_or_bool = not is_mask_bool and not is_float_float
        if is_mask_float_or_bool:
            raise RuntimeError(
                f"only bool and floating types of {mask_name} are supported"
            )

    def __ensure_mask_is_correct_dtype(
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
