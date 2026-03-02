from torch import Tensor

from typing import TYPE_CHECKING

from emperor.attention.utils.enums import AttentionOptions

if TYPE_CHECKING:
    from emperor.attention.utils.handlers.reshaper import ReshaperBase as Reshaper


class ReshaperValidator:
    def __init__(
        self,
        reshaper: "Reshaper",
    ):
        self.model = reshaper
        self.__ensure_correct_head_dim()
        self.__ensure_perfect_projector_divisibility()

    def __ensure_correct_head_dim(self) -> None:
        assert (
            self.model.head_dim * self.model.num_heads
        ) == self.model.embedding_dim, (
            "`embedding_dim` must be perfectly divisible by `number_of_heads`."
        )

    def __ensure_perfect_projector_divisibility(self) -> None:
        if self.model.query_key_projection_dim != 0:
            if not self.model.query_key_projection_dim % self.model.num_heads == 0:
                raise ValueError(
                    f"`query_key_projection_dim` ({self.model.query_key_projection_dim}) must be perfectly divisible by `num_heads` ({self.model.num_heads})."
                )
        if self.model.value_projection_dim != 0:
            if not self.model.value_projection_dim % self.model.num_heads == 0:
                raise ValueError(
                    f"`value_projection_dim` ({self.model.value_projection_dim}) must be perfectly divisible by `num_heads` ({self.model.num_heads})."
                )
        if not self.model.embedding_dim % self.model.num_heads == 0:
            raise ValueError(
                f"`embedding_dim` ({self.model.embedding_dim}) must be perfectly divisible by `num_heads` ({self.model.num_heads})."
            )

    def ensure_not_self_attention(self) -> None:
        if self.model.attention_option == AttentionOptions.SELF_ATTENTION:
            raise ValueError(
                "This method should not be used when self attention is enabled."
            )

    def check_static_projection_shapes(
        self,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ):
        self.__resolve_static_projection_shape(static_keys)
        self.__resolve_static_projection_shape(static_values, True)

    def __resolve_static_projection_shape(
        self,
        static_tensor: Tensor | None = None,
        value_tensor_flag: bool = False,
    ) -> None:
        if static_tensor is not None:
            tensor_type = self.__resolve_static_projection_type(value_tensor_flag)
            expected_first_dim = self.model.batch_size * self.model.num_heads
            assert (
                static_tensor.size(0) == expected_first_dim
            ), f"expecting {tensor_type}.size(0) of {expected_first_dim}, but got {static_tensor.size(0)}"
            assert (
                static_tensor.size(2) == self.model.head_dim
            ), f"expecting {tensor_type}.size(2) of {self.model.head_dim}, but got {static_tensor.size(2)}"

    def __resolve_static_projection_type(self, value_tensor_flag: bool = False) -> str:
        return "static_values" if value_tensor_flag else "static_keys"
