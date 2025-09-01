from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.attention import MultiHeadAttentionConfig


class BatchDimensionManager:
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.should_transpose_first_two_dims = False

    def enforce_batch_as_second_dim(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        is_input_batched = query.dim() == 3
        _, batch_size, _ = query.size()
        is_expected_batch_size = batch_size != self.batch_size

        self.should_transpose_first_two_dims = (
            is_expected_batch_size and is_input_batched
        )
        if not self.should_transpose_first_two_dims:
            return query, key, value
        if key is value:
            return self.__transpose_shared_tensors(query, key)
        return (tensor.transpose(0, 1) for tensor in (query, key, value))

    def __transpose_shared_tensors(
        self, query: Tensor, key: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if query is key:
            query = key = value = query.transpose(0, 1)
            return query, key, value
        query, key = (tensor.transpose(0, 1) for tensor in (query, key))
        value = key
        return query, key, value

    def reverse_enforced_batch_as_second_dim(self, attention_output: Tensor) -> Tensor:
        if not self.should_transpose_first_two_dims:
            return attention_output
        return attention_output.transpose(1, 0)
