from __future__ import annotations

from dataclasses import dataclass, replace

from torch import Tensor


@dataclass(frozen=True, eq=False, kw_only=True)
class QKV:
    query: Tensor
    key: Tensor
    value: Tensor


@dataclass(frozen=True, eq=False, kw_only=True)
class AttentionMasks:
    key_padding_mask: Tensor | None = None
    attention_mask: Tensor | None = None


@dataclass(frozen=True)
class AttentionRuntimeShape:
    batch_size: int
    target_sequence_length: int
    source_sequence_length: int
    input_was_batched: bool = True
    input_was_batch_first: bool = False

    def branch_count(self, num_heads: int, multiplier: int = 1) -> int:
        return self.batch_size * num_heads * multiplier

    def with_source_extension(self, count: int = 1) -> AttentionRuntimeShape:
        return replace(
            self,
            source_sequence_length=self.source_sequence_length + count,
        )


__all__ = [
    "AttentionRuntimeShape",
    "AttentionMasks",
    "QKV",
]
