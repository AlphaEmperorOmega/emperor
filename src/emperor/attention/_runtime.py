"""Private attention runtime value objects."""

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
    source_extension_count: int = 0

    def branch_count(self, num_heads: int) -> int:
        return self.batch_size * num_heads

    def with_source_extension(self, count: int = 1) -> AttentionRuntimeShape:
        return replace(
            self,
            source_sequence_length=self.source_sequence_length + count,
            source_extension_count=self.source_extension_count + count,
        )

    @property
    def real_source_sequence_length(self) -> int:
        return self.source_sequence_length - self.source_extension_count


__all__ = [
    "AttentionRuntimeShape",
    "AttentionMasks",
    "QKV",
]
