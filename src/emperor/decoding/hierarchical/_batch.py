from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from ._records import END_OF_DOCUMENT, END_OF_TOKEN


@dataclass(frozen=True)
class HierarchicalLanguageModelBatch:
    """Teacher forcing, packed in the same row-major order as decoder logits.

    Each context slot predicts its already-shifted target. Literal target bytes
    are both decoder prefixes and the first labels; append EOW to the labels
    only. A None target contributes one EOS label and an empty byte prefix.
    """

    context_texts: tuple[tuple[str, ...], ...]
    byte_prefix_ids: Tensor
    byte_lengths: Tensor
    labels: Tensor
    attention_mask: Tensor
    bos_mask: Tensor
    byte_count: int

    def to(self, device, **kwargs):
        return replace(
            self,
            **{
                name: getattr(self, name).to(device=device, **kwargs)
                for name in (
                    "byte_prefix_ids",
                    "byte_lengths",
                    "labels",
                    "attention_mask",
                    "bos_mask",
                )
            },
        )

    @classmethod
    def collate(cls, examples: Sequence[dict]):
        import torch

        if not examples:
            raise ValueError("collation requires at least one example")
        maximum_tokens = max(len(row["context_texts"]) for row in examples)
        if not maximum_tokens:
            raise ValueError("examples require at least one context/target pair")
        encoded = []
        maximum_bytes = 0
        for row in examples:
            contexts, targets, bos = (
                row["context_texts"],
                row["target_texts"],
                row["bos_mask"],
            )
            if (
                not contexts
                or len(contexts) != len(targets)
                or len(contexts) != len(bos)
            ):
                raise ValueError(
                    "context_texts, target_texts, and bos_mask must have equal nonzero lengths"
                )
            if any(not isinstance(text, str) for text in contexts) or any(
                type(value) is not bool for value in bos
            ):
                raise TypeError("contexts must be strings and BOS values must be bools")
            try:
                values = [
                    None if text is None else text.encode("utf-8") for text in targets
                ]
                for text in contexts:
                    text.encode("utf-8")
            except (AttributeError, UnicodeEncodeError) as error:
                raise ValueError(
                    "contexts and targets must be valid UTF-8 strings (None targets mean EOS)"
                ) from error
            encoded.append(values)
            maximum_bytes = max(
                maximum_bytes,
                max((len(value) for value in values if value is not None), default=0),
            )
        shape = (len(examples), maximum_tokens)
        prefixes = torch.zeros((*shape, maximum_bytes), dtype=torch.long)
        lengths = torch.zeros(shape, dtype=torch.long)
        mask = torch.zeros(shape, dtype=torch.bool)
        bos_mask = torch.zeros_like(mask)
        contexts = []
        labels = []
        byte_count = 0
        for row_index, (row, values) in enumerate(zip(examples, encoded, strict=True)):
            count = len(values)
            contexts.append(
                tuple(row["context_texts"]) + ("",) * (maximum_tokens - count)
            )
            mask[row_index, :count] = True
            bos_mask[row_index, :count] = torch.tensor(
                row["bos_mask"], dtype=torch.bool
            )
            for column, value in enumerate(values):
                if value is None:
                    labels.append(END_OF_DOCUMENT)
                    continue
                size = len(value)
                lengths[row_index, column] = size
                prefixes[row_index, column, :size] = torch.tensor(
                    list(value), dtype=torch.long
                )
                labels.extend(value)
                labels.append(END_OF_TOKEN)
                byte_count += size
        return cls(
            tuple(contexts),
            prefixes,
            lengths,
            torch.tensor(labels, dtype=torch.long),
            mask,
            bos_mask,
            byte_count,
        )
