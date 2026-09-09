"""Gather/scatter token states while preserving sequence order and attention masks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

import torch
from torch import Tensor

from emperor.layers._composition.recurrent.validation.inner_thinking import (
    InnerThinkingRecurrentValidator,
)
from emperor.layers._state import LayerState


@dataclass(frozen=True)
class RecurrentTokenSelection:
    VALIDATOR = InnerThinkingRecurrentValidator
    indices: Tensor
    valid: Tensor
    sequence_length: int

    def gather_hidden(self, hidden: Tensor) -> Tensor:
        return hidden.gather(
            -2, self.indices.unsqueeze(-1).expand(*self.indices.shape, hidden.shape[-1])
        )

    def scatter_update(self, previous_hidden: Tensor, update: Tensor) -> Tensor:
        valid_token_update = torch.where(self.valid.unsqueeze(-1), update, 0)
        feature_indices = self.indices.unsqueeze(-1).expand_as(valid_token_update)
        return previous_hidden.scatter_add(-2, feature_indices, valid_token_update)

    def select_state(self, state: LayerState, *, attention_heads: int) -> LayerState:
        updates = {"hidden": self.gather_hidden(state.hidden)}
        updates.update(self.__select_padding_masks(state))
        updates.update(self.__select_attention_masks(state, attention_heads))
        return replace(state, **updates)

    def __select_padding_masks(self, state: LayerState) -> dict[str, Tensor]:
        selected_masks = {}
        for name in ("key_padding_mask", "target_key_padding_mask"):
            if hasattr(state, name):
                mask = getattr(state, name)
                if mask is not None:
                    selected_mask = mask.gather(-1, self.indices)
                    invalid_token_fill = (
                        True if mask.dtype == torch.bool else -torch.inf
                    )
                    selected_masks[name] = selected_mask.masked_fill(
                        ~self.valid, invalid_token_fill
                    )
        return selected_masks

    def __select_attention_masks(
        self, state: LayerState, attention_heads: int
    ) -> dict[str, Tensor]:
        selected_masks = {}
        for name in ("attention_mask", "target_attention_mask", "cross_attention_mask"):
            mask = getattr(state, name, None)
            if mask is not None:
                selected_masks[name] = self.__select_attention_mask(
                    mask,
                    attention_heads=attention_heads,
                    select_keys=name != "cross_attention_mask",
                )
        return selected_masks

    def __select_attention_mask(
        self, mask: Tensor, *, attention_heads: int, select_keys: bool
    ) -> Tensor:
        sequence_batch_size = prod(self.indices.shape[:-1])
        selected_token_count = self.indices.shape[-1]
        sequence_indices = self.indices.reshape(
            sequence_batch_size, selected_token_count
        )
        attention_branch_count = sequence_batch_size * attention_heads
        self.VALIDATOR.validate_attention_mask(
            mask,
            self.indices,
            self.sequence_length,
            select_keys=select_keys,
            branch_count=attention_branch_count,
        )
        if mask.ndim == 2:
            expanded_mask = mask.unsqueeze(0).expand(attention_branch_count, -1, -1)
        else:
            expanded_mask = mask.expand(attention_branch_count, -1, -1)
        attention_branch_indices = sequence_indices.repeat_interleave(
            attention_heads, dim=0
        )
        selected_mask = expanded_mask.gather(
            1, attention_branch_indices.unsqueeze(-1).expand(-1, -1, mask.shape[-1])
        )
        if select_keys:
            selected_mask = selected_mask.gather(
                2,
                attention_branch_indices.unsqueeze(1).expand(
                    -1, selected_token_count, -1
                ),
            )
        if self.indices.ndim == 1 and mask.ndim == 2:
            return selected_mask[0]
        return selected_mask
