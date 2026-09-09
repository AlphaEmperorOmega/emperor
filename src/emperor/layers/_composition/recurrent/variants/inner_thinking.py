"""Generalized ITT-style recurrence with unchanged skipped-token residual states.

The first transition transforms every token. Later transitions gather selected
tokens, reuse the block, and scatter score/step-weighted residual contributions.
Attention blocks therefore see a selected subsequence, without a persistent KV
cache. Optional inherited controllers wrap the full reconstructed transition.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from emperor.layers._composition.recurrent.config import InnerThinkingRecurrentConfig
from emperor.layers._composition.recurrent.runtime.execution import (
    PreparedRecurrentTransition,
)
from emperor.layers._composition.recurrent.runtime.token_selection import (
    RecurrentTokenSelection,
)
from emperor.layers._composition.recurrent.validation.inner_thinking import (
    InnerThinkingRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.standard import RecurrentLayer
from emperor.layers._state import LayerState

if TYPE_CHECKING:
    from emperor.layers._composition.recurrent.variants.standard import (
        _StandardRecurrentState,
    )


class InnerThinkingRecurrent(RecurrentLayer):
    VALIDATOR = InnerThinkingRecurrentValidator

    def __init__(
        self,
        cfg: InnerThinkingRecurrentConfig,
        overrides: InnerThinkingRecurrentConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.cfg: InnerThinkingRecurrentConfig
        self.thinking_step_scale = (
            1.0
            if self.cfg.thinking_step_scale is None
            else float(self.cfg.thinking_step_scale)
        )
        self.sampler_config = self.cfg.sampler_config

        self.samplers = nn.ModuleList(
            self.sampler_config.build_with_router_input_dim(self.output_dim)
            for _ in range(self.max_steps - 1)
        )
        self.thinking_step_weights = nn.Parameter(
            torch.ones(self.max_steps, self.output_dim)
        )

    def _prepare_recurrent_transition(
        self, recurrent_state: _StandardRecurrentState, *, tracks_gradients: bool
    ) -> PreparedRecurrentTransition:
        prepared = super()._prepare_recurrent_transition(
            recurrent_state, tracks_gradients=tracks_gradients
        )
        return replace(
            prepared,
            run_transition=partial(
                self.__run_thinking_transition,
                transition_index=recurrent_state.transition_index,
                previous_hidden=recurrent_state.hidden,
            ),
        )

    def __run_thinking_transition(
        self, state: LayerState, *, transition_index: int, previous_hidden: Tensor
    ) -> LayerState:
        transition_input = state.hidden
        self.VALIDATOR.validate_block_layout(self.block_model, transition_input)
        if transition_index == 0:
            return self.__run_initial_thinking_transition(state, transition_input)
        return self.__run_selected_thinking_transition(
            state,
            transition_input,
            transition_index=transition_index,
            previous_hidden=previous_hidden,
        )

    def __run_initial_thinking_transition(
        self, state: LayerState, transition_input: Tensor
    ) -> LayerState:
        output_state = self.block_model(state)
        self.VALIDATOR.validate_transition_output(
            output_state, transition_input, expected_feature_dim=self.output_dim
        )
        return replace(
            output_state,
            hidden=output_state.hidden
            * self.thinking_step_weights[0].to(output_state.hidden),
        )

    def __run_selected_thinking_transition(
        self,
        state: LayerState,
        transition_input: Tensor,
        *,
        transition_index: int,
        previous_hidden: Tensor,
    ) -> LayerState:
        padding_mask = getattr(state, "target_key_padding_mask", None)
        if padding_mask is None:
            padding_mask = getattr(state, "key_padding_mask", None)
        token_sample = self.samplers[transition_index - 1](
            transition_input, padding_mask
        )
        selection = RecurrentTokenSelection(
            token_sample.indices, token_sample.valid, token_sample.sequence_length
        )
        selected_state = selection.select_state(
            state, attention_heads=self.__attention_heads(state)
        )
        selected_input = selected_state.hidden
        output_state = self.block_model(selected_state)
        self.VALIDATOR.validate_transition_output(
            output_state, selected_input, expected_feature_dim=self.output_dim
        )
        selected_token_weights = token_sample.weights.unsqueeze(-1)
        step_weight = self.thinking_step_weights[transition_index].to(
            output_state.hidden
        )
        selected_token_update = (
            output_state.hidden
            * selected_token_weights
            * step_weight
            * self.thinking_step_scale
        )
        return replace(
            state,
            hidden=selection.scatter_update(previous_hidden, selected_token_update),
            loss=output_state.loss,
        )

    def __attention_heads(self, state: LayerState) -> int:
        if not any(
            getattr(state, name, None) is not None
            for name in (
                "attention_mask",
                "target_attention_mask",
                "cross_attention_mask",
            )
        ):
            return 1
        head_counts = {
            module.num_heads
            for module in self.block_model.modules()
            if hasattr(module, "num_heads")
        }
        self.VALIDATOR.validate_attention_heads(head_counts)
        return next(iter(head_counts), 1)
