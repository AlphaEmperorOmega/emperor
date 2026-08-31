from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from emperor.layers._composition.residual.base import (
    ResidualConnectionAbstract,
    ResidualRuntimeRequirement,
    ResidualStackRequirements,
    ResidualState,
    ResidualStateLifecycle,
)
from emperor.layers._composition.residual.config import AttentionResidualConfig
from emperor.layers._composition.residual.validation import (
    ResidualConnectionValidator,
)

if TYPE_CHECKING:
    from emperor.layers._row_layout import RowLayout
    from emperor.layers._state import LayerState


@dataclass(slots=True)
class AttentionResidualState(ResidualState):
    """Forward-local sources mixed once per physical residual-depth execution."""

    VALIDATOR: ClassVar[type[ResidualConnectionValidator]] = ResidualConnectionValidator

    initial_source: Tensor
    block_size: int
    _completed_blocks: list[Tensor] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _partial_block: Tensor | None = field(default=None, init=False, repr=False)
    _partial_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.VALIDATOR.validate_positive_integer(
            self.block_size,
            name="block_size",
        )

    @property
    def sources(self) -> tuple[Tensor, ...]:
        partial_sources = () if self._partial_block is None else (self._partial_block,)
        return (self.initial_source, *self._completed_blocks, *partial_sources)

    def append(self, raw_output: Tensor) -> None:
        self._partial_block = (
            raw_output
            if self._partial_block is None
            else self._partial_block + raw_output
        )
        self._partial_count += 1
        if self._partial_count == self.block_size:
            self._completed_blocks.append(self._partial_block)
            self._partial_block = None
            self._partial_count = 0

    def fork(self) -> AttentionResidualState:
        forked = AttentionResidualState(
            self.initial_source,
            block_size=self.block_size,
        )
        forked._completed_blocks = list(self._completed_blocks)
        forked._partial_block = self._partial_block
        forked._partial_count = self._partial_count
        return forked


@dataclass(frozen=True, slots=True)
class _AttentionResidualStateLifecycle(ResidualStateLifecycle):
    residual_dim: int
    block_size: int
    validator: type[ResidualConnectionValidator]

    def create_state(self, initial_source: Tensor) -> AttentionResidualState:
        self.validator.validate_source(
            initial_source,
            residual_dim=self.residual_dim,
        )
        return AttentionResidualState(
            initial_source,
            block_size=self.block_size,
        )


class AttentionResidual(ResidualConnectionAbstract):
    """Learned softmax routing across raw residual-depth sources."""

    DEFAULT_BLOCK_SIZE = 1
    DEFAULT_RMS_NORM_EPSILON = 1e-6
    RUNTIME_REQUIREMENTS = frozenset(
        {
            ResidualRuntimeRequirement.FORWARD_LOCAL_STATE,
            ResidualRuntimeRequirement.DEPTH_SPECIFIC_CONNECTIONS,
        }
    )
    STACK_REQUIREMENTS = ResidualStackRequirements(
        requires_uniform_dimensions=True,
        requires_output_postprocessing=True,
        allows_halting=False,
    )

    def __init__(
        self,
        cfg: AttentionResidualConfig,
        overrides: AttentionResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.__initialize_from_config()
        self.query = nn.Parameter(torch.zeros(self.residual_dim))
        self.key_norm = self.__build_key_norm()
        self.__residual_state_lifecycle = self.__build_residual_state_lifecycle()

    def __initialize_from_config(self) -> None:
        self.block_size = self.__resolve_block_size(self.cfg.block_size)
        self.rms_norm_epsilon = self.__resolve_rms_norm_epsilon(
            self.cfg.rms_norm_epsilon
        )

    @classmethod
    def __resolve_block_size(cls, configured_block_size: int | None) -> int:
        return (
            cls.DEFAULT_BLOCK_SIZE
            if configured_block_size is None
            else configured_block_size
        )

    @classmethod
    def __resolve_rms_norm_epsilon(
        cls,
        configured_rms_norm_epsilon: float | None,
    ) -> float:
        configured_rms_norm_epsilon = (
            cls.DEFAULT_RMS_NORM_EPSILON
            if configured_rms_norm_epsilon is None
            else configured_rms_norm_epsilon
        )
        return float(configured_rms_norm_epsilon)

    def __build_key_norm(self) -> nn.RMSNorm:
        return nn.RMSNorm(
            self.residual_dim,
            eps=self.rms_norm_epsilon,
            elementwise_affine=True,
        )

    def __build_residual_state_lifecycle(
        self,
    ) -> _AttentionResidualStateLifecycle:
        return _AttentionResidualStateLifecycle(
            residual_dim=self.residual_dim,
            block_size=self.block_size,
            validator=self.VALIDATOR,
        )

    @property
    def residual_state_lifecycle(self) -> ResidualStateLifecycle:
        return self.__residual_state_lifecycle

    def apply_to_layer_state(
        self,
        state: LayerState,
        previous: Tensor,
    ) -> LayerState:
        if state.residual_state is None:
            created_residual_state = self.new_state(previous)
            state.residual_state = self.VALIDATOR.validate_created_attention_state(
                created_residual_state,
                block_size=self.block_size,
            )
        return super().apply_to_layer_state(state, previous)

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: ResidualState | None = None,
        row_layout: RowLayout | None = None,
    ) -> Tensor:
        attention_state = self.__validate_attention_forward_inputs(
            current, residual_state
        )
        residual_sources = self.__append_and_stack_residual_sources(
            attention_state, current
        )
        accumulator_sources = self.__convert_to_accumulator_precision(
            residual_sources,
        )
        normalized_source_keys = self.__normalize_residual_source_keys(
            accumulator_sources,
        )
        depth_weights = self.__calculate_residual_depth_weights(normalized_source_keys)
        mixed_residual_sources = self.__mix_depth_weighted_residual_sources(
            accumulator_sources, depth_weights
        )
        return mixed_residual_sources.to(dtype=residual_sources.dtype)

    def __validate_attention_forward_inputs(
        self,
        current: Tensor,
        residual_state: ResidualState | None,
    ) -> AttentionResidualState:
        return self.VALIDATOR.validate_attention_forward_inputs(
            current,
            residual_state,
            residual_dim=self.residual_dim,
            block_size=self.block_size,
        )

    @staticmethod
    def __append_and_stack_residual_sources(
        attention_state: AttentionResidualState,
        current: Tensor,
    ) -> Tensor:
        attention_state.append(current)
        return torch.stack(attention_state.sources, dim=0)

    def __convert_to_accumulator_precision(
        self,
        residual_sources: Tensor,
    ) -> Tensor:
        accumulator_dtype = self.__resolve_accumulator_dtype(
            residual_sources.dtype,
        )
        return residual_sources.to(dtype=accumulator_dtype)

    @staticmethod
    def __resolve_accumulator_dtype(values_dtype: torch.dtype) -> torch.dtype:
        if values_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        else:
            return values_dtype

    def __normalize_residual_source_keys(
        self,
        accumulator_sources: Tensor,
    ) -> Tensor:
        return F.rms_norm(
            accumulator_sources,
            normalized_shape=(self.residual_dim,),
            weight=self.key_norm.weight.to(dtype=accumulator_sources.dtype),
            eps=self.rms_norm_epsilon,
        )

    def __calculate_residual_depth_weights(
        self,
        normalized_source_keys: Tensor,
    ) -> Tensor:
        accumulator_query = self.query.to(dtype=normalized_source_keys.dtype)
        query_weighted_source_keys = normalized_source_keys * accumulator_query
        depth_attention_logits = torch.sum(query_weighted_source_keys, dim=-1)
        return torch.softmax(depth_attention_logits, dim=0)

    @staticmethod
    def __mix_depth_weighted_residual_sources(
        accumulator_sources: Tensor,
        depth_weights: Tensor,
    ) -> Tensor:
        feature_broadcast_depth_weights = depth_weights.unsqueeze(-1)
        depth_weighted_residual_sources = (
            feature_broadcast_depth_weights * accumulator_sources
        )
        return torch.sum(
            depth_weighted_residual_sources,
            dim=0,
        )
