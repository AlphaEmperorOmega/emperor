from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from emperor.layers._config import AttentionResidualConfig
from emperor.layers._validation import AttentionResidualValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._composition.residual import ResidualConnection


@dataclass(slots=True)
class AttentionResidualState:
    """Forward-local sources mixed once per physical residual-depth execution.

    The mutable history is intentionally scoped to one eager stack forward.
    Activation-checkpoint recomputation needs a future functional-state adapter.
    """

    VALIDATOR: ClassVar[type[AttentionResidualValidator]] = AttentionResidualValidator

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


class AttentionResidual(Module):
    """Learned softmax routing across raw residual-depth sources."""

    VALIDATOR = AttentionResidualValidator
    DEFAULT_BLOCK_SIZE = 1
    DEFAULT_RMS_NORM_EPSILON = 1e-6

    def __init__(
        self,
        cfg: AttentionResidualConfig,
        overrides: AttentionResidualConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.VALIDATOR.validate_positive_integer(
            self.cfg.residual_dim,
            name="residual_dim",
        )
        self.residual_dim = cast(int, self.cfg.residual_dim)
        configured_block_size = (
            self.DEFAULT_BLOCK_SIZE
            if self.cfg.block_size is None
            else self.cfg.block_size
        )
        self.VALIDATOR.validate_positive_integer(
            configured_block_size,
            name="block_size",
        )
        self.block_size = configured_block_size
        configured_rms_norm_epsilon = (
            self.DEFAULT_RMS_NORM_EPSILON
            if self.cfg.rms_norm_epsilon is None
            else self.cfg.rms_norm_epsilon
        )
        self.VALIDATOR.validate_finite_positive_number(
            configured_rms_norm_epsilon,
            name="rms_norm_epsilon",
        )
        self.rms_norm_epsilon = float(configured_rms_norm_epsilon)
        self.query = nn.Parameter(torch.zeros(self.residual_dim))
        self.key_norm = nn.RMSNorm(
            self.residual_dim,
            eps=self.rms_norm_epsilon,
            elementwise_affine=True,
        )

    def new_state(self, initial_source: Tensor) -> AttentionResidualState:
        self.VALIDATOR.validate_source(
            initial_source,
            residual_dim=self.residual_dim,
        )
        return AttentionResidualState(initial_source, block_size=self.block_size)

    def forward(
        self,
        current: Tensor,
        state: AttentionResidualState,
    ) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(
            current,
            state,
            residual_dim=self.residual_dim,
            block_size=self.block_size,
        )
        state.append(current)
        values = torch.stack(state.sources, dim=0)
        accumulator_dtype = (
            torch.float32
            if values.dtype in (torch.float16, torch.bfloat16)
            else values.dtype
        )
        accumulator_values = values.to(dtype=accumulator_dtype)
        keys = F.rms_norm(
            accumulator_values,
            normalized_shape=(self.residual_dim,),
            weight=self.key_norm.weight.to(dtype=accumulator_dtype),
            eps=self.rms_norm_epsilon,
        )
        logits = torch.sum(
            keys * self.query.to(dtype=accumulator_dtype),
            dim=-1,
        )
        depth_weights = torch.softmax(logits, dim=0)
        mixed = torch.sum(
            depth_weights.unsqueeze(-1) * accumulator_values,
            dim=0,
        )
        return mixed.to(dtype=values.dtype)


class AttentionResidualOption:
    @staticmethod
    def forward(
        connection: ResidualConnection,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: AttentionResidualState | None = None,
    ) -> Tensor:
        connection.VALIDATOR.validate_attention_residual_state(residual_state)
        attention_residual = cast(AttentionResidual, connection.attention_residual)
        return attention_residual(
            current,
            cast(AttentionResidualState, residual_state),
        )
