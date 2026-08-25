from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.embedding.contextual._validation import CausalPrefixKernelValidator
from emperor.linears import LinearLayerConfig
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.embedding.contextual._config import CausalPrefixKernelConfig


class CausalPrefixKernel(Module):
    VALIDATOR = CausalPrefixKernelValidator

    def __init__(
        self,
        cfg: CausalPrefixKernelConfig,
        overrides: CausalPrefixKernelConfig | None = None,
    ) -> None:
        self.VALIDATOR.validate_config_type(cfg)
        self.VALIDATOR.validate_overrides_type(overrides)
        super().__init__()
        self.cfg = self.__resolve_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.hidden_dim = self.cfg.hidden_dim
        self.kernel_dim = self.cfg.kernel_dim
        self.query_projection = LinearLayerConfig(
            input_dim=self.hidden_dim,
            output_dim=self.kernel_dim,
            bias_flag=False,
        ).build()
        self.key_projection = LinearLayerConfig(
            input_dim=self.hidden_dim,
            output_dim=self.kernel_dim,
            bias_flag=False,
        ).build()
        self.value_projection = LinearLayerConfig(
            input_dim=self.hidden_dim,
            output_dim=self.kernel_dim,
            bias_flag=False,
        ).build()
        self.output_projection = LinearLayerConfig(
            input_dim=self.kernel_dim,
            output_dim=self.hidden_dim,
            bias_flag=False,
        ).build()
        self.relative_position = self.cfg.relative_position_config.build()

    @staticmethod
    def __resolve_config(
        cfg: CausalPrefixKernelConfig,
        overrides: CausalPrefixKernelConfig | None,
    ) -> CausalPrefixKernelConfig:
        if overrides is None:
            return cfg
        resolved = copy.deepcopy(cfg)
        if overrides.hidden_dim is not None:
            resolved.hidden_dim = overrides.hidden_dim
        if overrides.kernel_dim != 32:
            resolved.kernel_dim = overrides.kernel_dim
        if overrides.relative_position_config is not None:
            resolved.relative_position_config = overrides.relative_position_config
        return resolved

    def forward(self, inputs: Tensor, attention_mask: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(self, inputs, attention_mask)
        query = self.query_projection(inputs)
        key = self.key_projection(inputs)
        value = self.value_projection(inputs)
        scaled_query = query * self.kernel_dim**-0.5
        scores = torch.matmul(scaled_query, key.transpose(-2, -1))
        sequence_length = inputs.size(1)
        relative_bias = self.relative_position(
            scaled_query.unsqueeze(1),
            sequence_length,
        ).squeeze(1)
        scores = scores + relative_bias

        causal_visibility = torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=inputs.device,
        ).tril()
        visible_keys = causal_visibility.unsqueeze(0) & attention_mask.unsqueeze(1)
        masked_scores = scores.masked_fill(~visible_keys, -torch.inf)
        fully_masked_rows = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)
        safe_scores = torch.where(
            fully_masked_rows,
            torch.zeros_like(masked_scores),
            masked_scores,
        )
        normalized_weights = F.softmax(safe_scores, dim=-1)
        normalized_weights = torch.where(
            fully_masked_rows,
            torch.zeros_like(normalized_weights),
            normalized_weights,
        )
        mixed_values = torch.matmul(normalized_weights, value)
        output = self.output_projection(mixed_values)
        query_mask = attention_mask.unsqueeze(-1)
        return output * query_mask
